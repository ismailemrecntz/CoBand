"""CoBand: RGB-guided complementary frequency band routing for RGB-thermal fusion.

Network layout:
    two ConvNeXt V2 encoders (RGB, single-channel thermal)
    stages 1-2 (stride 4, 8):   CoBand module  (thermal band split + RGB-conditioned band gate)
    stages 3-4 (stride 16, 32): lightweight semantic fusion (LSF)
    PANet decoder with global context gate and deep supervision

The ablation variants of the paper are built by the same class:
    stage12='fgf'           FGF module: same band split, gate sees thermal statistics only
    deep='enhanced'         heavier enhanced semantic fusion block at stages 3-4
    fusion_type='add' | 'concat' | 'lsf_all'   one simple fusion at all four stages
    modality='rgb_only' | 'ir_only'            single-encoder references, no fusion

Note: module attribute names (e.g. ``lowpass``, ``freq_gate``, ``rgb_context``, ``ir_encoder``) define the
parameter names in saved checkpoints and must not be renamed, otherwise released weights no longer load.
"""
from typing import List, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================================================
# Building blocks of the CoBand module
# =====================================================================================================
class ScalarConfidenceGate(nn.Module):
    """Scalar confidence gamma in (0, 1) for the RGB feature (global, not per channel)."""
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gate(x)


class SafeResidualFusion(nn.Module):
    """F = F_rgb + R([gamma * F_rgb ; T_hat]).

    gamma only scales the copy of F_rgb that enters the refinement block, never the identity path, and the
    refinement block ends without an activation, so the correction Delta can be negative.
    """
    def __init__(self, channels):
        super().__init__()
        self.rgb_gate = ScalarConfidenceGate(channels)
        self.refine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, rgb_feat, ir_final):
        score = self.rgb_gate(rgb_feat)
        fused_input = torch.cat([rgb_feat * score, ir_final], dim=1)
        delta = self.refine(fused_input)
        return rgb_feat + delta


class SpatialAttention(nn.Module):
    """Spatial attention from channel-wise average and max pooling."""
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        mask = self.conv(x_cat)
        return self.sigmoid(mask)


class GaussianLowPass(nn.Module):
    """Fixed, normalized depthwise Gaussian low-pass filter G_sigma. Not trained."""
    def __init__(self, channels, k=7, sigma=2.0):
        super().__init__()
        grid = torch.arange(k) - k // 2
        gauss = torch.exp(-grid ** 2 / (2 * sigma ** 2))
        kernel = gauss[:, None] * gauss[None, :]
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel[None, None, :, :].repeat(channels, 1, 1, 1))
        self.groups = channels
        self.pad = k // 2

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=self.pad, groups=self.groups)


class LearnedLowPass(GaussianLowPass):
    """Filter ablation ('learned filter'): same initialization as GaussianLowPass, but the per-channel kernel
    is a free parameter (no normalization, no constraint). The high band is still T - G * T."""
    def __init__(self, channels, k=7, sigma=2.0):
        super().__init__(channels, k=k, sigma=sigma)
        kernel = self.kernel.clone()
        del self.kernel                                   # buffer -> parameter with the same name ('kernel')
        self.kernel = nn.Parameter(kernel)


class CoBand(nn.Module):
    """CoBand module (stages 1-2).

    (a) T = phi(F_th); T_L = G_sigma * T, T_H = T - T_L; each band refined by its own spatial attention
    (b) alpha = sigmoid(W_g [s ; c]) with band statistics s and RGB context c (one weight per channel)
        T_hat = alpha * T_L + (1 - alpha) * T_H
    (c) safe residual fusion with the RGB feature
    """
    def __init__(self, rgb_c, ir_c, kernel_size=7, reduction=16, sigma=2.0, learnable_filter=False):
        super().__init__()

        self.ir_proj = nn.Sequential(                                     # phi
            nn.Conv2d(ir_c, rgb_c, 1, bias=False),
            nn.BatchNorm2d(rgb_c),
            nn.GELU()
        )
        if learnable_filter:
            self.lowpass = LearnedLowPass(rgb_c, k=kernel_size, sigma=sigma)
        else:
            self.lowpass = GaussianLowPass(rgb_c, k=kernel_size, sigma=sigma)

        self.sa_low = SpatialAttention(kernel_size=7)                     # low band: large structures
        self.sa_high = SpatialAttention(kernel_size=3)                    # high band: thin structures

        hidden = max(rgb_c // reduction, 16)
        self.rgb_context = nn.Sequential(                                 # c
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rgb_c, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, rgb_c, 1, bias=False)
        )
        self.freq_gate = nn.Sequential(                                   # W_g, input [s ; c]
            nn.Conv2d(rgb_c * 3, rgb_c, 1, bias=False),
            nn.Sigmoid()
        )
        self.fusion_block = SafeResidualFusion(rgb_c)

    def decompose(self, x):
        x_low = self.lowpass(x)
        x_high = x - x_low
        return x_low, x_high

    def forward(self, rgb_feat, ir_feat):
        ir_feat = self.ir_proj(ir_feat)

        ir_low, ir_high = self.decompose(ir_feat)
        mask_low = self.sa_low(ir_low)
        mask_high = self.sa_high(ir_high)
        ir_low_refined = ir_low * mask_low
        ir_high_refined = ir_high * mask_high

        rgb_ctx = self.rgb_context(rgb_feat)                              # [B, C, 1, 1]
        ir_mixed = torch.cat([ir_low_refined, ir_high_refined], dim=1)
        ir_global = F.adaptive_avg_pool2d(torch.abs(ir_mixed), 1)         # s: [B, 2C, 1, 1]
        gate_input = torch.cat([ir_global, rgb_ctx], dim=1)               # [s ; c]: [B, 3C, 1, 1]
        alpha = self.freq_gate(gate_input)                                # [B, C, 1, 1], weight of the low band

        ir_final = alpha * ir_low_refined + (1 - alpha) * ir_high_refined
        return self.fusion_block(rgb_feat, ir_final)


class FGF(nn.Module):
    """Frequency-guided fusion (FGF, our earlier workshop paper): same band split as CoBand, but the band gate
    sees only the thermal band statistics (no RGB context). Used as baseline and in the ablations ('FGF+...')."""
    def __init__(self, rgb_c, ir_c):
        super().__init__()
        self.ir_proj = nn.Sequential(
            nn.Conv2d(ir_c, rgb_c, 1, bias=False),
            nn.BatchNorm2d(rgb_c),
            nn.GELU()
        )
        self.lowpass = GaussianLowPass(rgb_c)                            # 7x7, sigma = 2 at every stage
        self.sa_low = SpatialAttention(kernel_size=7)
        self.sa_high = SpatialAttention(kernel_size=3)
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rgb_c * 2, rgb_c, 1, bias=False),
            nn.Sigmoid()
        )
        self.fusion_block = SafeResidualFusion(rgb_c)

    def decompose(self, x):
        x_low = self.lowpass(x)
        x_high = x - x_low
        return x_low, x_high

    def forward(self, rgb_feat, ir_feat):
        ir_feat = self.ir_proj(ir_feat)
        ir_low, ir_high = self.decompose(ir_feat)
        mask_low = self.sa_low(ir_low)
        mask_high = self.sa_high(ir_high)
        ir_low_refined = ir_low * mask_low
        ir_high_refined = ir_high * mask_high
        ir_mixed = torch.cat([ir_low_refined, ir_high_refined], dim=1)
        alpha = self.freq_gate(torch.abs(ir_mixed))                       # [B, C, 1, 1]
        ir_final = alpha * ir_low_refined + (1 - alpha) * ir_high_refined
        return self.fusion_block(rgb_feat, ir_final)


# =====================================================================================================
# Deep-stage fusion blocks and simple fusion baselines (ablations)
# =====================================================================================================
class LightweightSemanticFusion(nn.Module):
    """LSF: channel-gated thermal residual plus depthwise-separable refinement (stages 3-4)."""
    def __init__(self, rgb_c, ir_c, reduction=16):
        super().__init__()
        self.ir_proj = nn.Sequential(
            nn.Conv2d(ir_c, rgb_c, 1, bias=False),
            nn.BatchNorm2d(rgb_c)
        )
        self.cross_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rgb_c * 2, rgb_c // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_c // reduction, rgb_c, 1, bias=False),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(rgb_c, rgb_c, 3, padding=1, groups=rgb_c, bias=False),
            nn.BatchNorm2d(rgb_c),
            nn.GELU(),
            nn.Conv2d(rgb_c, rgb_c, 1, bias=False),
            nn.BatchNorm2d(rgb_c)
        )

    def forward(self, rgb_feat, ir_feat):
        ir_proj = self.ir_proj(ir_feat)
        joint = torch.cat([rgb_feat, ir_proj], dim=1)
        gate = self.cross_gate(joint)
        fused = rgb_feat + gate * ir_proj
        refined = self.refine(fused)
        return fused + refined


class EnhancedSemanticFusion(nn.Module):
    """Enhanced semantic fusion block of FGF (ablation 'Enh.'): cross-modal channel gate, multi-scale
    depthwise convolutions, channel and spatial attention, residual refinement."""
    def __init__(self, rgb_c, ir_c, reduction=16):
        super().__init__()
        self.ir_proj = nn.Sequential(
            nn.Conv2d(ir_c, rgb_c, 1, bias=False),
            nn.BatchNorm2d(rgb_c),
            nn.GELU()
        )
        self.cross_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rgb_c * 2, rgb_c // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_c // reduction, rgb_c * 2, 1, bias=False),
        )
        self.ms_conv_3x3 = nn.Sequential(
            nn.Conv2d(rgb_c, rgb_c, 3, padding=1, groups=rgb_c, bias=False),
            nn.BatchNorm2d(rgb_c),
            nn.GELU(),
            nn.Conv2d(rgb_c, rgb_c, 1, bias=False),
            nn.BatchNorm2d(rgb_c)
        )
        self.ms_conv_5x5 = nn.Sequential(
            nn.Conv2d(rgb_c, rgb_c, 5, padding=2, groups=rgb_c, bias=False),
            nn.BatchNorm2d(rgb_c),
            nn.GELU(),
            nn.Conv2d(rgb_c, rgb_c, 1, bias=False),
            nn.BatchNorm2d(rgb_c)
        )
        self.ms_fusion = nn.Sequential(
            nn.Conv2d(rgb_c * 2, rgb_c, 1, bias=False),
            nn.BatchNorm2d(rgb_c)
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rgb_c, rgb_c // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(rgb_c // reduction, rgb_c, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(rgb_c, rgb_c, 3, padding=1, groups=rgb_c, bias=False),
            nn.BatchNorm2d(rgb_c),
            nn.GELU(),
            nn.Conv2d(rgb_c, rgb_c, 1, bias=False),
            nn.BatchNorm2d(rgb_c)
        )
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, rgb_feat, ir_feat):
        ir_proj = self.ir_proj(ir_feat)
        joint = torch.cat([rgb_feat, ir_proj], dim=1)
        gate = torch.sigmoid(self.cross_gate(joint))
        gate_rgb, gate_ir = gate.chunk(2, dim=1)
        fused = gate_rgb * rgb_feat + gate_ir * ir_proj

        feat_3x3 = self.ms_conv_3x3(fused)
        feat_5x5 = self.ms_conv_5x5(fused)
        multi_scale = self.ms_fusion(torch.cat([feat_3x3, feat_5x5], dim=1))
        multi_scale = multi_scale * self.channel_attention(multi_scale)

        max_pool = torch.max(multi_scale, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(multi_scale, dim=1, keepdim=True)
        multi_scale = multi_scale * self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1))

        refined = self.refine(multi_scale)
        output = fused + self.res_scale * refined
        if torch.is_autocast_enabled():                                   # float16 overflow protection
            output = torch.clamp(output, min=-65000.0, max=65000.0)
        return output


class AddFusion(nn.Module):
    """Ablation 'Add': project the thermal feature and add it to the RGB feature."""
    def __init__(self, rgb_c, ir_c):
        super().__init__()
        self.ir_proj = nn.Sequential(
            nn.Conv2d(ir_c, rgb_c, 1, bias=False),
            nn.BatchNorm2d(rgb_c),
        )

    def forward(self, rgb_feat, ir_feat):
        return rgb_feat + self.ir_proj(ir_feat)


class ConcatFusion(nn.Module):
    """Ablation 'Concat': concatenate and reduce with a 1x1 convolution."""
    def __init__(self, rgb_c, ir_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(rgb_c + ir_c, rgb_c, 1, bias=False),
            nn.BatchNorm2d(rgb_c),
            nn.GELU(),
        )

    def forward(self, rgb_feat, ir_feat):
        return self.conv(torch.cat([rgb_feat, ir_feat], dim=1))


# =====================================================================================================
# PANet decoder
# =====================================================================================================
class PANetDecoder(nn.Module):
    """Top-down and bottom-up paths, global context gate (0.5 + 0.5 * sigmoid), multi-level aggregation,
    auxiliary head on the middle level and one 1x1 deep-supervision head per level (training only)."""

    def __init__(self, in_channels: List[int], num_classes: int,
                 output_resolution: Tuple[int, int] = (480, 640), deep_supervision: bool = True):
        super().__init__()
        assert len(in_channels) >= 2, "At least two feature levels are required"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.output_resolution = output_resolution
        self.deep_supervision = deep_supervision
        L = len(in_channels)
        self.num_levels = L
        self.fpn_channels = in_channels[0]

        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels[i], self.fpn_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.fpn_channels),
                nn.ReLU(inplace=True)
            )
            for i in range(L)
        ])
        self.td_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.fpn_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(L - 1)
        ])
        self.bu_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(self.fpn_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(L - 1)
        ])
        self.bu_merge_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.fpn_channels * 2, self.fpn_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.fpn_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(L - 1)
        ])
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels[-1], self.fpn_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.agg_conv = nn.Sequential(
            nn.Conv2d(self.fpn_channels * L, self.fpn_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.fpn_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fpn_channels * 2, self.fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.fpn_channels),
            nn.ReLU(inplace=True)
        )
        self.seg_head = nn.Sequential(
            nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.fpn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(self.fpn_channels, num_classes, kernel_size=1)
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(self.fpn_channels, self.fpn_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.fpn_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fpn_channels // 2, num_classes, kernel_size=1)
        )
        if self.deep_supervision:
            self.deep_heads = nn.ModuleList([
                nn.Conv2d(self.fpn_channels, num_classes, kernel_size=1)
                for _ in range(L)
            ])

    def forward(self, features: List[torch.Tensor]):
        L = len(features)
        assert L == self.num_levels

        laterals = [self.lateral_convs[i](features[i]) for i in range(L)]

        # top-down path
        td_features = [None] * L
        td_features[-1] = laterals[-1]
        for i in range(L - 2, -1, -1):
            up = F.interpolate(td_features[i + 1], size=laterals[i].shape[2:], mode='bilinear', align_corners=False)
            td_features[i] = self.td_convs[i](laterals[i] + up)

        # bottom-up path
        bu_features = [None] * L
        bu_features[0] = td_features[0]
        for i in range(1, L):
            down = self.bu_convs[i - 1](bu_features[i - 1])
            if down.shape[2:] != td_features[i].shape[2:]:
                down = F.interpolate(down, size=td_features[i].shape[2:], mode='bilinear', align_corners=False)
            merged = torch.cat([td_features[i], down], dim=1)
            bu_features[i] = self.bu_merge_convs[i - 1](merged)

        # global context gate
        global_ctx = self.global_context(features[-1])
        bu_features_ctx = [f * (0.5 + 0.5 * global_ctx) for f in bu_features]

        # multi-level aggregation at stride 4
        target_size = bu_features_ctx[0].shape[2:]
        upsampled = [bu_features_ctx[0]]
        for i in range(1, L):
            upsampled.append(F.interpolate(bu_features_ctx[i], size=target_size, mode='bilinear', align_corners=False))
        aggregated = self.agg_conv(torch.cat(upsampled, dim=1))

        main_logits = self.seg_head(aggregated)
        main_logits = F.interpolate(main_logits, size=self.output_resolution, mode='bilinear', align_corners=False)

        mid = L // 2
        aux_logits = self.aux_head(bu_features[mid])
        aux_logits = F.interpolate(aux_logits, size=self.output_resolution, mode='bilinear', align_corners=False)

        if self.deep_supervision and self.training:
            deep_outputs = []
            for i in range(L):
                deep_out = self.deep_heads[i](bu_features[i])
                deep_outputs.append(F.interpolate(deep_out, size=self.output_resolution, mode='bilinear',
                                                  align_corners=False))
            return main_logits, {'aux': aux_logits, 'deep': deep_outputs}

        return main_logits, aux_logits


# =====================================================================================================
# Full network
# =====================================================================================================
class CoBandNet(nn.Module):
    """RGB-thermal segmentation network with CoBand fusion.

    Args:
        rgb_arch, ir_arch: timm model names of the two encoders (ConvNeXt V2 Nano / Tiny / Base).
        context_dim: channel widths of the four encoder stages (e.g. [80, 160, 320, 640] for Nano).
        output_resolution: (H, W) of the logits (equal to the input resolution).
        deep_supervision: add one 1x1 head per decoder level during training.
        stage12: 'coband' (paper) or 'fgf' (gate without RGB context) at stages 1-2.
        deep: 'lsf' (paper) or 'enhanced' at stages 3-4.
        fusion_type: 'default' (stage12 / deep as above) or one simple fusion at all four stages:
                     'add', 'concat', 'lsf_all'.
        coband_sigma, coband_kernels, coband_learned_filter: Gaussian filter of the band split
                     (paper: sigma 2, 7x7 at stage 1 and 5x5 at stage 2, fixed).
        modality: 'fusion', or 'rgb_only' / 'ir_only' for the single-encoder references.
    """

    def __init__(
        self,
        rgb_arch: str = 'convnextv2_nano.fcmae_ft_in22k_in1k_384',
        ir_arch: str = 'convnextv2_nano.fcmae_ft_in22k_in1k_384',
        pretrained: bool = True,
        num_classes: int = 9,
        context_dim: List[int] = (80, 160, 320, 640),
        output_resolution: Tuple[int, int] = (480, 640),
        deep_supervision: bool = True,
        stage12: str = 'coband',
        deep: str = 'lsf',
        fusion_type: str = 'default',
        coband_sigma: float = 2.0,
        coband_kernels: Tuple[int, int] = (7, 5),
        coband_learned_filter: bool = False,
        modality: str = 'fusion',
    ):
        super().__init__()
        assert stage12 in ('coband', 'fgf'), stage12
        assert deep in ('lsf', 'enhanced'), deep
        assert fusion_type in ('default', 'add', 'concat', 'lsf_all'), fusion_type
        assert modality in ('fusion', 'rgb_only', 'ir_only'), modality
        self.modality = modality
        self.num_classes = num_classes

        # The construction order (decoder, RGB encoder, thermal encoder, fusion stages 1-4) is kept as in the
        # original training code: it fixes the random initialization for a given seed.
        self.decoder = PANetDecoder(in_channels=list(context_dim), num_classes=num_classes,
                                    output_resolution=output_resolution, deep_supervision=deep_supervision)

        if modality != 'ir_only':
            self.rgb_encoder = timm.create_model(rgb_arch, pretrained=pretrained, features_only=True)
            self.rgb_channels = self.rgb_encoder.feature_info.channels()
        else:
            self.rgb_encoder = None
            self.rgb_channels = None

        if modality != 'rgb_only':
            self.ir_encoder = timm.create_model(ir_arch, pretrained=pretrained, features_only=True, in_chans=1)
            self.ir_channels = self.ir_encoder.feature_info.channels()
        else:
            self.ir_encoder = None
            self.ir_channels = None

        if modality == 'fusion':
            rc, ic = self.rgb_channels, self.ir_channels
            if fusion_type == 'add':
                for i in range(4):
                    setattr(self, f'fusion_stage{i + 1}', AddFusion(rgb_c=rc[i], ir_c=ic[i]))
            elif fusion_type == 'concat':
                for i in range(4):
                    setattr(self, f'fusion_stage{i + 1}', ConcatFusion(rgb_c=rc[i], ir_c=ic[i]))
            elif fusion_type == 'lsf_all':
                for i in range(4):
                    setattr(self, f'fusion_stage{i + 1}', LightweightSemanticFusion(rgb_c=rc[i], ir_c=ic[i]))
            else:
                # stages 1-2
                if stage12 == 'coband':
                    self.fusion_stage1 = CoBand(rgb_c=rc[0], ir_c=ic[0], kernel_size=coband_kernels[0],
                                                sigma=coband_sigma, learnable_filter=coband_learned_filter)
                    self.fusion_stage2 = CoBand(rgb_c=rc[1], ir_c=ic[1], kernel_size=coband_kernels[1],
                                                sigma=coband_sigma, learnable_filter=coband_learned_filter)
                else:
                    self.fusion_stage1 = FGF(rgb_c=rc[0], ir_c=ic[0])
                    self.fusion_stage2 = FGF(rgb_c=rc[1], ir_c=ic[1])
                # stages 3-4
                if deep == 'lsf':
                    self.fusion_stage3 = LightweightSemanticFusion(rgb_c=rc[2], ir_c=ic[2])
                    self.fusion_stage4 = LightweightSemanticFusion(rgb_c=rc[3], ir_c=ic[3])
                else:
                    self.fusion_stage3 = EnhancedSemanticFusion(rgb_c=rc[2], ir_c=ic[2])
                    self.fusion_stage4 = EnhancedSemanticFusion(rgb_c=rc[3], ir_c=ic[3])

    def forward(self, rgb, ir):
        if self.modality in ('fusion', 'rgb_only'):
            rgb_features = self.rgb_encoder(rgb)
        if self.modality in ('fusion', 'ir_only'):
            ir_features = self.ir_encoder(ir)

        if self.modality == 'ir_only':
            fused = ir_features
        elif self.modality == 'rgb_only':
            fused = rgb_features
        else:
            fused = [self.fusion_stage1(rgb_features[0], ir_features[0]),
                     self.fusion_stage2(rgb_features[1], ir_features[1]),
                     self.fusion_stage3(rgb_features[2], ir_features[2]),
                     self.fusion_stage4(rgb_features[3], ir_features[3])]

        main_logits, aux_logits = self.decoder(fused)
        return main_logits, aux_logits
