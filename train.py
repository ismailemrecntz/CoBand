"""Training of CoBand and of the ablation variants.

The exact commands for every model of the paper are in scripts/. All paper models were trained with
--paper_pipeline (see README, "Training pipeline of the paper models").
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import CLASS_NAMES, build_dataset
from losses import SegmentationLoss
from utils import build_model, iou_from_confusion


# ---------------------------------------------------------------------------------------------------------
# Exponential moving average of the weights
# ---------------------------------------------------------------------------------------------------------
class ModelEMA:
    """EMA of the parameters and, unless include_buffers=False (--ema_params_only), of the BatchNorm buffers."""
    def __init__(self, model, decay=0.999, include_buffers=True):
        self.decay = decay
        self.include_buffers = include_buffers
        self.shadow = {}
        self.backup = {}
        self._init_from(model)

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, "module") else model

    def _init_from(self, model):
        m = self._unwrap(model)
        self.shadow = {}
        for name, p in m.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.detach().clone().to(p.device)
        if self.include_buffers:
            for name, b in m.named_buffers():
                self.shadow[name] = b.detach().clone()

    @torch.no_grad()
    def update(self, model):
        m = self._unwrap(model)
        for name, p in m.named_parameters():
            if name in self.shadow and p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
        for name, b in m.named_buffers():
            if name not in self.shadow:
                continue
            if b.dtype.is_floating_point:
                self.shadow[name].mul_(self.decay).add_(b, alpha=1.0 - self.decay)
            else:  # counters such as num_batches_tracked
                self.shadow[name].copy_(b)

    def _tensors(self, model):
        m = self._unwrap(model)
        yield from m.named_parameters()
        yield from m.named_buffers()

    def apply_shadow(self, model):
        self.backup = {}
        for name, t in self._tensors(model):
            if name in self.shadow:
                self.backup[name] = t.data.clone()
                t.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, t in self._tensors(model):
            if name in self.backup:
                t.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return self.shadow


# ---------------------------------------------------------------------------------------------------------
# DataLoader worker seeding
# ---------------------------------------------------------------------------------------------------------
def seed_augmentation_worker(worker_id):
    """albumentations 2.x keeps its own random generator per pipeline and the DataLoader does not reseed it, so
    without this function all workers draw the same augmentations and the sequence repeats every epoch
    (this is the behaviour of the paper models, --no_worker_reseed). torch.initial_seed() differs per worker
    and per epoch and is reproducible for a given --seed."""
    from albumentations.core.composition import BaseCompose
    from albumentations.core.transforms_interface import BasicTransform

    seed = torch.initial_seed() % 2 ** 32
    dataset = torch.utils.data.get_worker_info().dataset
    for k, attr in enumerate(vars(dataset).values()):
        if isinstance(attr, (BaseCompose, BasicTransform)):
            attr.set_random_state(np.random.default_rng([seed, k]), random.Random(seed * 1000 + k))


# ---------------------------------------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------------------------------------
class Trainer:
    # The order of the setup steps fixes the use of the random number generators; keep it unchanged.
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f'cuda:{config.gpu}' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.set_seed(config.seed)
        self.setup_directories()
        self.model = self.setup_model()
        self.train_loader, self.val_loader = self.setup_data()
        self.criterion_main, self.criterion_aux = self.setup_loss()
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()
        # bf16 has the exponent range of fp32 and needs no loss scaling
        self.amp_dtype = torch.bfloat16 if config.amp_dtype == 'bf16' else torch.float16
        self.scaler = GradScaler(enabled=torch.cuda.is_available() and self.amp_dtype == torch.float16)
        self.ema = ModelEMA(self.model, decay=config.ema_decay, include_buffers=not config.ema_params_only)
        self.setup_logging()

        self.start_epoch = 1
        self.best_miou = 0.0

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup_directories(self):
        self.exp_dir = Path(self.config.exp_dir) / self.config.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = self.exp_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(vars(self.config), f, indent=4)
        print(f"Experiment directory: {self.exp_dir}")

    def setup_model(self):
        model = build_model(vars(self.config), pretrained=True).to(self.device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        return model

    def setup_data(self):
        c = self.config
        resolution = (c.img_height, c.img_width)
        train_dataset = build_dataset(c.dataset, c.data_root, 'train', resolution, train_source=c.train_source,
                                      strong_gauss_noise=c.strong_gauss_noise)
        val_dataset = build_dataset(c.dataset, c.data_root, 'test', resolution)

        train_loader = DataLoader(
            train_dataset,
            batch_size=c.batch_size,
            shuffle=True,
            num_workers=c.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=None if c.no_worker_reseed else seed_augmentation_worker,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=c.batch_size,
            shuffle=False,
            num_workers=c.num_workers,
            pin_memory=True
        )
        print(f"Train samples: {len(train_dataset)} | Test samples: {len(val_dataset)} | "
              f"Batch size: {c.batch_size} | Iterations per epoch: {len(train_loader)}")
        return train_loader, val_loader

    def setup_loss(self):
        c = self.config
        class_weights = self.calculate_class_weights() if c.class_weights else None
        ignore_idx = -100          # no label is ignored (the 'unlabeled' class of MFNet is learned)
        focal_target = c.focal_target_class if c.focal_target_class >= 0 else None   # < 0: all classes
        criterion_main = SegmentationLoss(
            ce_w=c.ce_weight,
            dice_w=c.dice_weight,
            lovasz_w=c.lovasz_weight,
            ohem_w=c.ohem_weight,
            boundary_w=c.boundary_weight,
            class_weights=class_weights,
            ignore_index=ignore_idx,
            num_classes=c.num_classes,
            ohem_thresh=c.ohem_thresh,
            ohem_min_kept=c.ohem_min_kept,
            focal_weight=c.focal_weight,
            focal_gamma=c.focal_gamma,
            focal_target_class=focal_target,
            rank_weighted=c.rank_weighted_lovasz,
        ).to(self.device)
        criterion_aux = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_idx).to(self.device)
        return criterion_main, criterion_aux

    def calculate_class_weights(self):
        """Inverse class frequency on the (augmented) training set, normalized to max 1, multiplied by
        --class_weight_multiplier and clamped to [0.5, 15]; extra boosts for rare MFNet / PST900 classes."""
        c = self.config
        class_counts = torch.zeros(c.num_classes, device=self.device)
        for _, _, mask, _ in tqdm(self.train_loader, desc="Computing class weights"):
            mask = mask.to(self.device)
            valid = (mask >= 0) & (mask < c.num_classes)
            class_counts += torch.bincount(mask[valid], minlength=c.num_classes)

        total_pixels = class_counts.sum()
        weights = total_pixels / (c.num_classes * class_counts.clamp(min=1))
        weights = weights / weights.max()
        weights = weights * c.class_weight_multiplier
        weights = torch.clamp(weights, 0.5, 15.0)

        if c.dataset == 'pst900' and c.num_classes == 5:
            weights[3] *= 1.3      # hand drill
        if c.dataset == 'mfnet' and c.num_classes == 9:
            weights[5] *= 1.4      # car stop
            weights[6] *= 2.0      # guardrail
            weights[8] *= 1.3      # bump
        print("Class weights: " + " ".join(f"{w:.4f}" for w in weights.tolist()))
        return weights

    def setup_optimizer(self):
        """AdamW with parameter groups: layer-wise decayed learning rate for the encoders (factor 0.9 per stage
        from the deepest stage), separate learning rates for fusion modules and decoder, and fixed weight decay
        per group (RGB encoder 0.05, thermal encoder 0.03, fusion 0.01, decoder 0.005)."""
        model = self.model.module if hasattr(self.model, "module") else self.model
        base_lr = self.config.lr_backbone
        decay = 0.90
        param_groups = {}

        def get_rgb_layer_id(name):
            if "stem" in name: return 0
            if "stages_0" in name: return 1
            if "stages_1" in name: return 2
            if "stages_2" in name: return 3
            if "stages_3" in name: return 4
            return 5

        ir_lr_by_block = self.config.ir_lr_by_block

        def get_ir_layer_id(name):
            if ("stages_" in name or ".stem_" in name) and not ir_lr_by_block:
                return get_rgb_layer_id(name)
            # --ir_lr_by_block (paper models): the thermal encoder ids follow the substring "blocks.K", i.e. the
            # block index inside each stage (blocks 0-4 -> id 1-5; stem, downsampling layers and blocks >= 5
            # -> id 6, i.e. lr_backbone / 0.9). Note that e.g. "blocks.10" also matches "blocks.1".
            if "conv_stem" in name: return 0
            if "blocks.0" in name: return 1
            if "blocks.1" in name: return 2
            if "blocks.2" in name: return 3
            if "blocks.3" in name: return 4
            if "blocks.4" in name: return 5
            return 6

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("rgb_encoder"):
                layer_id = get_rgb_layer_id(name)
                lr = base_lr * (decay ** (5 - layer_id))
                wd = 0.05
                key = f"rgb_{layer_id}"
            elif name.startswith("ir_encoder"):
                layer_id = get_ir_layer_id(name)
                lr = base_lr * (decay ** (5 - layer_id))
                wd = 0.03
                key = f"ir_{layer_id}"
            elif "fusion_stage" in name:
                lr = self.config.lr_fusion
                wd = 0.01
                key = "fusion"
            elif "decoder" in name:
                lr = self.config.lr_decoder
                wd = 0.005
                key = "decoder"
            else:
                lr = base_lr * 2.0
                wd = 0.005
                key = "other"
            if key not in param_groups:
                param_groups[key] = {"params": [], "lr": lr, "weight_decay": wd, "name": key}
            param_groups[key]["params"].append(param)

        return torch.optim.AdamW(list(param_groups.values()), betas=(0.9, 0.999))

    def setup_scheduler(self):
        c = self.config
        if c.scheduler == 'cosine_restart':
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            return CosineAnnealingWarmRestarts(self.optimizer, T_0=c.t0, T_mult=2, eta_min=1e-7)
        if c.scheduler == 'cosine_warmup':
            # linear warm-up, constant phase, then the learning rate halves every 15 epochs
            warmup_epochs, flat_epochs = c.warmup_epochs, c.flat_epochs

            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                elif epoch < warmup_epochs + flat_epochs:
                    return 1.0
                else:
                    return 0.5 ** ((epoch - warmup_epochs - flat_epochs) / 15)

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        raise ValueError(f"Unknown scheduler: {c.scheduler}")

    def setup_logging(self):
        self.writer = SummaryWriter(self.log_dir)
        self._log_file = open(self.exp_dir / 'log.txt', 'a', buffering=1)
        self._log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Experiment started: {self.exp_dir}\n")

    def log(self, msg, print_console=True):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
        self._log_file.write(line)
        if print_console:
            print(line, end='')

    def get_aux_weight(self, epoch):
        """Weight of the auxiliary losses: linear decay from aux_weight_start to aux_weight_end."""
        c = self.config
        if epoch < c.aux_weight_decay_epoch:
            alpha = epoch / c.aux_weight_decay_epoch
            return (1 - alpha) * c.aux_weight_start + alpha * c.aux_weight_end
        return c.aux_weight_end

    def train_epoch(self, epoch):
        self.model.train()
        accumulation_steps = max(1, int(self.config.grad_accum_steps))
        running_loss = running_main_loss = running_aux_loss = 0.0
        num_batches = 0
        aux_weight = self.get_aux_weight(epoch)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.epochs}')
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (rgb, ir, masks, _) in enumerate(pbar):
            if masks.max() >= self.config.num_classes:          # e.g. fill value of geometric augmentation
                masks = masks.clamp_(0, self.config.num_classes - 1)

            rgb = rgb.to(device=self.device) if self.config.modality != "ir_only" else None
            ir = ir.to(device=self.device) if self.config.modality != "rgb_only" else None
            masks = masks.to(device=self.device)

            with autocast('cuda', enabled=True, dtype=self.amp_dtype):
                main_out, aux_out = self.model(rgb, ir)
                loss_main = self.criterion_main(main_out, masks)
                if isinstance(aux_out, dict):
                    # deep supervision: {'aux': tensor, 'deep': [level 0, ..., level 3]}
                    loss_aux = self.criterion_aux(aux_out['aux'], masks)
                    deep_weights = [0.1, 0.2, 0.3, 0.4]
                    for i, deep_out in enumerate(aux_out.get('deep', [])):
                        weight = deep_weights[i] if i < len(deep_weights) else 0.1
                        loss_aux = loss_aux + weight * self.criterion_aux(deep_out, masks)
                else:
                    loss_aux = self.criterion_aux(aux_out, masks) if aux_out is not None else 0
                loss = loss_main + aux_weight * loss_aux
                loss = loss / accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):            # skip the batch
                self.optimizer.zero_grad()
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
                continue

            running_loss += loss.item() * accumulation_steps
            running_main_loss += loss_main.item()
            if aux_out is not None and (isinstance(aux_out, dict) or loss_aux != 0):
                running_aux_loss += loss_aux.item() if isinstance(loss_aux, torch.Tensor) else loss_aux

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
                for param in self.model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        param.grad.zero_()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.ema.update(self.model)

            num_batches += 1
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'main': f'{loss_main.item():.4f}',
                'aux': f'{loss_aux.item() if loss_aux != 0 else 0:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

        if num_batches == 0:
            print(f"[WARNING] All batches in epoch {epoch} were skipped due to NaN/Inf loss!")
            return float('nan'), float('nan'), float('nan')
        return running_loss / num_batches, running_main_loss / num_batches, running_aux_loss / num_batches

    def validate(self, epoch):
        """Test-set evaluation with the EMA weights (single scale)."""
        n = self.config.num_classes
        self.model.eval()
        running_loss = 0.0
        aux_weight = self.get_aux_weight(epoch)
        conf_total = np.zeros((n, n))
        self.ema.apply_shadow(self.model)

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for rgb, ir, masks, _ in pbar:
                rgb = rgb.to(self.device) if self.config.modality != "ir_only" else None
                ir = ir.to(self.device) if self.config.modality != "rgb_only" else None
                masks = masks.to(self.device)

                with autocast(device_type='cuda', enabled=True, dtype=self.amp_dtype):
                    main_out, aux_out = self.model(rgb, ir)
                loss_main = self.criterion_main(main_out, masks)
                loss_aux = self.criterion_aux(aux_out, masks)
                loss = loss_main + aux_weight * loss_aux
                running_loss += loss.item()

                preds = main_out.argmax(dim=1).cpu().numpy().reshape(-1)
                labels = masks.cpu().numpy().reshape(-1)
                valid = (labels >= 0) & (labels < n)
                conf_total += np.bincount(n * labels[valid] + preds[valid], minlength=n * n).reshape(n, n)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        self.ema.restore(self.model)

        iou_per_class = iou_from_confusion(conf_total)
        miou = np.mean(iou_per_class)
        pixel_acc = np.trace(conf_total) / np.sum(conf_total)
        torch.cuda.empty_cache()
        return running_loss / len(self.val_loader), miou, pixel_acc, iou_per_class

    def save_checkpoint(self, epoch):
        """best.pth: raw weights + optimizer state; best_model.ema.pth: EMA weights (the released models)."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "best_miou": float(self.best_miou),
            "config": vars(self.config),
        }
        torch.save(ckpt, self.checkpoint_dir / "best.pth")
        self.ema.apply_shadow(self.model)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "config": vars(self.config),
            "best_miou": float(self.best_miou)
        }, self.checkpoint_dir / "best_model.ema.pth")
        self.ema.restore(self.model)

    def train(self):
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            torch.cuda.empty_cache()
            train_loss, train_main_loss, train_aux_loss = self.train_epoch(epoch)
            val_loss, miou, pixel_acc, class_ious = self.validate(epoch)
            self.scheduler.step()

            is_best = miou > self.best_miou
            if is_best:
                self.best_miou = miou
                self.save_checkpoint(epoch)

            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/mIoU', miou, epoch)
            self.writer.add_scalar('Val/PixelAcc', pixel_acc, epoch)
            for pg in self.optimizer.param_groups:
                self.writer.add_scalar(f"LR/{pg.get('name', 'default')}", pg['lr'], epoch)
            lrs = ','.join(f"{pg.get('name', 'default')}:{pg['lr']:.2e}" for pg in self.optimizer.param_groups)
            self.log(f"Epoch {epoch}/{self.config.epochs} | "
                     f"TrainLoss={train_loss:.4f} (Main={train_main_loss:.4f}, Aux={train_aux_loss:.4f}) | "
                     f"ValLoss={val_loss:.4f} | mIoU={miou:.4f} | PixelAcc={pixel_acc:.4f} | LR={lrs}")
            class_names = CLASS_NAMES.get(self.config.dataset, [f"c{i}" for i in range(len(class_ious))])
            self.log(f"Epoch {epoch} ClassIoU | " + " ".join(
                f"{n}={iou:.4f}" for n, iou in zip(class_names, class_ious)), print_console=False)
            for n, iou in zip(class_names, class_ious):
                self.writer.add_scalar(f'ValClassIoU/{n}', iou, epoch)
        print(f"\nTraining completed. Best mIoU: {self.best_miou:.4f}")


# ---------------------------------------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------------------------------------
def get_parser():
    p = argparse.ArgumentParser(description='Train CoBand (RGB-thermal semantic segmentation)')

    g = p.add_argument_group('data')
    g.add_argument('--dataset', type=str, default='mfnet', choices=['mfnet', 'pst900', 'iron'])
    g.add_argument('--data_root', type=str, required=True)
    g.add_argument('--train_source', type=str, default='train_', choices=['train', 'train_'],
                   help="MFNet training list: 'train_' = train + val (paper), 'train' = train only")
    g.add_argument('--img_height', type=int, default=480)
    g.add_argument('--img_width', type=int, default=640)
    g.add_argument('--num_classes', type=int, default=9)

    g = p.add_argument_group('model')
    g.add_argument('--rgb_arch', type=str, default='convnextv2_nano.fcmae_ft_in22k_in1k_384')
    g.add_argument('--ir_arch', type=str, default='convnextv2_nano.fcmae_ft_in22k_in1k_384')
    g.add_argument('--context_dim', type=str, default='[80,160,320,640]', help='channel widths of the encoder stages')
    g.add_argument('--deep_supervision', action='store_true', default=False)
    g.add_argument('--stage12', type=str, default='coband', choices=['coband', 'fgf'],
                   help='fusion at stages 1-2: CoBand, or FGF (band gate without RGB context)')
    g.add_argument('--deep', type=str, default='lsf', choices=['lsf', 'enhanced'],
                   help='fusion at stages 3-4: lightweight semantic fusion (LSF) or the enhanced block')
    g.add_argument('--fusion_type', type=str, default='default', choices=['default', 'add', 'concat', 'lsf_all'],
                   help="ablation: one simple fusion at all four stages ('default' = --stage12 / --deep)")
    g.add_argument('--modality', type=str, default='fusion', choices=['fusion', 'ir_only', 'rgb_only'])
    g.add_argument('--coband_sigma', type=float, default=2.0, help='sigma of the Gaussian band split')
    g.add_argument('--coband_kernels', type=str, default='7,5', help='Gaussian kernel size at stage 1 and 2')
    g.add_argument('--coband_learned_filter', action='store_true', default=False,
                   help='ablation: free learnable depthwise low-pass kernel (Gaussian initialization)')

    g = p.add_argument_group('optimization')
    g.add_argument('--epochs', type=int, default=120)
    g.add_argument('--batch_size', type=int, default=4)
    g.add_argument('--num_workers', type=int, default=4)
    g.add_argument('--lr_backbone', type=float, default=5e-5)
    g.add_argument('--lr_fusion', type=float, default=5e-4)
    g.add_argument('--lr_decoder', type=float, default=5e-4)
    g.add_argument('--clip_grad', type=float, default=0.5)
    g.add_argument('--grad_accum_steps', type=int, default=1)
    g.add_argument('--scheduler', type=str, default='cosine_restart', choices=['cosine_restart', 'cosine_warmup'])
    g.add_argument('--t0', type=int, default=40, help='first period of cosine_restart (T_mult = 2)')
    g.add_argument('--warmup_epochs', type=int, default=10, help='cosine_warmup only')
    g.add_argument('--flat_epochs', type=int, default=20, help='cosine_warmup only')
    g.add_argument('--ema_decay', type=float, default=0.999)
    g.add_argument('--amp_dtype', type=str, default='fp16', choices=['fp16', 'bf16'],
                   help='mixed precision type (bf16 for runs whose fp16 activations overflow)')

    g = p.add_argument_group('loss')
    g.add_argument('--ce_weight', type=float, default=0.30)
    g.add_argument('--dice_weight', type=float, default=0.25)
    g.add_argument('--lovasz_weight', type=float, default=0.15)
    g.add_argument('--ohem_weight', type=float, default=0.15)
    g.add_argument('--boundary_weight', type=float, default=0.08)
    g.add_argument('--ohem_thresh', type=float, default=0.7)
    g.add_argument('--ohem_min_kept', type=int, default=100000)
    g.add_argument('--focal_weight', type=float, default=0.0, help='0 disables the focal term')
    g.add_argument('--focal_gamma', type=float, default=2.0)
    g.add_argument('--focal_target_class', type=int, default=-1,
                   help='-1: all classes; >= 0: only this class (paper: 6 guardrail on MFNet, 4 survivor on PST900)')
    g.add_argument('--class_weights', action='store_true', default=False, help='inverse-frequency class weights')
    g.add_argument('--class_weight_multiplier', type=float, default=1.0)
    g.add_argument('--aux_weight_start', type=float, default=0.3)
    g.add_argument('--aux_weight_end', type=float, default=0.05)
    g.add_argument('--aux_weight_decay_epoch', type=int, default=80)

    g = p.add_argument_group('training pipeline of the paper models (see README)')
    g.add_argument('--paper_pipeline', action='store_true', default=False, help='switch on all five options below')
    g.add_argument('--rank_weighted_lovasz', action='store_true', default=False,
                   help='rank-weighted Lovasz term instead of Lovasz-Softmax')
    g.add_argument('--strong_gauss_noise', action='store_true', default=False,
                   help='Gaussian noise with std 0.2-0.44 (of 255) in the photometric augmentation')
    g.add_argument('--ema_params_only', action='store_true', default=False,
                   help='EMA of the parameters only (BatchNorm statistics of the raw model)')
    g.add_argument('--ir_lr_by_block', action='store_true', default=False,
                   help='layer-wise learning rate of the thermal encoder by block index instead of stage')
    g.add_argument('--no_worker_reseed', action='store_true', default=False,
                   help='do not reseed the augmentation generators of the DataLoader workers')

    g = p.add_argument_group('run')
    g.add_argument('--exp_name', type=str, required=True)
    g.add_argument('--exp_dir', type=str, default='experiments')
    g.add_argument('--gpu', type=int, default=0)
    g.add_argument('--seed', type=int, default=42)
    return p


def main():
    args = get_parser().parse_args()
    if args.paper_pipeline:
        args.rank_weighted_lovasz = args.strong_gauss_noise = args.ema_params_only = True
        args.ir_lr_by_block = args.no_worker_reseed = True
    Trainer(args).train()


if __name__ == '__main__':
    main()
