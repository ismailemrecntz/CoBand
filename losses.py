"""Training objective: weighted sum of cross-entropy, Dice, Lovasz, OHEM cross-entropy, boundary
and (optional) focal terms on the main output. Auxiliary and deep-supervision heads use plain cross-entropy
(see train.py)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------------------
# Lovasz terms
# ---------------------------------------------------------------------------------------------------------
def lovasz_grad(gt_sorted):
    """Gradient of the Lovasz extension of the Jaccard loss w.r.t. sorted errors."""
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax(logits, labels, classes='present', ignore_index=None):
    """Lovasz-Softmax loss (Berman et al., CVPR 2018). logits: [B, C, H, W] (raw), labels: [B, H, W]."""
    probs = torch.softmax(logits.float(), dim=1)
    C = probs.shape[1]
    probs = probs.permute(0, 2, 3, 1).reshape(-1, C)
    labels = labels.reshape(-1)
    if ignore_index is not None:
        valid = labels != ignore_index
        probs, labels = probs[valid], labels[valid]
    if labels.numel() == 0:
        return logits.sum() * 0.0
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg[perm])))
    return torch.stack(losses).mean() if len(losses) else logits.sum() * 0.0


def rank_weighted_lovasz(logits, labels, classes='present', eps=1e-6, ignore_index=None):
    """Lovasz-style term used for the paper models (--rank_weighted_lovasz).

    For every present class c the errors |1[y=c] - p_c| are sorted in descending order; the i-th error is
    weighted by w_i = 1 - (number of class-c pixels among the first i) / (number of class-c pixels), the sum is
    divided by H * W (not by the batch size) and averaged over the present classes.
    """
    probs = torch.softmax(logits.float(), dim=1)
    B, C, H, W = probs.shape
    losses = []
    for c in range(C):
        if classes == 'present' and (labels == c).sum() == 0:
            continue
        fg = (labels == c).float()
        pc = probs[:, c, ...]
        errors = (fg - pc).abs().reshape(-1)
        errors_sorted, perm = torch.sort(errors, descending=True)
        gt_sorted = fg.reshape(-1)[perm]
        grad = torch.cumsum(gt_sorted, 0) / (gt_sorted.sum() + eps)
        grad = 1.0 - grad
        losses.append(torch.dot(errors_sorted, grad) / (H * W))
    return torch.stack(losses).mean() if len(losses) else probs.new_tensor(0.)


# ---------------------------------------------------------------------------------------------------------
# Other terms
# ---------------------------------------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """FL = (1 - p_t)^gamma * CE. With target_class >= 0 the loss is averaged over the pixels of that class only."""
    def __init__(self, gamma=2.0, target_class=None, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.target_class = target_class
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        p = torch.exp(-ce_loss)
        focal_weight = (1 - p) ** self.gamma
        if self.target_class is not None and self.target_class >= 0:
            mask = (targets == self.target_class).float()
            focal_loss = focal_weight * ce_loss * mask
            return focal_loss.sum() / (mask.sum() + 1e-8)
        return (focal_weight * ce_loss).mean()


class OHEMCrossEntropyLoss(nn.Module):
    """Online hard example mining: mean cross-entropy over the pixels whose maximum class probability is below
    `thresh`, but at least the `min_kept` pixels with the largest loss."""
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.thresh = thresh
        self.min_kept = min_kept
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')

    def forward(self, logits, targets):
        loss = self.ce(logits, targets)                                   # [B, H, W]
        loss_flat = loss.view(-1)
        valid_mask = (targets != self.ignore_index).view(-1)
        loss_flat = loss_flat[valid_mask]
        if loss_flat.numel() == 0:
            return loss.mean()

        loss_sorted, _ = torch.sort(loss_flat, descending=True)
        num_pixels = loss_flat.numel()
        min_kept = min(self.min_kept, num_pixels)

        probs = torch.softmax(logits, dim=1)
        max_probs, _ = probs.max(dim=1)
        max_probs_flat = max_probs.view(-1)[valid_mask]
        hard_mask = max_probs_flat < self.thresh
        num_hard = hard_mask.sum().item()

        k = max(min_kept, num_hard)
        k = min(k, num_pixels)
        threshold_loss = loss_sorted[int(k) - 1] if k > 0 else loss_sorted[0]
        hard_loss = loss_flat[loss_flat >= threshold_loss]
        return hard_loss.mean()


class BoundaryLoss(nn.Module):
    """Weighted BCE between Laplacian edge maps of the ground truth and of the predicted class probabilities
    (weight 3 on ground-truth boundary pixels, 1 elsewhere)."""
    def __init__(self, num_classes=9, ignore_index=255, kernel_size=3):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.kernel_size = kernel_size
        laplacian = torch.tensor([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('laplacian', laplacian)

    def get_boundary(self, mask):
        one_hot = F.one_hot(mask.clamp(0, self.num_classes - 1), self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()                    # [B, C, H, W]
        boundaries = []
        for c in range(self.num_classes):
            edges = F.conv2d(one_hot[:, c:c + 1, :, :], self.laplacian, padding=1)
            boundaries.append((edges.abs() > 0.1).float())
        boundary = torch.cat(boundaries, dim=1).max(dim=1, keepdim=True)[0]
        return boundary.squeeze(1)                                        # [B, H, W]

    def forward(self, logits, targets):
        gt_boundary = self.get_boundary(targets)
        probs = torch.softmax(logits, dim=1)
        pred_boundary = torch.zeros_like(gt_boundary)
        for c in range(self.num_classes):
            edges = F.conv2d(probs[:, c:c + 1, :, :], self.laplacian.to(probs.device), padding=1)
            pred_boundary = torch.max(pred_boundary, edges.abs().squeeze(1))
        pred_boundary = pred_boundary * 5.0                               # used as logits

        valid_mask = (targets != self.ignore_index).float()
        boundary_weight = 1.0 + 2.0 * gt_boundary
        loss = F.binary_cross_entropy_with_logits(pred_boundary, gt_boundary,
                                                  weight=boundary_weight * valid_mask, reduction='sum')
        return loss / (valid_mask.sum() + 1e-6)


class SegmentationLoss(nn.Module):
    """CE + Dice + Lovasz + OHEM + boundary (+ focal) on the main output."""
    def __init__(self, ce_w=0.4, dice_w=0.2, lovasz_w=0.2, ohem_w=0.1, boundary_w=0.1,
                 class_weights=None, ignore_index=255, num_classes=9,
                 ohem_thresh=0.7, ohem_min_kept=100000,
                 focal_weight=0.0, focal_gamma=2.0, focal_target_class=6,
                 rank_weighted=False):
        super().__init__()
        self.lovasz_fn = rank_weighted_lovasz if rank_weighted else lovasz_softmax
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.ohem = OHEMCrossEntropyLoss(ignore_index=ignore_index, thresh=ohem_thresh, min_kept=ohem_min_kept,
                                         weight=class_weights)
        self.boundary = BoundaryLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.class_weights = class_weights
        self.ce_w = ce_w
        self.dice_w = dice_w
        self.lovasz_w = lovasz_w
        self.ohem_w = ohem_w
        self.boundary_w = boundary_w
        self.focal_w = focal_weight
        if focal_weight > 0:
            self.focal_loss = FocalLoss(gamma=focal_gamma, target_class=focal_target_class, ignore_index=ignore_index)
        else:
            self.focal_loss = None
        target = 'all classes' if focal_target_class is None or focal_target_class < 0 else f'class {focal_target_class}'
        print(f"[Loss] CE={ce_w} Dice={dice_w} Lovasz={lovasz_w} ({'rank-weighted' if rank_weighted else 'Lovasz-Softmax'}) "
              f"OHEM={ohem_w} Boundary={boundary_w} Focal={focal_weight}"
              + (f" (gamma={focal_gamma}, {target})" if focal_weight > 0 else ""))

    def dice(self, logits, y, eps=1e-6):
        logits = logits.float()
        C = logits.shape[1]
        y1h = F.one_hot(y.clamp_min(0), C).permute(0, 3, 1, 2).float()
        p = torch.softmax(logits, dim=1)
        inter = (p * y1h).sum(dim=(0, 2, 3))
        union = p.sum(dim=(0, 2, 3)) + y1h.sum(dim=(0, 2, 3))
        dice = (2 * inter + eps) / (union + eps)
        return 1 - dice.mean()

    def forward(self, logits, y):
        loss = (self.ce_w * self.ce(logits, y)
                + self.dice_w * self.dice(logits, y)
                + self.lovasz_w * self.lovasz_fn(logits, y, ignore_index=self.ce.ignore_index)
                + self.ohem_w * self.ohem(logits, y)
                + self.boundary_w * self.boundary(logits, y))
        if self.focal_loss is not None:
            loss = loss + self.focal_w * self.focal_loss(logits, y)
        return loss
