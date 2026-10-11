"""Single-scale test-set evaluation of a trained model (protocol of the paper).

    python evaluate.py --run weights/mfnet_coband --data_root /path/to/MFNet

--run is a folder with config.json and best_model.ema.pth (or checkpoints/best_model.ema.pth, as written by
train.py). Reported numbers:
    all datasets : mIoU and per-class IoU (per class: summed intersection / summed union over the test set)
    MFNet        : additionally day and night images (name ends with D / N)
    IRON         : drivable-class IoU on all frames, per scene and on low-light frames (mean RGB intensity
                   of the frame < 0.2 on a [0, 1] scale)

Inference uses fp16 autocast, except for models trained with bf16 (amp_dtype = bf16), which are evaluated in
fp32 because their activations can exceed the fp16 range.
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import CLASS_NAMES, NORM, build_dataset
from utils import build_model, load_config

LOW_LIGHT_THR = 0.2


def find_checkpoint(run):
    for p in (os.path.join(run, 'best_model.ema.pth'), os.path.join(run, 'checkpoints', 'best_model.ema.pth')):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'no best_model.ema.pth in {run}')


def load_model(run, device):
    c = load_config(os.path.join(run, 'config.json'))
    model = build_model(c, pretrained=False).to(device)
    ckpt = torch.load(find_checkpoint(run), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    return model, c


@torch.no_grad()
def predict(model, c, device, data_root, amp, batch_size, num_workers=4):
    """Per-image intersection / union per class, image names and mean RGB intensity."""
    n = c['num_classes']
    H, W = c['img_height'], c['img_width']
    loader = DataLoader(build_dataset(c['dataset'], data_root, 'test', (H, W)), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    rgb_mean, rgb_std = (torch.as_tensor(np.asarray(a, dtype=np.float32), device=device).view(1, -1, 1, 1)
                         for a in NORM[c['dataset']][:2])
    cls = torch.arange(n, device=device).view(1, n, 1, 1)
    inter, union, names, lum = [], [], [], []
    for rgb, ir, y, name in loader:
        rgb, ir, y = rgb.to(device), ir.to(device), y.to(device)
        with torch.autocast('cuda', enabled=amp):
            logits, _ = model(rgb, ir)
        P = logits.float().argmax(1).unsqueeze(1) == cls
        G = y.unsqueeze(1) == cls
        inter.append((P & G).sum((2, 3)).cpu())
        union.append((P | G).sum((2, 3)).cpu())
        lum.append((rgb * rgb_std + rgb_mean).clamp(0, 1).mean((1, 2, 3)).cpu())
        names.extend(list(name))
    return (torch.cat(inter).numpy().astype(np.int64), torch.cat(union).numpy().astype(np.int64),
            np.array(names), torch.cat(lum).numpy())


def iou(inter, union, mask=None):
    m = np.ones(len(inter), bool) if mask is None else mask
    return inter[m].sum(0) / np.maximum(union[m].sum(0), 1) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='folder with config.json and best_model.ema.pth')
    ap.add_argument('--data_root', default=None, help='dataset root (default: data_root of the config)')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--precision', choices=['auto', 'fp16', 'fp32'], default='auto',
                    help='auto: fp16 autocast, fp32 for models trained with bf16')
    ap.add_argument('--batch_size', type=int, default=None, help='default: 4 for PST900, 8 otherwise')
    ap.add_argument('--save', default=None, help='optional JSON file for the results')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    model, c = load_model(args.run, device)
    ds = c['dataset']
    amp = {'auto': c.get('amp_dtype', 'fp16') != 'bf16', 'fp16': True, 'fp32': False}[args.precision]
    bs = args.batch_size or (4 if ds == 'pst900' else 8)
    data_root = args.data_root or c.get('data_root')
    if not data_root:
        ap.error('--data_root is required (the config has no data_root)')
    inter, union, names, lum = predict(model, c, device, data_root, amp, bs)

    res = {'run': args.run, 'dataset': ds, 'precision': 'fp16' if amp else 'fp32', 'n_images': len(names)}
    all_iou = iou(inter, union)
    res['miou'] = float(all_iou.mean())
    res['iou'] = dict(zip(CLASS_NAMES[ds], map(float, all_iou)))
    print(f"[{ds}] {args.run}: mIoU {res['miou']:.2f}  ({res['precision']}, {len(names)} images)")
    print('  ' + '  '.join(f'{k} {v:.2f}' for k, v in res['iou'].items()))
    if ds == 'mfnet':
        night = np.array([nm.endswith('N') for nm in names])
        res['miou_day'] = float(iou(inter, union, ~night).mean())
        res['miou_night'] = float(iou(inter, union, night).mean())
        print(f"  day {res['miou_day']:.2f} ({(~night).sum()} images)  night {res['miou_night']:.2f} ({night.sum()} images)")
    if ds == 'iron':
        low = lum < LOW_LIGHT_THR
        res['drivable_iou'] = float(all_iou[1])
        res['drivable_iou_low_light'] = float(iou(inter, union, low)[1])
        scene = np.array([nm.split('/')[0] for nm in names])
        res['drivable_iou_scene'] = {s: float(iou(inter, union, scene == s)[1]) for s in np.unique(scene)}
        print(f"  drivable IoU {res['drivable_iou']:.2f}  low light {res['drivable_iou_low_light']:.2f} "
              f"({low.sum()} frames)  " + '  '.join(f'{s} {v:.2f}' for s, v in res['drivable_iou_scene'].items()))
    if args.save:
        with open(args.save, 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
