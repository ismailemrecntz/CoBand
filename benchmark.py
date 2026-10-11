"""Parameters, MACs and latency (protocol of the paper): batch 1, eval mode, cudnn.benchmark, 50 warm-up and
300 timed iterations with CUDA events, fp32 and fp16 autocast. Weights are random (timing does not depend on them).

    python benchmark.py --config configs/mfnet_coband.json            # any config.json of a run also works
    python benchmark.py --config configs/pst900_coband.json --height 480 --width 640

MACs need the optional package thop (pip install thop).
"""
import argparse
import copy
import statistics

import torch

from utils import build_model, load_config


@torch.no_grad()
def time_model(model, inputs, amp, warmup=50, iters=300):
    for _ in range(warmup):
        with torch.autocast('cuda', enabled=amp):
            model(*inputs)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.autocast('cuda', enabled=amp):
            model(*inputs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    mean = statistics.mean(times)
    return mean, 1000.0 / mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--height', type=int, default=None, help='default: img_height of the config')
    ap.add_argument('--width', type=int, default=None, help='default: img_width of the config')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    c = load_config(args.config)
    c['img_height'] = args.height or c['img_height']
    c['img_width'] = args.width or c['img_width']
    H, W = c['img_height'], c['img_width']
    inputs = (torch.randn(1, 3, H, W, device=device), torch.randn(1, 1, H, W, device=device))
    model = build_model(c).to(device).eval()

    print(f"{args.config} @ {H}x{W}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")
    try:
        from thop import profile
        with torch.no_grad():
            macs, _ = profile(copy.deepcopy(model), inputs=inputs, verbose=False)
        print(f"  {macs / 1e9:.2f} GMACs")
    except ImportError:
        print("  (install thop for MACs)")
    for name, amp in (('fp32', False), ('fp16', True)):
        ms, fps = time_model(model, inputs, amp)
        print(f"  {name}: {ms:.2f} ms ({fps:.1f} FPS)")


if __name__ == '__main__':
    main()
