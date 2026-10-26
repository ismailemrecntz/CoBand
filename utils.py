"""Shared helpers: model construction from a run configuration, IoU."""
import json

import numpy as np

from model import CoBandNet


def parse_int_list(s):
    """'[80,160,320,640]' or '7,5' -> [80, 160, 320, 640] / [7, 5]"""
    return [int(x.strip()) for x in str(s).strip('[]').split(',')]


def load_config(path):
    with open(path) as f:
        return json.load(f)


def build_model(c, pretrained=False):
    """CoBandNet for a run configuration (config.json written by train.py)."""
    H, W = c['img_height'], c['img_width']
    return CoBandNet(
        rgb_arch=c['rgb_arch'], ir_arch=c['ir_arch'], pretrained=pretrained, num_classes=c['num_classes'],
        context_dim=parse_int_list(c['context_dim']), output_resolution=(H, W),
        deep_supervision=bool(c.get('deep_supervision', False)),
        stage12=c.get('stage12', 'coband'), deep=c.get('deep', 'lsf'),
        fusion_type=c.get('fusion_type', 'default'),
        coband_sigma=float(c.get('coband_sigma', 2.0)),
        coband_kernels=tuple(parse_int_list(c.get('coband_kernels', '7,5'))),
        coband_learned_filter=bool(c.get('coband_learned_filter', False)),
        coband_gate=c.get('coband_gate', 'rgb_context'),
        modality=c.get('modality', 'fusion'))


# ---------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------
def iou_from_confusion(conf):
    """Per-class IoU = TP / (TP + FP + FN) from a confusion matrix (rows: ground truth, cols: prediction).
    Classes that occur neither in the ground truth nor in the prediction get NaN."""
    n = conf.shape[0]
    iou = np.zeros(n)
    for c in range(n):
        denom = conf[c, :].sum() + conf[:, c].sum() - conf[c, c]
        iou[c] = np.nan if denom == 0 else float(conf[c, c]) / float(denom)
    return iou
