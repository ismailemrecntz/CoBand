from data.iron import IRONDataset
from data.mfnet import MFNetDataset
from data.pst900 import PST900Dataset
from data import iron, mfnet, pst900

CLASS_NAMES = {'mfnet': mfnet.CLASS_NAMES, 'pst900': pst900.CLASS_NAMES, 'iron': iron.CLASS_NAMES}

# (rgb_mean, rgb_std, thermal_mean, thermal_std) used for input normalization
NORM = {
    'mfnet': (mfnet.RGB_MEAN, mfnet.RGB_STD, mfnet.IR_MEAN, mfnet.IR_STD),
    'pst900': (pst900.RGB_MEAN, pst900.RGB_STD, pst900.THERMAL_MEAN, pst900.THERMAL_STD),
    'iron': (iron.RGB_MEAN, iron.RGB_STD, iron.IR_MEAN, iron.IR_STD),
}


def build_dataset(dataset, data_root, split, resolution, train_source='train_', strong_gauss_noise=False):
    """Training set (split='train') with augmentation, or the test set without augmentation."""
    train = split == 'train'
    if dataset == 'mfnet':
        if train:
            return MFNetDataset(data_root, train_source, have_label=True, strong_gauss_noise=strong_gauss_noise)
        return MFNetDataset(data_root, split, have_label=True)
    if dataset == 'pst900':
        if train:
            return PST900Dataset(data_dir=data_root, split='train', rgb_size=resolution, thermal_size=resolution,
                                 use_augmentation=True, strong_gauss_noise=strong_gauss_noise)
        return PST900Dataset(data_dir=data_root, split=split, rgb_size=resolution, thermal_size=resolution,
                             use_augmentation=False)
    if dataset == 'iron':
        if train:
            return IRONDataset(data_dir=data_root, split='train', rgb_size=resolution, ir_size=resolution,
                               use_augmentation=True, strong_gauss_noise=strong_gauss_noise)
        return IRONDataset(data_dir=data_root, split=split, rgb_size=resolution, ir_size=resolution,
                           use_augmentation=False)
    raise ValueError(f"Unknown dataset: {dataset}. Choose 'mfnet', 'pst900' or 'iron'.")
