"""IRON infrared off-road dataset (binary free-space segmentation, 512 x 640).

Expected layout:
    IRON_Dataset/
      aligned_rgb/<scene>/<seq>/<name>.jpg
      images/<scene>/<seq>/<name>.jpg      thermal (3 identical channels, read as grayscale)
      anns/<scene>/<seq>/<name>.png        0 background, 1 drivable
      train.txt test.txt                   created with tools/make_iron_splits.py

See the note on augmentation arguments in data/mfnet.py.
"""
import os
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from data.mfnet import STRONG_NOISE_STD_RANGE

RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# thermal statistics of the original IRON configuration (108.375 / 255, 50.7195 / 255)
IR_MEAN = np.array([0.4250], dtype=np.float32)
IR_STD = np.array([0.1989], dtype=np.float32)

CLASS_NAMES = ['background', 'drivable']


class IRONDataset(Dataset):
    # The order of the attributes set in __init__ must not change (see data/mfnet.py).
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        have_label: bool = True,
        rgb_size: Tuple[int, int] = (512, 640),
        ir_size: Tuple[int, int] = (512, 640),
        ignore_index: int = 255,
        use_augmentation: bool = True,
        strong_gauss_noise: bool = False,
    ):
        super().__init__()
        assert split in ["train", "test"], f'Invalid split: {split}. Use "train" or "test".'
        self.data_dir = data_dir
        self.split = split
        self.have_label = have_label
        self.rgb_size = rgb_size
        self.ir_size = ir_size
        self.ignore_index = ignore_index
        self.use_augmentation = use_augmentation and (split == "train")

        split_file = os.path.join(data_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}\n"
                                    f"Run tools/make_iron_splits.py first to create train.txt and test.txt")
        with open(split_file, "r") as f:
            self.names: List[str] = [line.strip() for line in f if line.strip()]
        self.n_data = len(self.names)
        print(f"[IRONDataset] Loaded {self.n_data} samples from '{split}' split.")

        # 1. shared geometric augmentation
        if self.use_augmentation:
            self.shared_geom = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.3,
                    rotate_limit=10,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=self.ignore_index,
                    p=0.5,
                ),
                A.OneOf([
                    A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                    A.Perspective(scale=(0.05, 0.1), p=1.0),
                ], p=0.2),
                A.CoarseDropout(
                    max_holes=6,
                    max_height=48,
                    max_width=48,
                    min_holes=1,
                    min_height=16,
                    min_width=16,
                    fill_value=0,
                    mask_fill_value=self.ignore_index,
                    p=0.2
                ),
            ], additional_targets={"ir": "image"})
        else:
            self.shared_geom = A.Compose([], additional_targets={"ir": "image"})

        # 2. resize
        self.resize_rgb = A.Resize(self.rgb_size[0], self.rgb_size[1], interpolation=cv2.INTER_LINEAR)
        self.resize_ir = A.Resize(self.ir_size[0], self.ir_size[1], interpolation=cv2.INTER_LINEAR)
        self.resize_mask = A.Resize(self.ir_size[0], self.ir_size[1], interpolation=cv2.INTER_NEAREST)

        # 3. photometric augmentation, RGB
        rgb_pipeline = []
        if self.use_augmentation:
            rgb_pipeline.extend([
                A.OneOf([
                    A.GaussNoise(std_range=STRONG_NOISE_STD_RANGE if strong_gauss_noise
                                 else (np.sqrt(10.0) / 255, np.sqrt(50.0) / 255), p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ], p=0.2),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
                # low-light simulation
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.0), contrast_limit=(-0.3, 0.2), p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.1), contrast_limit=0.2, p=1.0),
                ], p=0.3),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            ])
        rgb_pipeline.extend([
            A.Normalize(mean=RGB_MEAN, std=RGB_STD),
            ToTensorV2()
        ])
        self.rgb_transform = A.Compose(rgb_pipeline)

        # 4. photometric augmentation, thermal
        ir_pipeline = []
        if self.use_augmentation:
            ir_pipeline.extend([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.GaussNoise(std_range=STRONG_NOISE_STD_RANGE if strong_gauss_noise
                             else (np.sqrt(5.0) / 255, np.sqrt(40.0) / 255), p=0.2),
            ])
        ir_pipeline.extend([
            A.Normalize(mean=IR_MEAN, std=IR_STD),
            ToTensorV2()
        ])
        self.ir_transform = A.Compose(ir_pipeline)

    def _read_rgb(self, name: str) -> np.ndarray:
        path = os.path.join(self.data_dir, "aligned_rgb", f"{name}.jpg")
        return np.array(Image.open(path).convert('RGB'), dtype=np.uint8)

    def _read_ir(self, name: str) -> np.ndarray:
        path = os.path.join(self.data_dir, "images", f"{name}.jpg")
        return np.array(Image.open(path).convert('L'), dtype=np.uint8)

    def _read_mask(self, name: str) -> np.ndarray:
        path = os.path.join(self.data_dir, "anns", f"{name}.png")
        return np.array(Image.open(path).convert('L'), dtype=np.int32)

    def __getitem__(self, index: int):
        name = self.names[index]
        rgb = self._read_rgb(name)
        ir = self._read_ir(name)
        mask = self._read_mask(name) if self.have_label else None

        if mask is not None:
            augmented = self.shared_geom(image=rgb, ir=ir, mask=mask)
            rgb, ir, mask = augmented['image'], augmented['ir'], augmented['mask']
        else:
            augmented = self.shared_geom(image=rgb, ir=ir)
            rgb, ir = augmented['image'], augmented['ir']

        rgb = self.resize_rgb(image=rgb)['image']
        ir = self.resize_ir(image=ir)['image']
        if mask is not None:
            mask = self.resize_mask(image=mask)['image']

        rgb = self.rgb_transform(image=rgb)['image']                      # [3, H, W]
        ir = ir[..., np.newaxis]                                          # [H, W, 1]
        ir = self.ir_transform(image=ir)['image']                         # [1, H, W]

        if mask is not None:
            mask = torch.from_numpy(mask.copy()).long()
            return rgb, ir, mask, name
        return rgb, ir, name

    def __len__(self):
        return self.n_data
