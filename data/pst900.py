"""PST900 RGB-thermal dataset (5 classes, 720 x 1280).

Expected layout (official release of Shivakumar et al.):
    PST900_RGBT_Dataset/
      train/{rgb,thermal,labels}/<name>.png
      test/{rgb,thermal,labels}/<name>.png

Classes: 0 background, 1 fire extinguisher, 2 backpack, 3 hand drill, 4 survivor (rescue mannequin).
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
THERMAL_MEAN = np.array([0.2266], dtype=np.float32)  # PST900 train set
THERMAL_STD = np.array([0.2804], dtype=np.float32)

CLASS_NAMES = ['background', 'fire_extinguisher', 'backpack', 'hand_drill', 'survivor']


class PST900Dataset(Dataset):
    # The order of the attributes set in __init__ must not change (see data/mfnet.py).
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        have_label: bool = True,
        rgb_size: Tuple[int, int] = (720, 1280),
        thermal_size: Tuple[int, int] = (720, 1280),
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
        self.thermal_size = thermal_size
        self.ignore_index = ignore_index
        self.use_augmentation = use_augmentation and (split == "train")

        rgb_dir = os.path.join(data_dir, split, "rgb")
        if not os.path.exists(rgb_dir):
            raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
        self.names: List[str] = [
            os.path.splitext(f)[0] for f in sorted(os.listdir(rgb_dir))
            if f.endswith('.png')
        ]
        self.n_data = len(self.names)
        print(f"[PST900Dataset] Loaded {self.n_data} samples from '{split}' split.")

        # 1. shared geometric augmentation
        if self.use_augmentation:
            self.shared_geom = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                    scale=(0.7, 1.3),
                    rotate=(-10, 10),
                    mode=cv2.BORDER_CONSTANT,
                    cval=0,
                    cval_mask=self.ignore_index,
                    p=0.5,
                ),
                A.OneOf([
                    A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                ], p=0.2),
                A.CoarseDropout(
                    num_holes_range=(1, 5),
                    hole_height_range=(20, 50),
                    hole_width_range=(20, 50),
                    fill_value=0,
                    p=0.2
                ),
            ], additional_targets={"thermal": "image"})
        else:
            self.shared_geom = A.Compose([], additional_targets={"thermal": "image"})

        # 2. resize
        self.resize_rgb = A.Resize(self.rgb_size[0], self.rgb_size[1], interpolation=cv2.INTER_LINEAR)
        self.resize_thermal = A.Resize(self.thermal_size[0], self.thermal_size[1], interpolation=cv2.INTER_LINEAR)
        self.resize_mask = A.Resize(self.thermal_size[0], self.thermal_size[1], interpolation=cv2.INTER_NEAREST)

        # 3. photometric augmentation, RGB
        rgb_pipeline = []
        if self.use_augmentation:
            rgb_pipeline.extend([
                A.OneOf([
                    A.GaussNoise(std_range=STRONG_NOISE_STD_RANGE if strong_gauss_noise
                                 else (np.sqrt(10.0) / 255, np.sqrt(50.0) / 255), noise_scale_factor=1.0, p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ], p=0.2),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            ])
        rgb_pipeline.extend([
            A.Normalize(mean=RGB_MEAN, std=RGB_STD),
            ToTensorV2()
        ])
        self.rgb_transform = A.Compose(rgb_pipeline)

        # 4. photometric augmentation, thermal
        thermal_pipeline = []
        if self.use_augmentation:
            thermal_pipeline.extend([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.GaussNoise(std_range=STRONG_NOISE_STD_RANGE if strong_gauss_noise
                             else (np.sqrt(5.0) / 255, np.sqrt(30.0) / 255), noise_scale_factor=1.0, p=0.2),
            ])
        thermal_pipeline.extend([
            A.Normalize(mean=THERMAL_MEAN, std=THERMAL_STD),
            ToTensorV2()
        ])
        self.thermal_transform = A.Compose(thermal_pipeline)

    def _read_rgb(self, name: str) -> np.ndarray:
        path = os.path.join(self.data_dir, self.split, "rgb", f"{name}.png")
        return np.array(Image.open(path).convert('RGB'), dtype=np.uint8)

    def _read_thermal(self, name: str) -> np.ndarray:
        path = os.path.join(self.data_dir, self.split, "thermal", f"{name}.png")
        return np.array(Image.open(path).convert('L'), dtype=np.uint8)

    def _read_mask(self, name: str) -> np.ndarray:
        path = os.path.join(self.data_dir, self.split, "labels", f"{name}.png")
        return np.array(Image.open(path).convert('L'), dtype=np.int32)

    def __getitem__(self, index: int):
        name = self.names[index]
        rgb = self._read_rgb(name)
        thermal = self._read_thermal(name)
        mask = self._read_mask(name) if self.have_label else None

        if mask is not None:
            augmented = self.shared_geom(image=rgb, thermal=thermal, mask=mask)
            rgb, thermal, mask = augmented['image'], augmented['thermal'], augmented['mask']
        else:
            augmented = self.shared_geom(image=rgb, thermal=thermal)
            rgb, thermal = augmented['image'], augmented['thermal']

        rgb = self.resize_rgb(image=rgb)['image']
        thermal = self.resize_thermal(image=thermal)['image']
        if mask is not None:
            mask = self.resize_mask(image=mask)['image']

        rgb = self.rgb_transform(image=rgb)['image']                      # [3, H, W]
        thermal = thermal[..., np.newaxis]                                # [H, W, 1]
        thermal = self.thermal_transform(image=thermal)['image']          # [1, H, W]

        if mask is not None:
            mask = torch.from_numpy(mask.copy()).long()
        else:
            mask = torch.zeros((self.rgb_size[0], self.rgb_size[1]), dtype=torch.long)
        return rgb, thermal, mask, name

    def __len__(self):
        return self.n_data
