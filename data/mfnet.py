"""MFNet RGB-thermal dataset (9 classes, 480 x 640).

Expected layout (the release of Ha et al. / RTFNet):
    MFNet/
      images/<name>.png    4-channel PNG: RGB + thermal
      labels/<name>.png    class indices 0-8
      train.txt val.txt test.txt test_day.txt test_night.txt
      train_.txt           = train.txt followed by val.txt (see README); used for training in the paper

Note on augmentation: the arguments below are kept exactly as they were used for the paper models. With
albumentations 2.0.5 some of them are deprecated and silently ignored (e.g. ``value``/``mask_value`` of
ShiftScaleRotate, ``max_holes``/``max_height`` of CoarseDropout, ``alpha_affine`` of ElasticTransform), so
the defaults of the library apply. Do not "fix" them if you want to reproduce the released models.
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

RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IR_MEAN = np.array([0.3616], dtype=np.float32)
IR_STD = np.array([0.0765], dtype=np.float32)

# Gaussian noise strength used for the paper models (--strong_gauss_noise): std 0.2-0.44 of the [0, 255] range.
# Without the flag the intended strength (variance 10-80 px^2) is used.
STRONG_NOISE_STD_RANGE = (0.2, 0.44)

TRAIN_SPLITS = ("train", "train_")
CLASS_NAMES = ['unlabeled', 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone', 'bump']


class MFNetDataset(Dataset):
    # The order of the attributes set in __init__ must not change: the worker seeding in train.py enumerates
    # vars(dataset) to give each augmentation pipeline its own random stream.
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        have_label: bool = True,
        rgb_size: Tuple[int, int] = (480, 640),
        ir_size: Tuple[int, int] = (480, 640),
        ignore_index: int = 255,
        strong_gauss_noise: bool = False,
    ):
        super().__init__()
        assert split in ["train", "train_", "val", "test", "test_day", "test_night"], 'Invalid split'
        self.data_dir = data_dir
        self.split = split
        self.have_label = have_label
        self.rgb_size = rgb_size
        self.ir_size = ir_size
        self.ignore_index = ignore_index

        with open(os.path.join(data_dir, f"{split}.txt"), "r") as f:
            self.names: List[str] = [line.strip() for line in f if line.strip()]
        self.n_data = len(self.names)

        # 1. shared geometric augmentation (RGB, thermal and label)
        if split in TRAIN_SPLITS:
            self.shared_geom = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.5,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=self.ignore_index,
                    p=0.5,
                ),
                A.OneOf([
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                    A.Perspective(scale=(0.05, 0.1), p=1.0),
                ], p=0.3),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=64,
                    max_width=64,
                    min_holes=1,
                    min_height=16,
                    min_width=16,
                    fill_value=0,
                    mask_fill_value=self.ignore_index,
                    p=0.3
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
        if split in TRAIN_SPLITS:
            rgb_pipeline.extend([
                A.OneOf([
                    A.GaussNoise(std_range=STRONG_NOISE_STD_RANGE if strong_gauss_noise
                                 else (np.sqrt(10.0) / 255, np.sqrt(80.0) / 255), p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ], p=0.3),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15, p=0.6),
                # low-light simulation
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=(-0.7, 0.0), contrast_limit=(-0.4, 0.2), p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.1), contrast_limit=(-0.2, 0.3), p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.05), contrast_limit=0.2, p=1.0),
                ], p=0.5),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            ])
        rgb_pipeline.extend([
            A.Normalize(mean=RGB_MEAN, std=RGB_STD),
            ToTensorV2()
        ])
        self.rgb_transform = A.Compose(rgb_pipeline)

        # 4. photometric augmentation, thermal
        ir_pipeline = []
        if split in TRAIN_SPLITS:
            ir_pipeline.extend([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.GaussNoise(std_range=STRONG_NOISE_STD_RANGE if strong_gauss_noise
                             else (np.sqrt(10.0) / 255, np.sqrt(60.0) / 255), p=0.3),
            ])
        ir_pipeline.extend([
            A.Normalize(mean=IR_MEAN, std=IR_STD),
            ToTensorV2()
        ])
        self.ir_transform = A.Compose(ir_pipeline)

    def _read_image_4ch(self, name: str) -> np.ndarray:
        path = os.path.join(self.data_dir, "images", f"{name}.png")
        img = np.asarray(Image.open(path))
        if img.ndim == 2:
            raise ValueError(f"Image {path} is single-channel")
        return img[:, :, :4].copy() if img.shape[2] >= 4 else img

    def _read_mask(self, name: str) -> np.ndarray:
        path = os.path.join(self.data_dir, "labels", f"{name}.png")
        mask = np.asarray(Image.open(path))
        return mask.astype(np.int32).copy()

    def __getitem__(self, index: int):
        name = self.names[index]
        img4 = self._read_image_4ch(name)
        rgb = img4[..., :3].astype(np.uint8)
        ir = img4[..., 3:]

        mask = self._read_mask(name) if self.have_label else None

        # shared geometric augmentation before resizing
        if mask is not None:
            aug = self.shared_geom(image=rgb, ir=ir, mask=mask)
            rgb, ir, mask = aug["image"], aug["ir"], aug["mask"]
        else:
            aug = self.shared_geom(image=rgb, ir=ir)
            rgb, ir = aug["image"], aug["ir"]

        rgb = self.resize_rgb(image=rgb)["image"]
        ir = self.resize_ir(image=ir)["image"]
        if mask is not None:
            mask = self.resize_mask(image=mask)["image"]

        rgb = self.rgb_transform(image=rgb)["image"]
        ir = self.ir_transform(image=ir)["image"]

        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.long)
            return rgb, ir, mask, name
        return rgb, ir, name

    def __len__(self) -> int:
        return self.n_data
