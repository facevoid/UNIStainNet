"""
Crop-based dataset loaders for training on random 512x512 crops from native 1024x1024.

Both BCI and MIST variants share the same crop + augmentation logic.
UNI features are extracted on-the-fly on GPU (not pre-computed).
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import pytorch_lightning as pl


class CropPairedDataset(Dataset):
    """Base class for random-crop paired H&E/IHC datasets.

    Loads 1024x1024 images, takes a random 512x512 crop (same position for both),
    and returns the crop + a UNI-ready version for on-the-fly feature extraction.
    """

    def __init__(
        self,
        he_dir: str,
        ihc_dir: str,
        image_size: Tuple[int, int] = (512, 512),
        crop_size: int = 512,
        augment: bool = False,
        labels: Optional[list] = None,
        null_class: int = 4,
    ):
        self.he_dir = Path(he_dir)
        self.ihc_dir = Path(ihc_dir)
        self.image_size = image_size
        self.crop_size = crop_size
        self.augment = augment
        self.null_class = null_class
        self.labels = labels

        # UNI normalization (ImageNet stats, 224x224 per sub-crop)
        self.uni_crop_transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def _random_crop_pair(self, he_img, ihc_img):
        """Take the same random 512x512 crop from both images."""
        w, h = he_img.size
        if w < self.crop_size or h < self.crop_size:
            raise ValueError(
                f"Image size {w}x{h} smaller than crop size {self.crop_size}"
            )
        if w == self.crop_size and h == self.crop_size:
            return he_img, ihc_img

        left = random.randint(0, w - self.crop_size)
        top = random.randint(0, h - self.crop_size)
        he_crop = he_img.crop((left, top, left + self.crop_size, top + self.crop_size))
        ihc_crop = ihc_img.crop((left, top, left + self.crop_size, top + self.crop_size))
        return he_crop, ihc_crop

    def _prepare_uni_sub_crops(self, he_pil):
        """Split 512x512 PIL crop into 4x4 sub-crops, each resized to 224x224 with UNI normalization.

        Returns: [16, 3, 224, 224] tensor ready for UNI forward pass on GPU.
        """
        w, h = he_pil.size
        num_crops = 4
        cw = w // num_crops
        ch = h // num_crops

        sub_crops = []
        for i in range(num_crops):
            for j in range(num_crops):
                left = j * cw
                top = i * ch
                sub = he_pil.crop((left, top, left + cw, top + ch))
                sub_crops.append(self.uni_crop_transform(sub))

        return torch.stack(sub_crops)  # [16, 3, 224, 224]

    def _apply_paired_augmentations(self, he_img, ihc_img):
        """Apply identical spatial transforms to both images."""
        if random.random() > 0.5:
            he_img = TF.hflip(he_img)
            ihc_img = TF.hflip(ihc_img)
        if random.random() > 0.5:
            he_img = TF.vflip(he_img)
            ihc_img = TF.vflip(ihc_img)
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            he_img = TF.rotate(he_img, k * 90)
            ihc_img = TF.rotate(ihc_img, k * 90)
        if random.random() > 0.7:
            angle = random.uniform(-15, 15)
            translate = [random.uniform(-0.05, 0.05) * self.image_size[1],
                         random.uniform(-0.05, 0.05) * self.image_size[0]]
            scale = random.uniform(0.9, 1.1)
            he_img = TF.affine(he_img, angle, translate, scale, shear=0,
                               interpolation=T.InterpolationMode.BILINEAR)
            ihc_img = TF.affine(ihc_img, angle, translate, scale, shear=0,
                                interpolation=T.InterpolationMode.BILINEAR)
        return he_img, ihc_img

    def _apply_he_color_augmentation(self, he_img):
        """Apply color jitter to H&E only (simulates staining variability)."""
        if random.random() > 0.5:
            he_img = TF.adjust_brightness(he_img, random.uniform(0.9, 1.1))
        if random.random() > 0.5:
            he_img = TF.adjust_contrast(he_img, random.uniform(0.9, 1.1))
        if random.random() > 0.5:
            he_img = TF.adjust_saturation(he_img, random.uniform(0.9, 1.1))
        return he_img

    def _process_pair(self, he_img, ihc_img, label, filename):
        """Common processing: crop -> augment -> tensorize -> UNI sub-crops.

        Returns: (he_tensor, ihc_tensor, uni_sub_crops, label, filename)
            - he_tensor: [3, 512, 512] in [-1, 1]
            - ihc_tensor: [3, 512, 512] in [-1, 1]
            - uni_sub_crops: [16, 3, 224, 224] with ImageNet normalization
        """
        # Random crop (same position for both)
        he_crop, ihc_crop = self._random_crop_pair(he_img, ihc_img)

        # Augmentations (applied to PIL before UNI extraction, so features match)
        if self.augment:
            he_crop, ihc_crop = self._apply_paired_augmentations(he_crop, ihc_crop)
            he_aug = self._apply_he_color_augmentation(he_crop)
        else:
            he_aug = he_crop

        # Prepare UNI sub-crops from the augmented H&E crop
        uni_sub_crops = self._prepare_uni_sub_crops(he_aug)

        # Convert to training tensors [-1, 1]
        he_tensor = TF.normalize(TF.to_tensor(he_aug), [0.5]*3, [0.5]*3)
        ihc_tensor = TF.normalize(TF.to_tensor(ihc_crop), [0.5]*3, [0.5]*3)

        return he_tensor, ihc_tensor, uni_sub_crops, label, filename


class BCICropDataset(CropPairedDataset):
    """BCI dataset with random 512 crops from 1024x1024 native images."""

    HER2_LABEL_MAP = {'0': 0, '1+': 1, '2+': 2, '3+': 3}

    def __init__(self, he_dir, ihc_dir, image_size=(512, 512),
                 crop_size=512, augment=False):
        super().__init__(he_dir, ihc_dir, image_size, crop_size, augment)

        self.he_images = sorted([f for f in os.listdir(he_dir) if f.endswith('.png')])
        self.ihc_images = sorted([f for f in os.listdir(ihc_dir) if f.endswith('.png')])
        assert len(self.he_images) == len(self.ihc_images)

        self.labels = [self._parse_label(f) for f in self.he_images]

        from collections import Counter
        dist = Counter(self.labels)
        print(f"BCI Crop Dataset: {len(self)} images, classes: {dict(sorted(dist.items()))}")

    def _parse_label(self, filename):
        parts = filename.replace('.png', '').split('_')
        if len(parts) >= 3:
            level = parts[2]
            if level in self.HER2_LABEL_MAP:
                return self.HER2_LABEL_MAP[level]
        raise ValueError(f"Cannot parse label from: {filename}")

    def __len__(self):
        return len(self.he_images)

    def __getitem__(self, idx):
        filename = self.he_images[idx]
        he_img = Image.open(self.he_dir / filename).convert('RGB')
        ihc_img = Image.open(self.ihc_dir / self.ihc_images[idx]).convert('RGB')
        return self._process_pair(he_img, ihc_img, self.labels[idx], filename)


class MISTCropDataset(CropPairedDataset):
    """MIST dataset with random 512 crops from 1024x1024 native images."""

    def __init__(self, he_dir, ihc_dir, image_size=(512, 512),
                 crop_size=512, augment=False, null_class=4):
        super().__init__(he_dir, ihc_dir, image_size, crop_size, augment,
                         null_class=null_class)

        valid_exts = ('.jpg', '.jpeg', '.png')
        self.he_images = sorted([f for f in os.listdir(he_dir)
                                 if f.lower().endswith(valid_exts)])
        self.ihc_images = sorted([f for f in os.listdir(ihc_dir)
                                  if f.lower().endswith(valid_exts)])

        # Verify pairing
        he_stems = {Path(f).stem for f in self.he_images}
        ihc_stems = {Path(f).stem for f in self.ihc_images}
        if he_stems != ihc_stems:
            common = he_stems & ihc_stems
            self.he_images = sorted([f for f in self.he_images if Path(f).stem in common])
            self.ihc_images = sorted([f for f in self.ihc_images if Path(f).stem in common])
            print(f"Using {len(self.he_images)} matched pairs")

        print(f"MIST Crop Dataset: {len(self)} images (null_class={null_class})")

    def __len__(self):
        return len(self.he_images)

    def __getitem__(self, idx):
        filename = self.he_images[idx]
        he_img = Image.open(self.he_dir / filename).convert('RGB')
        ihc_img = Image.open(self.ihc_dir / self.ihc_images[idx]).convert('RGB')
        return self._process_pair(he_img, ihc_img, self.null_class, filename)


class BCICropDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=4,
                 num_workers=4, image_size=(512, 512), crop_size=512):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.crop_size = crop_size

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = BCICropDataset(
                he_dir=self.data_dir / 'HE' / 'train',
                ihc_dir=self.data_dir / 'IHC' / 'train',
                image_size=self.image_size,
                crop_size=self.crop_size,
                augment=True,
            )
        if stage in ('fit', 'validate', 'test') or stage is None:
            self.val_dataset = BCICropDataset(
                he_dir=self.data_dir / 'HE' / 'test',
                ihc_dir=self.data_dir / 'IHC' / 'test',
                image_size=self.image_size,
                crop_size=self.crop_size,
                augment=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return self.val_dataloader()


class MISTCropDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=4,
                 num_workers=4, image_size=(512, 512), crop_size=512, null_class=4):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.crop_size = crop_size
        self.null_class = null_class

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MISTCropDataset(
                he_dir=self.data_dir / 'trainA',
                ihc_dir=self.data_dir / 'trainB',
                image_size=self.image_size,
                crop_size=self.crop_size,
                augment=True,
                null_class=self.null_class,
            )
        if stage in ('fit', 'validate', 'test') or stage is None:
            self.val_dataset = MISTCropDataset(
                he_dir=self.data_dir / 'valA',
                ihc_dir=self.data_dir / 'valB',
                image_size=self.image_size,
                crop_size=self.crop_size,
                augment=False,
                null_class=self.null_class,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return self.val_dataloader()
