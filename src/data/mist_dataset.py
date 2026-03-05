"""
Multi-stain dataset for training a single model on all MIST IHC stains.

Combines HER2, Ki67, ER, PR into one dataset, returning a stain label (0-3)
instead of a class label. Reuses the same crop + UNI sub-crop pipeline
from CropPairedDataset.

Stain label mapping:
    0 = HER2, 1 = Ki67, 2 = ER, 3 = PR, 4 = null (CFG dropout)
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pytorch_lightning as pl

from src.data.bci_dataset import CropPairedDataset


STAIN_TO_LABEL = {'HER2': 0, 'Ki67': 1, 'ER': 2, 'PR': 3}
LABEL_TO_STAIN = {v: k for k, v in STAIN_TO_LABEL.items()}


class MISTMultiStainCropDataset(CropPairedDataset):
    """Multi-stain MIST dataset with random 512 crops from native 1024x1024.

    Loads all 4 MIST stains into a single dataset. Each sample returns a
    stain label (0-3) as the conditioning signal, reusing the class embedding
    slot in the generator.
    """

    def __init__(
        self,
        base_dir: str,
        stains: List[str],
        split: str = 'train',
        image_size: Tuple[int, int] = (512, 512),
        crop_size: int = 512,
        augment: bool = False,
        null_class: int = 4,
    ):
        super().__init__(
            he_dir='.',  # placeholder, we override __getitem__
            ihc_dir='.',
            image_size=image_size,
            crop_size=crop_size,
            augment=augment,
            null_class=null_class,
        )

        self.base_dir = Path(base_dir)
        self.samples = []  # (he_path, ihc_path, stain_label)

        split_he = 'trainA' if split == 'train' else 'valA'
        split_ihc = 'trainB' if split == 'train' else 'valB'
        valid_exts = ('.jpg', '.jpeg', '.png')

        for stain in stains:
            if stain not in STAIN_TO_LABEL:
                raise ValueError(f"Unknown stain: {stain}. Must be one of {list(STAIN_TO_LABEL.keys())}")

            stain_label = STAIN_TO_LABEL[stain]
            he_dir = self.base_dir / stain / 'TrainValAB' / split_he
            ihc_dir = self.base_dir / stain / 'TrainValAB' / split_ihc

            if not he_dir.exists():
                raise FileNotFoundError(f"H&E directory not found: {he_dir}")
            if not ihc_dir.exists():
                raise FileNotFoundError(f"IHC directory not found: {ihc_dir}")

            he_files = sorted([f for f in os.listdir(he_dir)
                               if f.lower().endswith(valid_exts)])
            ihc_files = sorted([f for f in os.listdir(ihc_dir)
                                if f.lower().endswith(valid_exts)])

            # Match by stem (H&E may be .jpg, IHC may be .png)
            he_stems = {Path(f).stem: f for f in he_files}
            ihc_stems = {Path(f).stem: f for f in ihc_files}
            common = sorted(set(he_stems.keys()) & set(ihc_stems.keys()))

            for stem in common:
                self.samples.append((
                    he_dir / he_stems[stem],
                    ihc_dir / ihc_stems[stem],
                    stain_label,
                ))

            print(f"  {stain} ({split}): {len(common)} pairs")

        # Per-stain counts for logging
        from collections import Counter
        dist = Counter(s[2] for s in self.samples)
        stain_counts = {LABEL_TO_STAIN[k]: v for k, v in sorted(dist.items())}
        print(f"Multi-Stain Crop Dataset ({split}): {len(self.samples)} total | {stain_counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        he_path, ihc_path, stain_label = self.samples[idx]
        he_img = Image.open(he_path).convert('RGB')
        ihc_img = Image.open(ihc_path).convert('RGB')
        return self._process_pair(he_img, ihc_img, stain_label, he_path.name)


class MISTMultiStainCropDataModule(pl.LightningDataModule):
    """Lightning DataModule for multi-stain MIST training."""

    def __init__(
        self,
        base_dir: str,
        stains: Optional[List[str]] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (512, 512),
        crop_size: int = 512,
        null_class: int = 4,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.stains = stains or ['HER2', 'Ki67', 'ER', 'PR']
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.crop_size = crop_size
        self.null_class = null_class

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MISTMultiStainCropDataset(
                base_dir=self.base_dir,
                stains=self.stains,
                split='train',
                image_size=self.image_size,
                crop_size=self.crop_size,
                augment=True,
                null_class=self.null_class,
            )
        if stage in ('fit', 'validate', 'test') or stage is None:
            self.val_dataset = MISTMultiStainCropDataset(
                base_dir=self.base_dir,
                stains=self.stains,
                split='val',
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
