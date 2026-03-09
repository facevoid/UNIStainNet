#!/usr/bin/env python3
"""
Train UNIStainNet on BCI at native 1024x1024 resolution.

Key differences from 512:
- image_size=1024 adds enc0 (1024->512) and dec1 (512->1024) levels (+0.2% params)
- No cropping — full 1024x1024 images used directly
- Discriminator/edge encoder operate at 512 internally
- Smaller batch size due to ~4x memory vs 512

Usage:
    python scripts/train/train_bci_1024.py --data_dir /path/to/BCI_dataset
    python scripts/train/train_bci_1024.py --data_dir /path/to/BCI_dataset --batch_size 4
"""

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.models.trainer import UNIStainNetTrainer
from src.data.bci_dataset import BCICropDataModule


def main():
    parser = argparse.ArgumentParser(description='Train UNIStainNet on BCI 1024')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/bci_1024')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (8 for 80GB A100, 4 if OOM)')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--wandb_name', type=str, default='bci_1024')
    parser.add_argument('--resume_from', type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING: UNIStainNet on BCI — NATIVE 1024x1024 (class_dim=64)")
    print("=" * 70)

    model = UNIStainNetTrainer(
        # Architecture
        num_classes=5,
        null_class=4,
        class_dim=64,
        uni_dim=1024,
        ndf=64,
        input_skip=True,
        edge_encoder='v2',
        edge_base_ch=32,
        uni_spatial_size=32,
        image_size=1024,
        label_names=['0', '1+', '2+', '3+'],
        # Optimizer
        gen_lr=1e-4,
        disc_lr=4e-4,
        warmup_steps=1000,
        # Loss weights (same as 512 BCI)
        lpips_weight=1.0,
        lpips_256_weight=0.5,
        lpips_512_weight=0.0,
        he_edge_weight=0.5,
        l1_lowres_weight=1.0,
        adversarial_weight=0.0,
        uncond_disc_weight=1.0,
        dab_intensity_weight=0.2,
        dab_contrast_weight=0.0,
        dab_sharpness_weight=0.0,
        gram_style_weight=0.0,
        edge_weight=0.0,
        crop_disc_weight=0.0,
        feat_match_weight=10.0,
        patchnce_weight=0.0,
        bg_white_weight=0.0,
        # GAN training
        r1_weight=10.0,
        r1_every=16,
        adversarial_start_step=500,
        # CFG dropout
        cfg_drop_class_prob=0.10,
        cfg_drop_uni_prob=0.10,
        cfg_drop_both_prob=0.05,
        # EMA
        ema_decay=0.999,
        # On-the-fly UNI extraction
        extract_uni_on_the_fly=True,
        uni_spatial_pool_size=32,
    )

    dm = BCICropDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        image_size=(1024, 1024),
        crop_size=1024,
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename='bci_1024_{epoch:03d}_{step:06d}',
        save_top_k=3,
        monitor='val/lpips',
        mode='min',
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    wandb_logger = WandbLogger(
        project='unistainnet',
        name=args.wandb_name,
        save_dir='wandb',
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=1,
        precision='bf16',
        callbacks=[ckpt_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=10,
        val_check_interval=1.0,
    )

    trainer.fit(model, dm, ckpt_path=args.resume_from)
    print("Training complete!")


if __name__ == "__main__":
    main()
