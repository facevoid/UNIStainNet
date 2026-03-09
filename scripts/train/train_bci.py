#!/usr/bin/env python3
"""
Train UNIStainNet on BCI dataset.

Random 512x512 crops from 1024x1024 native images, on-the-fly UNI feature extraction.

Usage:
    python scripts/train/train_bci.py --data_dir /path/to/BCI_dataset
    python scripts/train/train_bci.py --data_dir /path/to/BCI_dataset --batch_size 2
"""

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.models.trainer import UNIStainNetTrainer
from src.data.bci_dataset import BCICropDataModule


def main():
    parser = argparse.ArgumentParser(description='Train UNIStainNet on BCI')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to BCI_dataset directory (contains HE/ and IHC/)')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/bci',
                        help='Checkpoint save directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (16 for 80GB A100, 8 for 40GB)')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--wandb_name', type=str, default='bci_train',
                        help='Wandb run name')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from checkpoint path')
    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING: UNIStainNet on BCI")
    print("=" * 70)

    # Paper hyperparameters
    model = UNIStainNetTrainer(
        # Architecture
        num_classes=5,          # 4 HER2 classes (0, 1+, 2+, 3+) + null
        null_class=4,
        class_dim=64,
        uni_dim=1024,
        ndf=64,
        input_skip=True,
        edge_encoder='v2',
        edge_base_ch=32,
        uni_spatial_size=32,    # 32x32 patch tokens from UNI
        label_names=['0', '1+', '2+', '3+'],
        # Optimizer
        gen_lr=1e-4,
        disc_lr=4e-4,
        warmup_steps=1000,
        # Loss weights (paper configuration)
        lpips_weight=1.0,       # LPIPS@128 (primary perceptual loss)
        lpips_256_weight=0.5,   # LPIPS@256 (fine detail)
        lpips_512_weight=0.0,
        he_edge_weight=0.5,     # H&E edge preservation
        l1_lowres_weight=1.0,   # L1@64 (color fidelity)
        adversarial_weight=0.0, # No conditional discriminator
        uncond_disc_weight=1.0, # Unconditional discriminator
        dab_intensity_weight=0.2,  # DAB p90 matching
        dab_contrast_weight=0.0,
        dab_sharpness_weight=0.0,
        gram_style_weight=0.0,
        edge_weight=0.0,
        crop_disc_weight=0.0,
        feat_match_weight=10.0, # Feature matching loss
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
        image_size=(512, 512),
        crop_size=512,
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename='bci_{epoch:03d}_{step:06d}',
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
