#!/usr/bin/env python3
"""BCI ablation: CLS 4x4 tokens instead of 32x32 patch tokens."""

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.models.trainer import UNIStainNetTrainer
from src.data.bci_dataset import BCICropDataModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/bci_ablation_cls4x4')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--wandb_name', type=str, default='bci_ablation_cls4x4')
    args = parser.parse_args()

    model = UNIStainNetTrainer(
        num_classes=5, null_class=4, class_dim=64, uni_dim=1024, ndf=64,
        input_skip=True,
        edge_encoder='v2', edge_base_ch=32,
        uni_spatial_size=4,         # ABLATION: 4x4 CLS tokens instead of 32x32
        label_names=['0', '1+', '2+', '3+'],
        gen_lr=1e-4, disc_lr=4e-4, warmup_steps=1000,
        lpips_weight=1.0, lpips_256_weight=0.5, lpips_512_weight=0.0,
        he_edge_weight=0.5,
        l1_lowres_weight=1.0,
        adversarial_weight=0.0, uncond_disc_weight=1.0,
        dab_intensity_weight=0.2, dab_contrast_weight=0.0, dab_sharpness_weight=0.0,
        gram_style_weight=0.0, edge_weight=0.0, crop_disc_weight=0.0,
        feat_match_weight=10.0, patchnce_weight=0.0, bg_white_weight=0.0,
        r1_weight=10.0, r1_every=16, adversarial_start_step=2000,
        cfg_drop_class_prob=0.10, cfg_drop_uni_prob=0.10, cfg_drop_both_prob=0.05,
        ema_decay=0.999,
        extract_uni_on_the_fly=True,
        uni_spatial_pool_size=4,    # ABLATION: pool to 4x4
    )

    dm = BCICropDataModule(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=4, image_size=(512, 512), crop_size=512,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs, accelerator='gpu', devices=1, precision='bf16',
        callbacks=[
            ModelCheckpoint(dirpath=args.ckpt_dir, filename='cls4x4_{epoch:03d}_{step:06d}',
                          save_top_k=3, monitor='val/lpips', mode='min',
                          every_n_train_steps=1000, save_last=True),
            LearningRateMonitor(logging_interval='step'),
        ],
        logger=WandbLogger(project='unistainnet', name=args.wandb_name, save_dir='wandb'),
        log_every_n_steps=10, val_check_interval=1.0,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
