#!/usr/bin/env python3
"""
Evaluate UNIStainNet on BCI dataset.

Computes: FID, KID, LPIPS, SSIM, PSNR, DAB metrics (per-class + overall),
IOD/mIOD, AUROC/SFS, and saves sample grids.

Usage:
    python scripts/eval/eval_bci.py \
        --checkpoint checkpoints/bci/last.ckpt \
        --data_dir /path/to/BCI_dataset

    # Skip expensive UNI-FID and downstream metrics:
    python scripts/eval/eval_bci.py \
        --checkpoint checkpoints/bci/last.ckpt \
        --data_dir /path/to/BCI_dataset \
        --skip_uni_fid --skip_downstream
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import timm
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

from src.models.trainer import UNIStainNetTrainer
from src.data.bci_dataset import BCICropDataModule
from src.utils.dab import DABExtractor
from src.utils.metrics import (
    compute_image_quality_metrics,
    compute_uni_fid,
    compute_dab_metrics,
    compute_iod_metrics,
    compute_downstream_metrics,
    save_sample_grid,
    composite_background,
)


def load_uni_model():
    """Load UNI ViT-L/16 for on-the-fly feature extraction during eval."""
    model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True,
                               init_values=1e-5, dynamic_img_size=True)
    model = model.cuda().eval()
    return model


def extract_features_for_crop(uni_model, he_crop_01, spatial_pool_size=32):
    """Extract UNI features from a 512x512 H&E crop.

    Splits into 4x4 sub-crops, runs UNI on each, reassembles spatial grid.

    Args:
        uni_model: UNI ViT-L/16 model on CUDA
        he_crop_01: [B, 3, 512, 512] in [0, 1]
        spatial_pool_size: target spatial grid size (32 = 32x32 = 1024 tokens)

    Returns:
        uni_features: [B, S*S, 1024]
    """
    uni_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    B = he_crop_01.shape[0]
    num_crops = 4
    patches_per_side = 14  # 224/16

    # Split into 4x4 sub-crops and resize to 224x224
    sub_crops = []
    crop_h = he_crop_01.shape[2] // num_crops
    crop_w = he_crop_01.shape[3] // num_crops
    for i in range(num_crops):
        for j in range(num_crops):
            sub = he_crop_01[:, :, i*crop_h:(i+1)*crop_h, j*crop_w:(j+1)*crop_w]
            sub = F.interpolate(sub, size=(224, 224), mode='bicubic', align_corners=False)
            sub = torch.stack([uni_transform(s) for s in sub])
            sub_crops.append(sub)

    # [B, 16, 3, 224, 224] -> [B*16, 3, 224, 224]
    all_crops = torch.stack(sub_crops, dim=1).reshape(B * 16, 3, 224, 224).cuda()

    with torch.no_grad():
        all_feats = uni_model.forward_features(all_crops)
        patch_tokens = all_feats[:, 1:, :]  # [B*16, 196, 1024]

    # Reassemble spatial grid
    patch_tokens = patch_tokens.reshape(
        B, num_crops, num_crops, patches_per_side, patches_per_side, 1024
    )
    full_size = num_crops * patches_per_side  # 56
    full_grid = patch_tokens.permute(0, 1, 3, 2, 4, 5).reshape(B, full_size, full_size, 1024)

    if spatial_pool_size < full_size:
        grid_bchw = full_grid.permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(grid_bchw, spatial_pool_size)
        result = pooled.permute(0, 2, 3, 1)
    else:
        result = full_grid

    S = result.shape[1]
    return result.reshape(B, S * S, 1024).cpu()


@torch.no_grad()
def generate_all(model, uni_model, dataloader, guidance_scale=1.0, seed=42):
    """Generate HER2 images for the entire test set."""
    all_gen, all_real, all_he, all_labels, all_fnames = [], [], [], [], []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Generating (cfg={guidance_scale})")):
        he, her2, uni_sub_crops, labels, fnames = batch
        he, her2 = he.cuda().float(), her2.cuda().float()
        labels = labels.cuda().long()

        # Extract UNI features on-the-fly
        he_01 = ((he + 1) / 2).clamp(0, 1)
        uni = extract_features_for_crop(uni_model, he_01).cuda()

        gen = model.generate(he, uni, labels,
                             guidance_scale=guidance_scale,
                             seed=seed + batch_idx)

        all_gen.append(gen.cpu())
        all_real.append(her2.cpu())
        all_he.append(he.cpu())
        all_labels.append(labels.cpu())
        all_fnames.extend(fnames)

    return (torch.cat(all_gen), torch.cat(all_real),
            torch.cat(all_he), torch.cat(all_labels), all_fnames)


def main():
    parser = argparse.ArgumentParser(description='Evaluate UNIStainNet on BCI')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to BCI_dataset directory')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--skip_uni_fid', action='store_true')
    parser.add_argument('--skip_downstream', action='store_true')
    parser.add_argument('--composite_bg', action='store_true',
                        help='Composite white background using H&E tissue mask')
    args = parser.parse_args()

    if args.output_dir is None:
        ckpt_name = Path(args.checkpoint).stem
        args.output_dir = f'eval_output/bci/{ckpt_name}'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EVALUATION: UNIStainNet on BCI")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir: {args.data_dir}")

    # Load model
    model = UNIStainNetTrainer.load_from_checkpoint(args.checkpoint, strict=False)
    model = model.cuda().eval()

    # Load UNI for on-the-fly feature extraction
    uni_model = load_uni_model()

    # Data
    dm = BCICropDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        image_size=(512, 512),
        crop_size=512,
    )
    dm.setup('test')
    test_loader = dm.test_dataloader()

    results = {
        'checkpoint': args.checkpoint,
        'guidance_scale': args.guidance_scale,
        'dataset': 'BCI',
    }

    # Generate
    print(f"\nGenerating (cfg={args.guidance_scale})...")
    gen, real, he, labels, fnames = generate_all(
        model, uni_model, test_loader, guidance_scale=args.guidance_scale)
    print(f"Generated {len(gen)} images")

    # Free UNI model (no longer needed for generation)
    del uni_model
    torch.cuda.empty_cache()

    if args.composite_bg:
        print("Compositing white background...")
        gen = composite_background(gen, he)

    # Sample grids
    print("\nSaving sample grids...")
    save_sample_grid(he, real, gen, output_dir / 'sample_grid.png', n=16)

    # Image quality
    print("\nComputing image quality metrics...")
    results['image_quality'] = compute_image_quality_metrics(gen, real)

    # DAB metrics (with per-class breakdown)
    print("Computing DAB metrics...")
    dab_extractor = DABExtractor(device='cpu')
    results['dab'] = compute_dab_metrics(gen, real, labels=labels, dab_extractor=dab_extractor)

    # IOD metrics
    print("Computing IOD/mIOD metrics...")
    results['iod'] = compute_iod_metrics(gen, real, labels=labels)

    # Downstream classifier (AUROC/SFS)
    if not args.skip_downstream:
        print("Computing AUROC/SFS (UNI linear probe)...")
        try:
            train_ihc_dir = Path(args.data_dir) / 'IHC' / 'train'
            results['downstream'] = compute_downstream_metrics(
                gen, real, labels, train_ihc_dir=str(train_ihc_dir))
        except Exception as e:
            print(f"  Downstream metrics skipped: {e}")

    # UNI-FID
    if not args.skip_uni_fid:
        print("Computing UNI-FID...")
        try:
            results['image_quality']['fid_uni'] = compute_uni_fid(gen, real)
        except Exception as e:
            print(f"  UNI-FID skipped: {e}")

    # Save results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary
    iq = results['image_quality']
    dab = results.get('dab', {})
    iod = results.get('iod', {})
    ds = results.get('downstream', {})

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (BCI)")
    print("=" * 70)

    print(f"\n--- Image Quality ---")
    print(f"  FID (Inception):     {iq.get('fid_inception', 'N/A'):.1f}")
    print(f"  KID (x1000):         {iq.get('kid_mean_x1000', 0):.1f}")
    print(f"  FID (UNI):           {iq.get('fid_uni', 'N/A')}")
    print(f"  LPIPS (512):         {iq['lpips_mean']:.4f}")
    print(f"  LPIPS (128):         {iq['lpips_128_mean']:.4f}")
    print(f"  SSIM:                {iq['ssim_mean']:.4f}")
    print(f"  PSNR:                {iq['psnr_mean']:.2f}")

    if dab:
        print(f"\n--- DAB Staining ---")
        print(f"  DAB MAE:             {dab['dab_mae_overall']:.4f}")
        if 'dab_pearson_r' in dab:
            print(f"  Pearson-R (overall): {dab['dab_pearson_r']:.4f}")
        if 'dab_pearson_r_within_class' in dab:
            print(f"  Pearson-R (w/in):    {dab['dab_pearson_r_within_class']:.4f}")
        if 'dab_kl' in dab:
            print(f"  DAB KL:              {dab['dab_kl']:.4f}")
        if 'dab_jsd' in dab:
            print(f"  DAB JSD:             {dab['dab_jsd']:.4f}")
        print(f"  Violations:          {dab.get('ordering_violations', '?')}/{dab.get('ordering_total_pairs', '?')}")

    if iod:
        print(f"\n--- IOD Metrics ---")
        print(f"  mIOD diff:           {iod['miod_diff']:.4f}")
        print(f"  IOD diff (x1e7):     {iod['iod_diff_1e7']:.3f}")

    if ds:
        print(f"\n--- Downstream (Star-Diff) ---")
        if 'auroc' in ds:
            print(f"  AUROC:               {ds['auroc']:.4f}")
        if 'sfs' in ds:
            print(f"  SFS:                 {ds['sfs']:.4f}")

    print("\nDONE")


if __name__ == '__main__':
    main()
