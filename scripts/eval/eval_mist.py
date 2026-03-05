#!/usr/bin/env python3
"""
Evaluate UNIStainNet on MIST dataset (per-stain evaluation).

For the unified multi-stain model, evaluates each stain separately by
setting the stain label conditioning. Computes image quality, DAB, and
IOD metrics, then reports per-stain and macro-averaged results.

Usage:
    # Evaluate all 4 stains:
    python scripts/eval/eval_mist.py \
        --checkpoint checkpoints/mist_multistain/last.ckpt \
        --data_dir /path/to/MIST

    # Evaluate specific stains:
    python scripts/eval/eval_mist.py \
        --checkpoint checkpoints/mist_multistain/last.ckpt \
        --data_dir /path/to/MIST \
        --stains HER2 Ki67
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
from src.data.bci_dataset import MISTCropDataModule
from src.data.mist_dataset import STAIN_TO_LABEL
from src.utils.dab import DABExtractor
from src.utils.metrics import (
    compute_image_quality_metrics,
    compute_uni_fid,
    compute_dab_metrics,
    compute_iod_metrics,
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
    """Extract UNI features from a 512x512 H&E crop."""
    uni_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    B = he_crop_01.shape[0]
    num_crops = 4
    patches_per_side = 14

    sub_crops = []
    crop_h = he_crop_01.shape[2] // num_crops
    crop_w = he_crop_01.shape[3] // num_crops
    for i in range(num_crops):
        for j in range(num_crops):
            sub = he_crop_01[:, :, i*crop_h:(i+1)*crop_h, j*crop_w:(j+1)*crop_w]
            sub = F.interpolate(sub, size=(224, 224), mode='bicubic', align_corners=False)
            sub = torch.stack([uni_transform(s) for s in sub])
            sub_crops.append(sub)

    all_crops = torch.stack(sub_crops, dim=1).reshape(B * 16, 3, 224, 224).cuda()

    with torch.no_grad():
        all_feats = uni_model.forward_features(all_crops)
        patch_tokens = all_feats[:, 1:, :]

    patch_tokens = patch_tokens.reshape(
        B, num_crops, num_crops, patches_per_side, patches_per_side, 1024
    )
    full_size = num_crops * patches_per_side
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
def generate_for_stain(model, uni_model, dataloader, stain_label, guidance_scale=1.0, seed=42):
    """Generate IHC images for a specific stain."""
    all_gen, all_real, all_he, all_fnames = [], [], [], []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Generating")):
        he, her2, uni_sub_crops, labels, fnames = batch
        he, her2 = he.cuda().float(), her2.cuda().float()

        # Override labels with stain label
        stain_labels = torch.full((he.size(0),), stain_label, device='cuda', dtype=torch.long)

        # Extract UNI features
        he_01 = ((he + 1) / 2).clamp(0, 1)
        uni = extract_features_for_crop(uni_model, he_01).cuda()

        gen = model.generate(he, uni, stain_labels,
                             guidance_scale=guidance_scale,
                             seed=seed + batch_idx)

        all_gen.append(gen.cpu())
        all_real.append(her2.cpu())
        all_he.append(he.cpu())
        all_fnames.extend(fnames)

    return torch.cat(all_gen), torch.cat(all_real), torch.cat(all_he), all_fnames


def main():
    parser = argparse.ArgumentParser(description='Evaluate UNIStainNet on MIST')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to MIST root directory')
    parser.add_argument('--stains', nargs='+', default=['HER2', 'Ki67', 'ER', 'PR'],
                        help='Stains to evaluate')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--skip_uni_fid', action='store_true')
    parser.add_argument('--composite_bg', action='store_true')
    args = parser.parse_args()

    if args.output_dir is None:
        ckpt_name = Path(args.checkpoint).stem
        args.output_dir = f'eval_output/mist/{ckpt_name}'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"EVALUATION: UNIStainNet on MIST")
    print(f"  Stains: {args.stains}")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")

    # Load model
    model = UNIStainNetTrainer.load_from_checkpoint(args.checkpoint, strict=False)
    model = model.cuda().eval()

    # Load UNI
    uni_model = load_uni_model()

    results = {
        'checkpoint': args.checkpoint,
        'guidance_scale': args.guidance_scale,
        'dataset': 'MIST',
        'stains': args.stains,
        'per_stain': {},
    }

    dab_extractor = DABExtractor(device='cpu')

    # Per-stain evaluation
    for stain in args.stains:
        stain_label = STAIN_TO_LABEL[stain]
        stain_lower = stain.lower()

        print(f"\n{'='*50}")
        print(f"EVALUATING: {stain} (label={stain_label})")
        print(f"{'='*50}")

        # Data for this stain
        stain_data_dir = Path(args.data_dir) / stain / 'TrainValAB'
        dm = MISTCropDataModule(
            data_dir=str(stain_data_dir),
            batch_size=args.batch_size,
            num_workers=4,
            image_size=(512, 512),
            crop_size=512,
            null_class=stain_label,  # Use stain label as the "class"
        )
        dm.setup('test')
        test_loader = dm.test_dataloader()

        # Generate
        gen, real, he, fnames = generate_for_stain(
            model, uni_model, test_loader, stain_label,
            guidance_scale=args.guidance_scale)
        print(f"Generated {len(gen)} images")

        if args.composite_bg:
            gen = composite_background(gen, he)

        # Save sample grid
        stain_dir = output_dir / stain_lower
        stain_dir.mkdir(parents=True, exist_ok=True)
        save_sample_grid(he, real, gen, stain_dir / 'sample_grid.png', n=16)

        stain_results = {}

        # Image quality
        print(f"  Computing image quality metrics...")
        stain_results['image_quality'] = compute_image_quality_metrics(gen, real)

        # DAB metrics (no class labels for MIST)
        print(f"  Computing DAB metrics...")
        stain_results['dab'] = compute_dab_metrics(gen, real, labels=None, dab_extractor=dab_extractor)

        # IOD metrics
        print(f"  Computing IOD metrics...")
        stain_results['iod'] = compute_iod_metrics(gen, real, labels=None)

        # UNI-FID (per-stain)
        if not args.skip_uni_fid:
            print(f"  Computing UNI-FID...")
            try:
                stain_results['image_quality']['fid_uni'] = compute_uni_fid(gen, real)
            except Exception as e:
                print(f"    UNI-FID skipped: {e}")

        results['per_stain'][stain] = stain_results

        # Print per-stain summary
        iq = stain_results['image_quality']
        dab = stain_results['dab']
        print(f"\n  {stain}: FID={iq['fid_inception']:.1f} | "
              f"KID={iq['kid_mean_x1000']:.1f} | "
              f"LPIPS={iq['lpips_mean']:.3f} | "
              f"SSIM={iq['ssim_mean']:.3f} | "
              f"Pearson-r={dab.get('dab_pearson_r', 0):.3f}")

    # Free UNI model
    del uni_model
    torch.cuda.empty_cache()

    # Macro-averaged summary
    print(f"\n{'='*70}")
    print("MACRO-AVERAGED RESULTS")
    print(f"{'='*70}")

    metric_keys = ['fid_inception', 'kid_mean_x1000', 'lpips_mean', 'lpips_128_mean',
                    'ssim_mean', 'psnr_mean']
    dab_keys = ['dab_mae_overall', 'dab_pearson_r', 'dab_kl', 'dab_jsd']
    iod_keys = ['miod_diff', 'miod_abs_diff']

    macro = {}
    for key in metric_keys:
        vals = [results['per_stain'][s]['image_quality'].get(key, float('nan'))
                for s in args.stains]
        macro[key] = float(np.mean([v for v in vals if not np.isnan(v)]))

    for key in dab_keys:
        vals = [results['per_stain'][s]['dab'].get(key, float('nan'))
                for s in args.stains]
        macro[key] = float(np.mean([v for v in vals if not np.isnan(v)]))

    for key in iod_keys:
        vals = [results['per_stain'][s]['iod'].get(key, float('nan'))
                for s in args.stains]
        macro[key] = float(np.mean([v for v in vals if not np.isnan(v)]))

    results['macro_average'] = macro

    # Print table
    header = f"{'Metric':<20s}"
    for s in args.stains:
        header += f" {s:>8s}"
    header += f" {'Macro':>8s}"
    print(header)
    print("-" * len(header))

    for key in metric_keys + dab_keys + iod_keys:
        row = f"{key:<20s}"
        for s in args.stains:
            if key in ['fid_inception', 'kid_mean_x1000']:
                src = 'image_quality'
            elif key.startswith('dab'):
                src = 'dab'
            else:
                src = 'iod'
            val = results['per_stain'][s][src].get(key, float('nan'))
            row += f" {val:>8.3f}"
        row += f" {macro.get(key, float('nan')):>8.3f}"
        print(row)

    # Save
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print("\nDONE")


if __name__ == '__main__':
    main()
