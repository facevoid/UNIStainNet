"""
UNIStainNet: Pixel-Space UNI-Guided Virtual Staining Network.

Architecture:
    Generator: SPADE-UNet conditioned on UNI pathology features + stain/class embedding
    Discriminator: Multi-scale PatchGAN (512 + 256)
    Losses: LPIPS@128 + adversarial + DAB intensity + DAB contrast

References:
    - Park et al., "Semantic Image Synthesis with SPADE" (CVPR 2019)
    - Chen et al., "A general-purpose self-supervised model for pathology" (Nature Medicine 2024)
    - Isola et al., "Image-to-Image Translation with pix2pix" (CVPR 2017)
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import pytorch_lightning as pl
import torchvision
import wandb

from src.models.discriminator import (
    PatchDiscriminator, MultiScaleDiscriminator,
    hinge_loss_d, hinge_loss_g, r1_gradient_penalty, feature_matching_loss,
)
from src.models.generator import SPADEUNetGenerator
from src.models.losses import VGGFeatureExtractor, gram_matrix, PatchNCELoss
from src.utils.dab import DABExtractor


# ======================================================================
# Training Module
# ======================================================================

class UNIStainNetTrainer(pl.LightningModule):
    """PyTorch Lightning training module for UNIStainNet.

    Handles GAN training with manual optimization, CFG dropout, EMA, and
    all loss computations.
    """

    def __init__(
        self,
        # Architecture
        num_classes=5,
        null_class=4,
        class_dim=64,
        uni_dim=1024,
        ndf=64,
        disc_n_layers=3,
        input_skip=False,
        # Optimizer
        gen_lr=1e-4,
        disc_lr=4e-4,
        warmup_steps=1000,
        # Loss weights
        lpips_weight=1.0,
        lpips_256_weight=0.5,
        lpips_512_weight=0.0,
        adversarial_weight=1.0,
        dab_intensity_weight=0.1,
        dab_contrast_weight=0.05,
        dab_sharpness_weight=0.0,
        gram_style_weight=0.0,
        edge_weight=0.0,
        he_edge_weight=0.0,
        bg_white_weight=0.0,
        bg_threshold=0.85,
        l1_lowres_weight=0.0,
        edge_encoder=False,
        edge_base_ch=32,
        uni_spatial_size=4,
        uncond_disc_weight=0.0,
        crop_disc_weight=0.0,
        crop_size=128,
        feat_match_weight=0.0,
        patchnce_weight=0.0,
        patchnce_layers=(2, 3, 4),
        patchnce_n_patches=256,
        patchnce_temperature=0.07,
        # Ablation
        disable_uni=False,
        disable_class=False,
        # GAN training
        r1_weight=10.0,
        r1_every=16,
        adversarial_start_step=2000,
        # CFG
        cfg_drop_class_prob=0.10,
        cfg_drop_uni_prob=0.10,
        cfg_drop_both_prob=0.05,
        # EMA
        ema_decay=0.999,
        # On-the-fly UNI extraction (for crop-based training)
        extract_uni_on_the_fly=False,
        uni_spatial_pool_size=32,
        # Resolution
        image_size=512,
        # 1024 architecture: extend UNI SPADE to 512 level
        uni_spade_at_512=False,
        # Per-label names for multi-stain logging
        label_names=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.null_class = null_class

        # On-the-fly UNI feature extraction (loaded lazily on first use)
        self._uni_model = None
        self._uni_extract_on_the_fly = extract_uni_on_the_fly

        # Generator
        self.generator = SPADEUNetGenerator(
            num_classes=num_classes,
            class_dim=class_dim,
            uni_dim=uni_dim,
            input_skip=input_skip,
            edge_encoder=edge_encoder,
            edge_base_ch=edge_base_ch,
            uni_spatial_size=uni_spatial_size,
            image_size=image_size,
            uni_spade_at_512=uni_spade_at_512,
        )

        # Discriminator (global multi-scale)
        self.discriminator = MultiScaleDiscriminator(
            in_channels=6, ndf=ndf, n_layers=disc_n_layers,
        )

        # Crop discriminator (local full-res detail)
        if crop_disc_weight > 0:
            self.crop_discriminator = PatchDiscriminator(
                in_channels=6, ndf=ndf, n_layers=disc_n_layers,
            )
        else:
            self.crop_discriminator = None

        # Unconditional discriminator (HER2-only, alignment-free texture judge)
        # Also needed for feature matching loss (FM uses uncond disc features)
        if uncond_disc_weight > 0 or feat_match_weight > 0:
            self.uncond_discriminator = PatchDiscriminator(
                in_channels=3, ndf=ndf, n_layers=disc_n_layers,
            )
        else:
            self.uncond_discriminator = None

        # PatchNCE loss (contrastive, alignment-free: H&E input vs generated)
        if patchnce_weight > 0:
            # Encoder channel dims: {1: 64, 2: 128, 3: 256, 4: 512}
            enc_channels = {1: 64, 2: 128, 3: 256, 4: 512}
            layer_channels = {l: enc_channels[l] for l in patchnce_layers}
            self.patchnce_loss = PatchNCELoss(
                layer_channels=layer_channels,
                num_patches=patchnce_n_patches,
                temperature=patchnce_temperature,
            )
        else:
            self.patchnce_loss = None

        # EMA generator
        self.generator_ema = copy.deepcopy(self.generator)
        self.generator_ema.requires_grad_(False)

        # Losses
        self.lpips_fn = lpips.LPIPS(net='alex')
        self.lpips_fn.requires_grad_(False)
        self.lpips_fn.eval()

        self.dab_extractor = DABExtractor(device='cpu')

        # VGG feature extractor for Gram-matrix style loss
        if gram_style_weight > 0:
            self.vgg_extractor = VGGFeatureExtractor()
            self.vgg_extractor.requires_grad_(False)
            self.vgg_extractor.eval()
        else:
            self.vgg_extractor = None

        # Param counts
        n_gen = sum(p.numel() for p in self.generator.parameters())
        n_disc = sum(p.numel() for p in self.discriminator.parameters())
        n_crop = sum(p.numel() for p in self.crop_discriminator.parameters()) if self.crop_discriminator else 0
        n_uncond = sum(p.numel() for p in self.uncond_discriminator.parameters()) if self.uncond_discriminator else 0
        print(f"Generator: {n_gen:,} params")
        print(f"Discriminator: {n_disc:,} params (global) + {n_crop:,} (crop) + {n_uncond:,} (uncond)")

    def configure_optimizers(self):
        gen_params = list(self.generator.parameters())
        if self.patchnce_loss is not None:
            gen_params += list(self.patchnce_loss.parameters())
        opt_g = torch.optim.Adam(
            gen_params,
            lr=self.hparams.gen_lr,
            betas=(0.0, 0.999),
        )
        # All discriminator params in one optimizer
        disc_params = list(self.discriminator.parameters())
        if self.crop_discriminator is not None:
            disc_params += list(self.crop_discriminator.parameters())
        if self.uncond_discriminator is not None:
            disc_params += list(self.uncond_discriminator.parameters())
        opt_d = torch.optim.Adam(
            disc_params,
            lr=self.hparams.disc_lr,
            betas=(0.0, 0.999),
        )
        return [opt_g, opt_d]

    def _get_lr_scale(self):
        """Linear warmup."""
        if self.global_step < self.hparams.warmup_steps:
            return self.global_step / max(1, self.hparams.warmup_steps)
        return 1.0

    @torch.no_grad()
    def _update_ema(self):
        """Update EMA generator weights."""
        decay = self.hparams.ema_decay
        for p_ema, p in zip(self.generator_ema.parameters(), self.generator.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)

    def on_save_checkpoint(self, checkpoint):
        """Exclude frozen UNI model from checkpoint (it's reloaded on-the-fly)."""
        state_dict = checkpoint.get('state_dict', {})
        keys_to_remove = [k for k in state_dict if k.startswith('_uni_model.')]
        for k in keys_to_remove:
            del state_dict[k]

    def on_load_checkpoint(self, checkpoint):
        """Filter out UNI model keys from old checkpoints that included them."""
        state_dict = checkpoint.get('state_dict', {})
        keys_to_remove = [k for k in state_dict if k.startswith('_uni_model.')]
        for k in keys_to_remove:
            del state_dict[k]

    def _load_uni_model(self):
        """Lazily load UNI ViT-L/16 for on-the-fly feature extraction."""
        if self._uni_model is None:
            import timm
            self._uni_model = timm.create_model(
                "hf-hub:MahmoodLab/uni",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True,
            )
            self._uni_model.eval()
            self._uni_model.requires_grad_(False)
            self._uni_model = self._uni_model.to(self.device)
            n_params = sum(p.numel() for p in self._uni_model.parameters())
            print(f"UNI model loaded for on-the-fly extraction: {n_params:,} params")
        return self._uni_model

    @torch.no_grad()
    def _extract_uni_from_sub_crops(self, uni_sub_crops):
        """Extract UNI features from pre-prepared sub-crops on GPU.

        Args:
            uni_sub_crops: [B, 16, 3, 224, 224] — batch of 4x4 sub-crop grids,
                           already normalized with ImageNet stats.

        Returns:
            uni_features: [B, S*S, 1024] where S = uni_spatial_pool_size (default 32)
        """
        uni_model = self._load_uni_model()
        B = uni_sub_crops.shape[0]
        spatial_size = self.hparams.uni_spatial_pool_size
        num_crops = 4  # 4x4 grid
        patches_per_side = 14  # 224/16

        # Batched UNI forward: [B, 16, 3, 224, 224] -> [B*16, 3, 224, 224]
        all_crops = uni_sub_crops.reshape(B * 16, 3, 224, 224).to(self.device)
        all_feats = uni_model.forward_features(all_crops)  # [B*16, 197, 1024]
        patch_tokens = all_feats[:, 1:, :]  # [B*16, 196, 1024]

        # Reshape back to per-sample grids: [B, 4, 4, 14, 14, 1024]
        patch_tokens = patch_tokens.reshape(
            B, num_crops, num_crops,
            patches_per_side, patches_per_side, 1024
        )
        # Interleave to spatial grid: [B, 56, 56, 1024]
        full_size = num_crops * patches_per_side  # 56
        full_grid = patch_tokens.permute(0, 1, 3, 2, 4, 5)
        full_grid = full_grid.reshape(B, full_size, full_size, 1024)

        # Pool to target spatial size (batched)
        if spatial_size < full_size:
            grid_bchw = full_grid.permute(0, 3, 1, 2)  # [B, 1024, 56, 56]
            pooled = F.adaptive_avg_pool2d(grid_bchw, spatial_size)  # [B, 1024, S, S]
            result = pooled.permute(0, 2, 3, 1)  # [B, S, S, 1024]
        else:
            result = full_grid

        S = result.shape[1]
        return result.reshape(B, S * S, 1024)  # [B, S*S, 1024]

    def _apply_cfg_dropout(self, labels, uni_features):
        """Apply classifier-free guidance dropout during training (vectorized)."""
        B = labels.shape[0]
        device = labels.device

        new_labels = labels.clone()
        new_uni = uni_features.clone()

        r = torch.rand(B, device=device)
        p_both = self.hparams.cfg_drop_both_prob
        p_class = p_both + self.hparams.cfg_drop_class_prob
        p_uni = p_class + self.hparams.cfg_drop_uni_prob

        drop_both = r < p_both
        drop_class = (r >= p_both) & (r < p_class)
        drop_uni = (r >= p_class) & (r < p_uni)

        new_labels[drop_both | drop_class] = self.null_class
        new_uni[drop_both | drop_uni] = 0.0

        return new_labels, new_uni

    def compute_dab_intensity_loss(self, generated, target):
        """Top-10% percentile matching for DAB intensity."""
        with torch.amp.autocast('cuda', enabled=False):
            gen = generated.float()
            tgt = target.float()

            dab_gen = self.dab_extractor.extract_dab_intensity(gen, normalize="none")
            dab_tgt = self.dab_extractor.extract_dab_intensity(tgt, normalize="none")

            def _batched_top10_mean(dab):
                """Compute mean of top-10% DAB intensity per sample (vectorized)."""
                B = dab.shape[0]
                flat = dab.reshape(B, -1)  # [B, H*W]
                p99 = torch.quantile(flat, 0.99, dim=1, keepdim=True)
                flat = flat.clamp(max=p99)
                p90 = torch.quantile(flat, 0.9, dim=1, keepdim=True)
                mask = flat >= p90  # [B, H*W]
                # Use masked mean: sum(vals * mask) / sum(mask), fallback to flat mean
                masked_sum = (flat * mask).sum(dim=1)
                mask_count = mask.sum(dim=1).clamp(min=1)
                return masked_sum / mask_count  # [B]

            gen_scores = _batched_top10_mean(dab_gen)
            tgt_scores = _batched_top10_mean(dab_tgt)
            return F.l1_loss(gen_scores, tgt_scores)

    def compute_dab_contrast_loss(self, generated, labels):
        """Class-ordering hinge loss: DAB(3+) > DAB(2+) > DAB(1+) > DAB(0)."""
        with torch.amp.autocast('cuda', enabled=False):
            gen = generated.float()
            # Only use non-null labels
            valid = labels < self.null_class
            if valid.sum() < 2:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            gen_valid = gen[valid]
            labels_valid = labels[valid]

            dab_gen = self.dab_extractor.extract_dab_intensity(gen_valid, normalize="none")

            B = dab_gen.shape[0]
            flat = dab_gen.reshape(B, -1)
            p99 = torch.quantile(flat, 0.99, dim=1, keepdim=True)
            flat = flat.clamp(max=p99)
            p90 = torch.quantile(flat, 0.9, dim=1, keepdim=True)
            mask = flat >= p90
            masked_sum = (flat * mask).sum(dim=1)
            mask_count = mask.sum(dim=1).clamp(min=1)
            dab_scores = masked_sum / mask_count

            class_pairs = [
                (3, 0, 0.20), (3, 1, 0.15),
                (2, 0, 0.08), (3, 2, 0.10),
            ]

            losses = []
            for high_cls, low_cls, margin in class_pairs:
                high_mask = labels_valid == high_cls
                low_mask = labels_valid == low_cls
                if high_mask.sum() > 0 and low_mask.sum() > 0:
                    high_score = dab_scores[high_mask].mean()
                    low_score = dab_scores[low_mask].mean()
                    losses.append(F.relu(margin - (high_score - low_score)))

            if losses:
                return torch.stack(losses).mean()
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def compute_edge_loss(self, generated, target):
        """Fourier spectral loss at 256x256 for boundary sharpness.

        Compares power spectrum magnitudes between generated and target.
        The Fourier magnitude is inherently translation-invariant — shifting
        an image doesn't change its frequency content — so this is robust to
        the ~30px misalignment in consecutive-cut BCI pairs.

        Focuses on high-frequency bands (outer 75% of spectrum) where
        blurriness manifests as reduced power.
        """
        with torch.amp.autocast('cuda', enabled=False):
            gen = F.interpolate(generated.float(), size=256, mode='bilinear', align_corners=False)
            tgt = F.interpolate(target.float(), size=256, mode='bilinear', align_corners=False)

            # Grayscale
            gen_gray = gen.mean(dim=1, keepdim=True)
            tgt_gray = tgt.mean(dim=1, keepdim=True)

            # 2D FFT -> power spectrum (log-scale for stability)
            gen_fft = torch.fft.fft2(gen_gray)
            tgt_fft = torch.fft.fft2(tgt_gray)
            gen_mag = torch.log1p(gen_fft.abs())
            tgt_mag = torch.log1p(tgt_fft.abs())

            # High-frequency mask: keep outer 75% of spectrum
            H, W = gen_mag.shape[-2], gen_mag.shape[-1]
            cy, cx = H // 2, W // 2
            y = torch.arange(H, device=gen.device).float() - cy
            x = torch.arange(W, device=gen.device).float() - cx
            dist = (y[:, None] ** 2 + x[None, :] ** 2).sqrt()
            max_dist = (cy ** 2 + cx ** 2) ** 0.5
            hf_mask = (dist > 0.25 * max_dist).float()

            # L1 on high-frequency magnitudes
            return F.l1_loss(gen_mag * hf_mask, tgt_mag * hf_mask)

    def compute_dab_sharpness_loss(self, generated, target):
        """DAB spatial sharpness loss: penalizes diffuse brown, rewards membrane-localized DAB.

        Two components:
        1. DAB gradient magnitude: mean Sobel gradient magnitude per image.
        2. DAB local variance distribution: sorted-L1 (Wasserstein-1) on
           patch variance vectors.
        """
        with torch.amp.autocast('cuda', enabled=False):
            gen = generated.float()
            tgt = target.float()

            dab_gen = self.dab_extractor.extract_dab_intensity(gen, normalize="none")
            dab_tgt = self.dab_extractor.extract_dab_intensity(tgt, normalize="none")

            # Ensure [B, 1, H, W]
            if dab_gen.dim() == 3:
                dab_gen = dab_gen.unsqueeze(1)
            if dab_tgt.dim() == 3:
                dab_tgt = dab_tgt.unsqueeze(1)

            B = dab_gen.shape[0]

            # --- Component 1: Gradient magnitude (batched) ---
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=torch.float32, device=gen.device).view(1, 1, 3, 3)
            sobel_y = sobel_x.transpose(-1, -2)

            gx_gen = F.conv2d(dab_gen, sobel_x, padding=1)
            gy_gen = F.conv2d(dab_gen, sobel_y, padding=1)
            grad_gen = (gx_gen**2 + gy_gen**2 + 1e-8).sqrt()

            gx_tgt = F.conv2d(dab_tgt, sobel_x, padding=1)
            gy_tgt = F.conv2d(dab_tgt, sobel_y, padding=1)
            grad_tgt = (gx_tgt**2 + gy_tgt**2 + 1e-8).sqrt()

            # Match mean gradient magnitude per image
            grad_loss = F.l1_loss(grad_gen.mean(dim=[1, 2, 3]), grad_tgt.mean(dim=[1, 2, 3]))

            # --- Component 2: Local variance distribution (sorted-L1) ---
            ps = 16  # patch size
            var_losses = []
            for i in range(B):
                g = dab_gen[i, 0]  # [H, W]
                t = dab_tgt[i, 0]

                H, W = g.shape
                nH, nW = H // ps, W // ps
                g_patches = g[:nH*ps, :nW*ps].reshape(nH, ps, nW, ps).permute(0, 2, 1, 3).reshape(-1, ps*ps)
                t_patches = t[:nH*ps, :nW*ps].reshape(nH, ps, nW, ps).permute(0, 2, 1, 3).reshape(-1, ps*ps)

                g_var = g_patches.var(dim=1)
                t_var = t_patches.var(dim=1)

                g_sorted, _ = g_var.sort()
                t_sorted, _ = t_var.sort()
                var_losses.append(F.l1_loss(g_sorted, t_sorted.detach()))

            var_loss = torch.stack(var_losses).mean()

            return grad_loss + var_loss

    def compute_he_edge_loss(self, generated, he_input):
        """H&E edge structure preservation loss.

        Extracts Sobel edges from H&E input and generated output, then
        computes L1 loss between edge maps at multiple scales.
        """
        with torch.amp.autocast('cuda', enabled=False):
            gen = generated.float()
            he = he_input.float()

            gen_gray = ((gen + 1) / 2).mean(dim=1, keepdim=True)
            he_gray = ((he + 1) / 2).mean(dim=1, keepdim=True)

            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=torch.float32, device=gen.device).view(1, 1, 3, 3)
            sobel_y = sobel_x.transpose(-1, -2)

            loss = 0.0
            full_size = gen_gray.shape[-1]
            scales = [full_size, full_size // 2]
            for size in scales:
                if size < full_size:
                    g = F.interpolate(gen_gray, size=size, mode='bilinear', align_corners=False)
                    h = F.interpolate(he_gray, size=size, mode='bilinear', align_corners=False)
                else:
                    g, h = gen_gray, he_gray

                gx_gen = F.conv2d(g, sobel_x, padding=1)
                gy_gen = F.conv2d(g, sobel_y, padding=1)
                edge_gen = (gx_gen**2 + gy_gen**2 + 1e-8).sqrt()

                gx_he = F.conv2d(h, sobel_x, padding=1)
                gy_he = F.conv2d(h, sobel_y, padding=1)
                edge_he = (gx_he**2 + gy_he**2 + 1e-8).sqrt()

                loss = loss + F.l1_loss(edge_gen, edge_he.detach())

            return loss / 2.0

    def compute_background_loss(self, generated, he_input):
        """Background white loss: push background regions toward white."""
        with torch.amp.autocast('cuda', enabled=False):
            gen = generated.float()
            he = he_input.float()

            he_bright = ((he + 1) / 2).mean(dim=1, keepdim=True)

            threshold = self.hparams.bg_threshold
            mask = torch.sigmoid((he_bright - threshold) * 20.0)

            white_target = torch.ones_like(gen)
            diff = (gen - white_target).abs()

            weighted_diff = diff * mask

            mask_sum = mask.sum() * 3
            if mask_sum > 0:
                return weighted_diff.sum() / mask_sum
            return torch.tensor(0.0, device=gen.device, requires_grad=True)

    def compute_gram_style_loss(self, generated, target):
        """Gram-matrix style loss: match texture statistics via VGG feature correlations."""
        with torch.amp.autocast('cuda', enabled=False):
            gen = generated.float()
            tgt = target.float()

            gen_256 = F.interpolate(gen, size=256, mode='bilinear', align_corners=False)
            tgt_256 = F.interpolate(tgt, size=256, mode='bilinear', align_corners=False)

            gen_feats = self.vgg_extractor(gen_256)
            tgt_feats = self.vgg_extractor(tgt_256)

            loss = 0.0
            for gf, tf in zip(gen_feats, tgt_feats):
                gram_gen = gram_matrix(gf)
                gram_tgt = gram_matrix(tf)
                loss = loss + F.l1_loss(gram_gen, gram_tgt.detach())

            return loss / len(gen_feats)

    def training_step(self, batch, batch_idx):
        he, her2, uni_or_crops, labels, fnames = batch
        opt_g, opt_d = self.optimizers()

        # On-the-fly UNI extraction: dataset returns [B, 16, 3, 224, 224] sub-crops
        if self._uni_extract_on_the_fly:
            uni = self._extract_uni_from_sub_crops(uni_or_crops)
        else:
            uni = uni_or_crops

        # Apply CFG dropout
        labels_dropped, uni_dropped = self._apply_cfg_dropout(labels, uni)

        # Ablation: zero out UNI features
        if self.hparams.disable_uni:
            uni_dropped = torch.zeros_like(uni_dropped)

        # Ablation: force all labels to null class
        if self.hparams.disable_class:
            labels_dropped = torch.full_like(labels_dropped, self.hparams.null_class)

        # ----------------------------------------------------------------
        # Generator step
        # ----------------------------------------------------------------
        generated = self.generator(he, uni_dropped, labels_dropped)

        # LPIPS main: 4x downsample (128 for 512 input, 256 for 1024)
        lpips_main_size = self.hparams.image_size // 4
        gen_lpips = F.interpolate(generated, size=lpips_main_size, mode='bilinear', align_corners=False)
        her2_lpips = F.interpolate(her2, size=lpips_main_size, mode='bilinear', align_corners=False)
        loss_lpips = self.lpips_fn(gen_lpips, her2_lpips).mean()

        loss_g = self.hparams.lpips_weight * loss_lpips

        # LPIPS fine: 2x downsample (256 for 512 input, 512 for 1024)
        if self.hparams.lpips_256_weight > 0:
            lpips_fine_size = self.hparams.image_size // 2
            gen_fine = F.interpolate(generated, size=lpips_fine_size, mode='bilinear', align_corners=False)
            her2_fine = F.interpolate(her2, size=lpips_fine_size, mode='bilinear', align_corners=False)
            loss_lpips_256 = self.lpips_fn(gen_fine, her2_fine).mean()
            loss_g = loss_g + self.hparams.lpips_256_weight * loss_lpips_256
            self.log('train/lpips_fine', loss_lpips_256, prog_bar=False)

        # LPIPS at full resolution (expensive)
        if self.hparams.lpips_512_weight > 0:
            loss_lpips_512 = self.lpips_fn(generated, her2).mean()
            loss_g = loss_g + self.hparams.lpips_512_weight * loss_lpips_512
            self.log('train/lpips_fullres', loss_lpips_512, prog_bar=False)

        # Low-resolution L1 (color fidelity, misalignment-robust at 64x64)
        if self.hparams.l1_lowres_weight > 0:
            gen_64 = F.interpolate(generated, size=64, mode='bilinear', align_corners=False)
            her2_64 = F.interpolate(her2, size=64, mode='bilinear', align_corners=False)
            loss_l1_lowres = F.l1_loss(gen_64, her2_64)
            loss_g = loss_g + self.hparams.l1_lowres_weight * loss_l1_lowres
            self.log('train/l1_lowres', loss_l1_lowres, prog_bar=False)

        # DAB losses (use original labels, not dropped)
        if self.hparams.dab_intensity_weight > 0:
            loss_dab = self.compute_dab_intensity_loss(generated, her2)
            loss_g = loss_g + self.hparams.dab_intensity_weight * loss_dab
            self.log('train/dab_intensity', loss_dab, prog_bar=False)

        if self.hparams.dab_contrast_weight > 0:
            loss_dab_contrast = self.compute_dab_contrast_loss(generated, labels)
            loss_g = loss_g + self.hparams.dab_contrast_weight * loss_dab_contrast
            self.log('train/dab_contrast', loss_dab_contrast, prog_bar=False)

        # Edge loss (boundary sharpness)
        if self.hparams.edge_weight > 0:
            loss_edge = self.compute_edge_loss(generated, her2)
            loss_g = loss_g + self.hparams.edge_weight * loss_edge
            self.log('train/edge_loss', loss_edge, prog_bar=False)

        # DAB sharpness loss (membrane-localized vs diffuse brown)
        if self.hparams.dab_sharpness_weight > 0:
            loss_dab_sharp = self.compute_dab_sharpness_loss(generated, her2)
            loss_g = loss_g + self.hparams.dab_sharpness_weight * loss_dab_sharp
            self.log('train/dab_sharpness', loss_dab_sharp, prog_bar=False)

        # Gram-matrix style loss
        if self.hparams.gram_style_weight > 0 and self.vgg_extractor is not None:
            loss_gram = self.compute_gram_style_loss(generated, her2)
            loss_g = loss_g + self.hparams.gram_style_weight * loss_gram
            self.log('train/gram_style', loss_gram, prog_bar=False)

        # H&E edge structure preservation (pixel-aligned)
        if self.hparams.he_edge_weight > 0:
            loss_he_edge = self.compute_he_edge_loss(generated, he)
            loss_g = loss_g + self.hparams.he_edge_weight * loss_he_edge
            self.log('train/he_edge', loss_he_edge, prog_bar=False)

        # Background white loss
        if self.hparams.bg_white_weight > 0:
            loss_bg = self.compute_background_loss(generated, he)
            loss_g = loss_g + self.hparams.bg_white_weight * loss_bg
            self.log('train/bg_white', loss_bg, prog_bar=False)

        # PatchNCE loss (contrastive: H&E input vs generated, never sees GT)
        if self.hparams.patchnce_weight > 0 and self.patchnce_loss is not None:
            feats_he = self.generator.encode(he)
            feats_gen = self.generator.encode(generated)
            loss_nce = self.patchnce_loss(feats_he, feats_gen)
            loss_g = loss_g + self.hparams.patchnce_weight * loss_nce
            self.log('train/patchnce', loss_nce, prog_bar=False)

        # Adversarial losses (after warmup)
        loss_adv = torch.tensor(0.0, device=self.device)
        loss_feat_match = torch.tensor(0.0, device=self.device)
        loss_crop_adv = torch.tensor(0.0, device=self.device)
        loss_uncond_adv = torch.tensor(0.0, device=self.device)
        any_adv = (self.hparams.adversarial_weight > 0 or
                   self.hparams.uncond_disc_weight > 0 or
                   self.hparams.crop_disc_weight > 0 or
                   self.hparams.feat_match_weight > 0)
        img_sz = self.hparams.image_size
        # Pre-compute disc-resolution tensors (512 for 1024 input, identity for 512)
        if img_sz == 1024:
            he_for_disc = F.interpolate(he, size=512, mode='bilinear', align_corners=False)
            her2_for_disc = F.interpolate(her2, size=512, mode='bilinear', align_corners=False)
        else:
            he_for_disc = he
            her2_for_disc = her2
        if self.global_step >= self.hparams.adversarial_start_step and any_adv:
            if img_sz == 1024:
                gen_for_disc = F.interpolate(generated, size=512, mode='bilinear', align_corners=False)
            else:
                gen_for_disc = generated

            # Conditional discriminator (paired: generated+HE vs real_HER2+HE)
            if self.hparams.adversarial_weight > 0:
                fake_input = torch.cat([gen_for_disc, he_for_disc], dim=1)
                disc_outputs = self.discriminator(fake_input)
                loss_adv = sum(hinge_loss_g(out) for out in disc_outputs) / len(disc_outputs)
                loss_g = loss_g + self.hparams.adversarial_weight * loss_adv

            # Feature matching from unconditional disc
            if (self.hparams.feat_match_weight > 0 and
                    self.uncond_discriminator is not None):
                _, fake_feats = self.uncond_discriminator(gen_for_disc, return_features=True)
                with torch.no_grad():
                    _, real_feats = self.uncond_discriminator(her2_for_disc, return_features=True)
                loss_feat_match = feature_matching_loss(fake_feats, real_feats)
                loss_g = loss_g + self.hparams.feat_match_weight * loss_feat_match

            # Crop discriminator: random crops at full resolution
            if self.crop_discriminator is not None and self.hparams.crop_disc_weight > 0:
                fake_input_crop = torch.cat([generated, he], dim=1)
                cs = self.hparams.crop_size
                top = torch.randint(0, img_sz - cs, (1,)).item()
                left = torch.randint(0, img_sz - cs, (1,)).item()
                fake_crop = fake_input_crop[:, :, top:top+cs, left:left+cs]
                loss_crop_adv = hinge_loss_g(self.crop_discriminator(fake_crop))
                loss_g = loss_g + self.hparams.crop_disc_weight * loss_crop_adv

            # Unconditional discriminator: HER2-only adversarial
            if self.uncond_discriminator is not None and self.hparams.uncond_disc_weight > 0:
                loss_uncond_adv = hinge_loss_g(self.uncond_discriminator(gen_for_disc))
                loss_g = loss_g + self.hparams.uncond_disc_weight * loss_uncond_adv

        # Generator backward + step
        lr_scale = self._get_lr_scale()
        for pg in opt_g.param_groups:
            pg['lr'] = self.hparams.gen_lr * lr_scale

        opt_g.zero_grad()
        self.manual_backward(loss_g)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        opt_g.step()

        # Update EMA
        self._update_ema()

        # ----------------------------------------------------------------
        # Discriminator step
        # ----------------------------------------------------------------
        loss_d = torch.tensor(0.0, device=self.device)
        loss_crop_d = torch.tensor(0.0, device=self.device)
        loss_uncond_d = torch.tensor(0.0, device=self.device)
        if self.global_step >= self.hparams.adversarial_start_step and any_adv:
            with torch.no_grad():
                fake_detached = self.generator(he, uni_dropped, labels_dropped)

            # For 1024, downsample for disc
            if img_sz == 1024:
                fake_det_disc = F.interpolate(fake_detached, size=512, mode='bilinear', align_corners=False)
            else:
                fake_det_disc = fake_detached

            # Conditional discriminator
            if self.hparams.adversarial_weight > 0:
                real_input = torch.cat([her2_for_disc, he_for_disc], dim=1)
                fake_input = torch.cat([fake_det_disc, he_for_disc], dim=1)

                disc_real = self.discriminator(real_input)
                disc_fake = self.discriminator(fake_input)

                loss_d = sum(
                    hinge_loss_d(dr, df)
                    for dr, df in zip(disc_real, disc_fake)
                ) / len(disc_real)

            # Crop discriminator
            if self.crop_discriminator is not None and self.hparams.crop_disc_weight > 0:
                real_input_c = torch.cat([her2, he], dim=1)
                fake_input_c = torch.cat([fake_detached, he], dim=1)
                cs = self.hparams.crop_size
                top = torch.randint(0, img_sz - cs, (1,)).item()
                left = torch.randint(0, img_sz - cs, (1,)).item()
                real_crop = real_input_c[:, :, top:top+cs, left:left+cs]
                fake_crop = fake_input_c[:, :, top:top+cs, left:left+cs]
                loss_crop_d = hinge_loss_d(
                    self.crop_discriminator(real_crop),
                    self.crop_discriminator(fake_crop),
                )
                loss_d = loss_d + self.hparams.crop_disc_weight * loss_crop_d

            # Unconditional discriminator
            if self.uncond_discriminator is not None and (
                    self.hparams.uncond_disc_weight > 0 or self.hparams.feat_match_weight > 0):
                uncond_real_out = self.uncond_discriminator(her2_for_disc)
                uncond_fake_out = self.uncond_discriminator(fake_det_disc)
                loss_uncond_d = hinge_loss_d(uncond_real_out, uncond_fake_out)
                loss_d = loss_d + max(self.hparams.uncond_disc_weight, 1.0) * loss_uncond_d

            # R1 gradient penalty
            loss_r1 = torch.tensor(0.0, device=self.device)
            if self.global_step % self.hparams.r1_every == 0:
                with torch.amp.autocast('cuda', enabled=False):
                    if self.hparams.adversarial_weight > 0:
                        real_input_r1 = torch.cat([her2_for_disc, he_for_disc], dim=1).float().detach().requires_grad_(True)
                        for disc in [self.discriminator.disc_512]:
                            d_real = disc(real_input_r1)
                            grad_real = torch.autograd.grad(
                                outputs=d_real.sum(), inputs=real_input_r1,
                                create_graph=True,
                            )[0]
                            loss_r1 = loss_r1 + self.hparams.r1_weight * grad_real.pow(2).mean()
                    if self.uncond_discriminator is not None and (
                            self.hparams.uncond_disc_weight > 0 or self.hparams.feat_match_weight > 0):
                        her2_r1 = her2_for_disc.float().detach().requires_grad_(True)
                        d_real_uncond = self.uncond_discriminator(her2_r1)
                        grad_uncond = torch.autograd.grad(
                            outputs=d_real_uncond.sum(), inputs=her2_r1,
                            create_graph=True,
                        )[0]
                        loss_r1 = loss_r1 + self.hparams.r1_weight * grad_uncond.pow(2).mean()
                loss_d = loss_d + loss_r1
                self.log('train/r1_penalty', loss_r1, prog_bar=False)

            opt_d.zero_grad()
            self.manual_backward(loss_d)
            if self.hparams.adversarial_weight > 0:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
            if self.crop_discriminator is not None:
                torch.nn.utils.clip_grad_norm_(self.crop_discriminator.parameters(), 1.0)
            if self.uncond_discriminator is not None:
                torch.nn.utils.clip_grad_norm_(self.uncond_discriminator.parameters(), 1.0)
            opt_d.step()

        # Logging
        self.log('train/loss_g', loss_g, prog_bar=True)
        self.log('train/loss_d', loss_d, prog_bar=True)
        self.log('train/lpips', loss_lpips, prog_bar=True)
        self.log('train/adversarial', loss_adv, prog_bar=False)
        self.log('train/lr_scale', lr_scale, prog_bar=False)
        if self.crop_discriminator is not None:
            self.log('train/crop_adv_g', loss_crop_adv, prog_bar=False)
            self.log('train/crop_adv_d', loss_crop_d, prog_bar=False)
        if self.uncond_discriminator is not None:
            self.log('train/uncond_adv_g', loss_uncond_adv, prog_bar=False)
            self.log('train/uncond_adv_d', loss_uncond_d, prog_bar=False)
        if self.hparams.feat_match_weight > 0:
            self.log('train/feat_match', loss_feat_match, prog_bar=False)

    def on_validation_epoch_start(self):
        # Pick a random batch index for the second sample grid
        n_val_batches = max(1, len(self.trainer.val_dataloaders))
        self._random_val_batch_idx = torch.randint(1, max(2, n_val_batches), (1,)).item()
        # Per-label sample collectors (for multi-stain visual grids)
        self._val_per_label_samples = {}

    def _log_sample_grid(self, he, her2_01, gen_01, key):
        """Log H&E | Real | Gen grid to wandb."""
        n = min(4, len(he))
        he_01 = ((he[:n].cpu() + 1) / 2).clamp(0, 1)
        grid_images = []
        for i in range(n):
            grid_images.extend([
                he_01[i],
                her2_01[i].cpu(),
                gen_01[i].cpu(),
            ])
        grid = torchvision.utils.make_grid(grid_images, nrow=3, padding=2)
        if self.logger:
            self.logger.experiment.log({
                key: [wandb.Image(grid, caption='H&E | Real | Gen')],
                'global_step': self.global_step,
            })

    def validation_step(self, batch, batch_idx):
        he, her2, uni_or_crops, labels, fnames = batch

        # On-the-fly UNI extraction
        if self._uni_extract_on_the_fly:
            uni = self._extract_uni_from_sub_crops(uni_or_crops)
        else:
            uni = uni_or_crops

        if self.hparams.disable_uni:
            uni = torch.zeros_like(uni)

        if self.hparams.disable_class:
            labels = torch.full_like(labels, self.hparams.null_class)

        # Use EMA generator
        with torch.no_grad():
            generated = self.generator_ema(he, uni, labels)

        # LPIPS (4x downsample: 128 for 512, 256 for 1024)
        lpips_size = self.hparams.image_size // 4
        gen_lpips = F.interpolate(generated, size=lpips_size, mode='bilinear', align_corners=False)
        her2_lpips = F.interpolate(her2, size=lpips_size, mode='bilinear', align_corners=False)
        lpips_val = self.lpips_fn(gen_lpips, her2_lpips).mean()

        # SSIM
        gen_01 = ((generated + 1) / 2).clamp(0, 1)
        her2_01 = ((her2 + 1) / 2).clamp(0, 1)
        from torchmetrics.functional.image import structural_similarity_index_measure
        ssim_val = structural_similarity_index_measure(gen_01, her2_01, data_range=1.0)

        # DAB MAE (canonical: mean of top-10%)
        dab_gen = self.dab_extractor.extract_dab_intensity(generated.float().cpu(), normalize="none")
        dab_real = self.dab_extractor.extract_dab_intensity(her2.float().cpu(), normalize="none")

        def p90_score(dab):
            flat = dab.flatten()
            p90 = torch.quantile(flat, 0.9)
            mask = flat >= p90
            return flat[mask].mean().item() if mask.sum() > 0 else flat.mean().item()

        dab_mae = sum(
            abs(p90_score(dab_gen[i]) - p90_score(dab_real[i]))
            for i in range(len(dab_gen))
        ) / len(dab_gen)

        self.log('val/lpips', lpips_val, prog_bar=True, sync_dist=True)
        self.log('val/ssim', ssim_val, prog_bar=True, sync_dist=True)
        self.log('val/dab_mae', dab_mae, prog_bar=True, sync_dist=True)

        # Collect per-label samples for visual grids (multi-stain only)
        if hasattr(self, '_val_per_label_samples'):
            for i in range(len(labels)):
                lbl = labels[i].item()
                if lbl == self.hparams.null_class:
                    continue
                if lbl not in self._val_per_label_samples:
                    self._val_per_label_samples[lbl] = {'he': [], 'real': [], 'gen': []}
                bucket = self._val_per_label_samples[lbl]
                if len(bucket['he']) < 4:
                    bucket['he'].append(he[i].cpu())
                    bucket['real'].append(her2_01[i].cpu())
                    bucket['gen'].append(gen_01[i].cpu())

        # Log sample grids: first batch (fixed) + one random batch
        if batch_idx == 0:
            self._log_sample_grid(he, her2_01, gen_01, 'val/samples_fixed')
        elif batch_idx == self._random_val_batch_idx:
            self._log_sample_grid(he, her2_01, gen_01, 'val/samples_random')

    def on_validation_epoch_end(self):
        """Log per-label sample grids if multiple labels are present."""
        if not hasattr(self, '_val_per_label_samples') or len(self._val_per_label_samples) <= 1:
            return

        label_names = getattr(self.hparams, 'label_names', None)

        for lbl, bucket in sorted(self._val_per_label_samples.items()):
            if not bucket['he'] or not self.logger:
                continue
            name = label_names[lbl] if label_names and lbl < len(label_names) else str(lbl)
            self._log_sample_grid(
                torch.stack(bucket['he']),
                torch.stack(bucket['real']),
                torch.stack(bucket['gen']),
                f'val/samples_{name}',
            )

        self._val_per_label_samples = {}

    @torch.no_grad()
    def generate(self, he_images, uni_features, labels,
                 num_inference_steps=None, guidance_scale=1.0, seed=None):
        """Generate IHC images from H&E input.

        Args:
            he_images: [B, 3, H, H] where H=512 or H=1024
            uni_features: [B, N, 1024] where N=16 (4x4 CLS) or N=1024 (32x32 patch)
            labels: [B] class/stain labels
            num_inference_steps: ignored (single forward pass)
            guidance_scale: CFG scale (1.0 = no guidance)
            seed: random seed (for reproducibility, though model is deterministic)
        """
        if seed is not None:
            torch.manual_seed(seed)

        gen = self.generator_ema if hasattr(self, 'generator_ema') else self.generator

        if guidance_scale <= 1.0:
            return gen(he_images, uni_features, labels)

        # Classifier-free guidance
        null_labels = torch.full_like(labels, self.null_class)

        output_cond = gen(he_images, uni_features, labels)
        output_uncond = gen(he_images, uni_features, null_labels)

        output = output_uncond + guidance_scale * (output_cond - output_uncond)
        return output.clamp(-1, 1)
