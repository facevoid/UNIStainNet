"""
PatchGAN discriminator for HER2 image realism scoring.

Supports both unconditional (3ch HER2 only) and conditional (6ch H&E+HER2) modes.
Returns patch-level logits AND intermediate features for feature matching loss.

Architecture:
    C64(SN) -> C128(SN+IN) -> C256(SN+IN) -> C512(SN+IN,s1) -> 1ch(SN,s1)
    70x70 receptive field, output [B, 1, 30, 30] for 512x512 input.
    ~2.8M params (3ch) or ~2.8M params (6ch).

References:
  - Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (CVPR 2017)
  - Miyato et al., "Spectral Normalization for GANs" (ICLR 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator with spectral normalization.

    Returns both logits and intermediate features (for feature matching loss).

    Args:
        in_channels: 3 for unconditional (HER2 only), 6 for conditional (H&E + HER2)
        ndf: base number of discriminator filters
        n_layers: number of intermediate conv layers
    """

    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super().__init__()
        self.n_layers = n_layers

        # Build layers as a list (not sequential) so we can extract features
        self.layers = nn.ModuleList()

        # First layer: spectral norm, no instance norm
        self.layers.append(nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ))

        # Intermediate layers: spectral norm + instance norm
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.layers.append(nn.Sequential(
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=2, padding=1)),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ))

        # Penultimate layer: stride 1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.layers.append(nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=1, padding=1)),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ))

        # Final layer: 1-channel output, no activation (hinge loss uses raw logits)
        self.layers.append(nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * nf_mult, 1, 4, stride=1, padding=1)),
        ))

    def forward(self, x, return_features=False):
        """
        Args:
            x: [B, C, H, W] in [-1, 1]. C=3 (unconditional) or C=6 (conditional).
            return_features: if True, also return intermediate features for FM loss.

        Returns:
            logits: [B, 1, H', W'] patch-level real/fake logits
            features: list of intermediate feature maps (only if return_features=True)
        """
        features = []
        h = x
        for layer in self.layers:
            h = layer(h)
            if return_features:
                features.append(h)

        if return_features:
            return h, features
        return h


# ======================================================================
# Loss functions
# ======================================================================

def hinge_loss_d(d_real, d_fake):
    """Discriminator hinge loss."""
    return (torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean()) / 2


def hinge_loss_g(d_fake):
    """Generator hinge loss."""
    return -d_fake.mean()


def r1_gradient_penalty(discriminator, real_images, weight=10.0):
    """R1 gradient penalty (Mescheder et al., 2018).

    Regularizes discriminator to have small gradients on real data,
    which prevents the discriminator from becoming too confident and
    stabilizes GAN training.
    """
    real_images = real_images.detach().requires_grad_(True)
    d_real = discriminator(real_images)
    grad_real = autograd.grad(
        outputs=d_real.sum(),
        inputs=real_images,
        create_graph=True,
    )[0]
    penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return weight * penalty


def feature_matching_loss(d_feats_fake, d_feats_real):
    """Feature matching loss: L1 between discriminator features of fake vs real.

    Matches statistics at each discriminator layer. Alignment-free because
    it compares feature distributions, not pixel-level correspondence.
    """
    loss = 0.0
    for feat_fake, feat_real in zip(d_feats_fake, d_feats_real):
        loss += torch.nn.functional.l1_loss(feat_fake, feat_real.detach())
    return loss / len(d_feats_fake)


class MultiScaleDiscriminator(nn.Module):
    """Two PatchGAN discriminators at different scales."""

    def __init__(self, in_channels=6, ndf=64, n_layers=3):
        super().__init__()
        self.disc_512 = PatchDiscriminator(in_channels, ndf, n_layers)
        self.disc_256 = PatchDiscriminator(in_channels, ndf, n_layers)

    def forward(self, x, return_features=False):
        """
        Args:
            x: [B, 6, 512, 512] concat(output, H&E)

        Returns:
            list of (logits, [features]) from each scale
        """
        x_256 = F.interpolate(x, size=256, mode='bilinear', align_corners=False)

        if return_features:
            out_512, feats_512 = self.disc_512(x, return_features=True)
            out_256, feats_256 = self.disc_256(x_256, return_features=True)
            return [(out_512, feats_512), (out_256, feats_256)]
        else:
            out_512 = self.disc_512(x)
            out_256 = self.disc_256(x_256)
            return [out_512, out_256]
