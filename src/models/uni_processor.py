"""
UNI feature processors: transform UNI pathology features into multi-scale spatial maps.

- UNIFeatureProcessor: for CLS-token features (4x4 = 16 tokens)
- UNIFeatureProcessorHighRes: for patch-token features (32x32 = 1024 tokens)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNIFeatureProcessor(nn.Module):
    """Process UNI features [B, 16, 1024] → multi-scale spatial feature maps.

    UNI produces 16 spatial tokens (4x4 grid) of 1024-dim. We project to
    generator channel dim and upsample to match each decoder layer resolution.
    """

    def __init__(self, uni_dim=1024, base_channels=512):
        super().__init__()
        self.base_channels = base_channels

        # Project UNI features to generator channel dim
        self.proj = nn.Sequential(
            nn.Linear(uni_dim, base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Multi-scale upsamplers: 4×4 → {16, 32, 64, 128, 256}
        # Each stage doubles spatial resolution
        ch = base_channels

        # 4→8→16
        self.up_16 = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 16→32
        self.up_32 = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 32→64
        ch_64 = base_channels // 2  # 256
        self.up_64 = nn.Sequential(
            nn.ConvTranspose2d(ch, ch_64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 64→128
        ch_128 = base_channels // 4  # 128
        self.up_128 = nn.Sequential(
            nn.ConvTranspose2d(ch_64, ch_128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 128→256
        ch_256 = base_channels // 8  # 64
        self.up_256 = nn.Sequential(
            nn.ConvTranspose2d(ch_128, ch_256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, uni_features):
        """
        Args:
            uni_features: [B, 16, 1024]

        Returns:
            dict of spatial feature maps at each resolution
        """
        B = uni_features.shape[0]

        # Project and reshape to spatial
        x = self.proj(uni_features)  # [B, 16, 512]
        x = x.permute(0, 2, 1).reshape(B, self.base_channels, 4, 4)  # [B, 512, 4, 4]

        # Multi-scale upsampling
        feat_16 = self.up_16(x)      # [B, 512, 16, 16]
        feat_32 = self.up_32(feat_16)   # [B, 512, 32, 32]
        feat_64 = self.up_64(feat_32)   # [B, 256, 64, 64]
        feat_128 = self.up_128(feat_64)  # [B, 128, 128, 128]
        feat_256 = self.up_256(feat_128)  # [B, 64, 256, 256]

        return {
            16: feat_16,
            32: feat_32,
            64: feat_64,
            128: feat_128,
            256: feat_256,
        }


class UNIFeatureProcessorHighRes(nn.Module):
    """Process high-res UNI features [B, 1024, 1024] → multi-scale spatial maps.

    With patch-token extraction, UNI produces 1024 tokens (32x32 spatial grid)
    of 1024-dim — 64x more spatial resolution than the CLS-only 4x4 grid.

    Since we START at 32x32, we process features with Conv2d (no hallucinated
    upsampling). Every spatial feature is backed by real UNI patch tokens.

    Architecture:
        32x32 input → conv process → feat_32 (512ch)
        32→64 upsample → conv → feat_64 (256ch)
        64→128 upsample → conv → feat_128 (128ch)
        128→256 upsample → conv → feat_256 (64ch)
        Also: 32→16 downsample → feat_16 (512ch, for bottleneck)
    """

    def __init__(self, uni_dim=1024, base_channels=512, spatial_size=32,
                 output_512=False):
        super().__init__()
        self.base_channels = base_channels
        self.spatial_size = spatial_size
        self.output_512 = output_512
        ch = base_channels

        # Project UNI 1024-dim → 512-dim per token
        self.proj = nn.Sequential(
            nn.Linear(uni_dim, ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Process at 32x32 (native resolution) — refine projected features
        self.proc_32 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 32→16 downsample (for bottleneck conditioning)
        self.down_16 = nn.Sequential(
            nn.Conv2d(ch, ch, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 32→64 upsample + refine
        ch_64 = ch // 2  # 256
        self.up_64 = nn.Sequential(
            nn.ConvTranspose2d(ch, ch_64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ch_64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch_64, ch_64, 3, padding=1),
            nn.InstanceNorm2d(ch_64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 64→128 upsample + refine
        ch_128 = ch // 4  # 128
        self.up_128 = nn.Sequential(
            nn.ConvTranspose2d(ch_64, ch_128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ch_128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch_128, ch_128, 3, padding=1),
            nn.InstanceNorm2d(ch_128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 128→256 upsample + refine
        ch_256 = ch // 8  # 64
        self.up_256 = nn.Sequential(
            nn.ConvTranspose2d(ch_128, ch_256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ch_256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch_256, ch_256, 3, padding=1),
            nn.InstanceNorm2d(ch_256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 256→512 upsample (for 1024 models with SPADE at dec1)
        if output_512:
            ch_512 = ch // 16  # 32
            self.up_512 = nn.Sequential(
                nn.ConvTranspose2d(ch_256, ch_512, 4, stride=2, padding=1),
                nn.InstanceNorm2d(ch_512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ch_512, ch_512, 3, padding=1),
                nn.InstanceNorm2d(ch_512),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, uni_features):
        """
        Args:
            uni_features: [B, S*S, 1024] where S = spatial_size (default 32)

        Returns:
            dict of spatial feature maps: {16, 32, 64, 128, 256}
        """
        B = uni_features.shape[0]
        S = self.spatial_size

        # Project and reshape to spatial grid
        x = self.proj(uni_features)  # [B, S*S, 512]
        x = x.permute(0, 2, 1).reshape(B, self.base_channels, S, S)  # [B, 512, 32, 32]

        # Process at native 32x32
        feat_32 = self.proc_32(x) + x  # residual connection

        # Downsample for bottleneck
        feat_16 = self.down_16(feat_32)  # [B, 512, 16, 16]

        # Upsample path — each level adds spatial detail from real UNI tokens
        feat_64 = self.up_64(feat_32)    # [B, 256, 64, 64]
        feat_128 = self.up_128(feat_64)  # [B, 128, 128, 128]
        feat_256 = self.up_256(feat_128) # [B, 64, 256, 256]

        out = {
            16: feat_16,
            32: feat_32,
            64: feat_64,
            128: feat_128,
            256: feat_256,
        }

        if self.output_512:
            feat_512 = self.up_512(feat_256)  # [B, 32, 512, 512]
            out[512] = feat_512

        return out
