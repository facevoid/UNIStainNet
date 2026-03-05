"""
Loss functions for UNIStainNet.

- VGGFeatureExtractor: intermediate VGG16 features for Gram-matrix style loss
- gram_matrix: compute Gram matrix of feature maps
- PatchNCELoss: contrastive loss between H&E input and generated output (alignment-free)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGFeatureExtractor(nn.Module):
    """Extract intermediate VGG16 features for Gram-matrix style loss.

    Uses early VGG layers (relu1_2, relu2_2, relu3_3) which capture texture
    at different scales. Gram matrices of these features are alignment-invariant
    texture descriptors — they measure feature co-occurrence statistics, not
    spatial layout (Gatys et al., 2016).
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        # Extract at relu1_2 (idx 4), relu2_2 (idx 9), relu3_3 (idx 16)
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # → relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # → relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16]) # → relu3_3
        # Freeze
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] in [-1, 1]
        Returns:
            list of feature maps at 3 scales
        """
        # Normalize: [-1,1] → [0,1] → ImageNet
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)
        return [f1, f2, f3]


def gram_matrix(feat):
    """Compute Gram matrix of feature map.

    Args:
        feat: [B, C, H, W]
    Returns:
        gram: [B, C, C] — normalized by spatial size
    """
    B, C, H, W = feat.shape
    feat_flat = feat.reshape(B, C, H * W)  # [B, C, N]
    gram = torch.bmm(feat_flat, feat_flat.transpose(1, 2))  # [B, C, C]
    return gram / (C * H * W)


class PatchNCELoss(nn.Module):
    """Patchwise Noise Contrastive Estimation loss.

    Compares H&E input and generated IHC through the generator's encoder.
    For each spatial position in the generated features, the corresponding
    position in the H&E features is the positive, and random other positions
    are negatives. Never sees GT IHC.

    Reference: Park et al., "Contrastive Learning for Unpaired Image-to-Image
    Translation" (ECCV 2020) — adapted for paired (misaligned) setting.
    """

    def __init__(self, layer_channels, num_patches=256, temperature=0.07):
        """
        Args:
            layer_channels: dict {layer_idx: channels} for each encoder layer
            num_patches: number of spatial positions to sample per layer
            temperature: InfoNCE temperature
        """
        super().__init__()
        self.num_patches = num_patches
        self.temperature = temperature

        # 2-layer MLP projection head per encoder layer
        self.mlps = nn.ModuleDict()
        for layer_idx, ch in layer_channels.items():
            self.mlps[str(layer_idx)] = nn.Sequential(
                nn.Linear(ch, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
            )

    def forward(self, feats_src, feats_tgt):
        """Compute PatchNCE loss across encoder layers.

        Args:
            feats_src: dict {layer_idx: [B, C, H, W]} from H&E input
            feats_tgt: dict {layer_idx: [B, C, H, W]} from generated IHC

        Returns:
            scalar loss
        """
        total_loss = 0.0
        n_layers = 0

        for layer_idx_str, mlp in self.mlps.items():
            layer_idx = int(layer_idx_str)
            feat_src = feats_src[layer_idx]  # [B, C, H, W]
            feat_tgt = feats_tgt[layer_idx]  # [B, C, H, W]

            B, C, H, W = feat_src.shape
            n_total = H * W

            # Reshape to [B, C, H*W] then [B, H*W, C]
            src_flat = feat_src.flatten(2).permute(0, 2, 1)  # [B, HW, C]
            tgt_flat = feat_tgt.flatten(2).permute(0, 2, 1)  # [B, HW, C]

            # Sample random spatial positions
            n_sample = min(self.num_patches, n_total)
            idx = torch.randperm(n_total, device=feat_src.device)[:n_sample]

            src_sampled = src_flat[:, idx, :]  # [B, n_sample, C]
            tgt_sampled = tgt_flat[:, idx, :]  # [B, n_sample, C]

            # Project through MLP
            src_proj = mlp(src_sampled)  # [B, n_sample, 256]
            tgt_proj = mlp(tgt_sampled)  # [B, n_sample, 256]

            # L2 normalize
            src_proj = F.normalize(src_proj, dim=-1)
            tgt_proj = F.normalize(tgt_proj, dim=-1)

            # InfoNCE: for each query (tgt), positive is matching src position
            # negatives are all other src positions
            # logits: [B, n_sample, n_sample] — (i,j) = similarity of tgt_i to src_j
            logits = torch.bmm(tgt_proj, src_proj.transpose(1, 2))  # [B, n, n]
            logits = logits / self.temperature

            # Target: diagonal (position i matches position i)
            target = torch.arange(n_sample, device=logits.device).unsqueeze(0).expand(B, -1)

            loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(0, 1))
            total_loss = total_loss + loss
            n_layers += 1

        return total_loss / n_layers if n_layers > 0 else total_loss
