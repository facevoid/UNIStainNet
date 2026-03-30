#!/usr/bin/env python3
"""
UNIStainNet Interactive Demo — Hugging Face Spaces

Virtual staining of H&E histopathology images to IHC (HER2, Ki67, ER, PR).
Supports ZeroGPU (HF Pro) for live inference, falls back to gallery-only on CPU.
"""

import json
import os
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from huggingface_hub import hf_hub_download

# ── ZeroGPU support ──────────────────────────────────────────────────
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    spaces = None
    HAS_SPACES = False

GPU_AVAILABLE = torch.cuda.is_available()

def _gpu_decorator(duration=60):
    if HAS_SPACES and hasattr(spaces, "GPU"):
        return spaces.GPU(duration=duration)
    return lambda fn: fn

# ── Constants ────────────────────────────────────────────────────────
STAIN_NAMES = ["HER2", "Ki67", "ER", "PR"]
GALLERY_DIR = Path(__file__).parent / "gallery"
TARGET_SIZE = 512
MODEL_REPO = os.environ.get("MODEL_REPO", "faceless-void/UNIStainNet")
CHECKPOINT_FILENAME = "mist_multistain_last.ckpt"

NO_GPU_MSG = (
    "GPU is not available on this Space (requires HF Pro for ZeroGPU). "
    "Please use the **Gallery** tab to browse pre-computed results, "
    "or run the app locally with a GPU: `python app.py`"
)


# ── Lazy imports (avoid crash if no GPU) ─────────────────────────────
_model_cache = {"model": None, "uni_model": None, "spatial_pool_size": 32}


def _get_checkpoint_path():
    local_path = Path(__file__).parent / "checkpoints" / CHECKPOINT_FILENAME
    if local_path.exists():
        return str(local_path)
    return hf_hub_download(repo_id=MODEL_REPO, filename=CHECKPOINT_FILENAME)


def _load_models():
    """Load models onto GPU. Only called when GPU is confirmed available."""
    from src.models.trainer import UNIStainNetTrainer
    import timm

    if _model_cache["model"] is None:
        ckpt_path = _get_checkpoint_path()
        print(f"Loading UNIStainNet from {ckpt_path} ...")
        model = UNIStainNetTrainer.load_from_checkpoint(ckpt_path, strict=False)
        model = model.cuda().eval()
        _model_cache["model"] = model
        _model_cache["spatial_pool_size"] = getattr(model.hparams, "uni_spatial_size", 32)

        print("Loading UNI ViT-L/16 ...")
        uni_model = timm.create_model(
            "hf-hub:MahmoodLab/uni", pretrained=True,
            init_values=1e-5, dynamic_img_size=True,
        )
        uni_model = uni_model.cuda().eval()
        _model_cache["uni_model"] = uni_model
        print("  Models loaded")
    else:
        _model_cache["model"] = _model_cache["model"].cuda()
        _model_cache["uni_model"] = _model_cache["uni_model"].cuda()

    return _model_cache["model"], _model_cache["uni_model"], _model_cache["spatial_pool_size"]


# ── Preprocessing ────────────────────────────────────────────────────

def preprocess_he(pil_image, target_size=TARGET_SIZE):
    w, h = pil_image.size
    short = min(w, h)
    left = (w - short) // 2
    top = (h - short) // 2
    pil_image = pil_image.crop((left, top, left + short, top + short))
    if short != target_size:
        pil_image = pil_image.resize((target_size, target_size), Image.BICUBIC)
    return pil_image


def pil_to_tensor(pil_image):
    t = TF.to_tensor(pil_image)
    t = TF.normalize(t, [0.5] * 3, [0.5] * 3)
    return t.unsqueeze(0)


def tensor_to_pil(tensor):
    t = ((tensor[0].cpu() + 1) / 2).clamp(0, 1)
    return TF.to_pil_image(t)


def extract_uni_features(uni_model, he_tensor_01, spatial_pool_size=32):
    from src.data.mist_dataset import STAIN_TO_LABEL
    uni_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    B = he_tensor_01.shape[0]
    num_crops, patches_per_side = 4, 14
    crop_h = he_tensor_01.shape[2] // num_crops
    crop_w = he_tensor_01.shape[3] // num_crops

    sub_crops = []
    for i in range(num_crops):
        for j in range(num_crops):
            sub = he_tensor_01[:, :, i*crop_h:(i+1)*crop_h, j*crop_w:(j+1)*crop_w]
            sub = F.interpolate(sub, size=(224, 224), mode="bicubic", align_corners=False)
            sub = torch.stack([uni_transform(s) for s in sub])
            sub_crops.append(sub)

    all_crops = torch.stack(sub_crops, dim=1).reshape(B * 16, 3, 224, 224).cuda()
    with torch.no_grad():
        all_feats = uni_model.forward_features(all_crops)
        patch_tokens = all_feats[:, 1:, :]

    patch_tokens = patch_tokens.reshape(B, num_crops, num_crops, patches_per_side, patches_per_side, 1024)
    full_size = num_crops * patches_per_side
    full_grid = patch_tokens.permute(0, 1, 3, 2, 4, 5).reshape(B, full_size, full_size, 1024)

    S = spatial_pool_size
    if S < full_size:
        grid_bchw = full_grid.permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(grid_bchw, S)
        result = pooled.permute(0, 2, 3, 1)
    else:
        result = full_grid
    return result.reshape(B, S * S, 1024)


# ── Inference functions ──────────────────────────────────────────────

@_gpu_decorator(duration=60)
def _generate_single_gpu(image, stain, guidance_scale):
    """GPU inference for single stain."""
    from src.data.mist_dataset import STAIN_TO_LABEL
    model, uni_model, spatial_pool_size = _load_models()

    he_pil = preprocess_he(image)
    he_tensor = pil_to_tensor(he_pil).cuda()
    he_01 = ((he_tensor + 1) / 2).clamp(0, 1)
    uni_feats = extract_uni_features(uni_model, he_01, spatial_pool_size).cuda()
    labels = torch.tensor([STAIN_TO_LABEL[stain]], device="cuda", dtype=torch.long)

    with torch.no_grad():
        gen = model.generate(he_tensor, uni_feats, labels, guidance_scale=guidance_scale)
    return tensor_to_pil(gen)


@_gpu_decorator(duration=120)
def _generate_all_gpu(image, guidance_scale):
    """GPU inference for all 4 stains."""
    from src.data.mist_dataset import STAIN_TO_LABEL
    model, uni_model, spatial_pool_size = _load_models()

    he_pil = preprocess_he(image)
    he_tensor = pil_to_tensor(he_pil).cuda()
    he_01 = ((he_tensor + 1) / 2).clamp(0, 1)
    uni_feats = extract_uni_features(uni_model, he_01, spatial_pool_size).cuda()

    results = {}
    for stain in STAIN_NAMES:
        labels = torch.tensor([STAIN_TO_LABEL[stain]], device="cuda", dtype=torch.long)
        with torch.no_grad():
            gen = model.generate(he_tensor, uni_feats, labels, guidance_scale=guidance_scale)
        results[stain] = tensor_to_pil(gen)
    return he_pil, results


def generate_single_stain(image, stain, guidance_scale):
    """Wrapper with GPU availability check."""
    if image is None:
        return None, "Please upload an H&E image first."
    if not GPU_AVAILABLE and not HAS_SPACES:
        return None, NO_GPU_MSG
    try:
        t0 = time.time()
        result = _generate_single_gpu(image, stain, guidance_scale)
        return result, f"Generated in {time.time() - t0:.2f}s"
    except RuntimeError as e:
        if "NVIDIA" in str(e) or "CUDA" in str(e) or "cuda" in str(e):
            return None, NO_GPU_MSG
        raise


def generate_all_stains(image, guidance_scale):
    """Wrapper with GPU availability check."""
    if image is None:
        return None, None, None, None, None, "Please upload an H&E image first."
    if not GPU_AVAILABLE and not HAS_SPACES:
        return None, None, None, None, None, NO_GPU_MSG
    try:
        t0 = time.time()
        he_pil, results = _generate_all_gpu(image, guidance_scale)
        elapsed = f"Generated all 4 stains in {time.time() - t0:.2f}s"
        return he_pil, results["HER2"], results["Ki67"], results["ER"], results["PR"], elapsed
    except RuntimeError as e:
        if "NVIDIA" in str(e) or "CUDA" in str(e) or "cuda" in str(e):
            return None, None, None, None, None, NO_GPU_MSG
        raise


# ── Gallery ──────────────────────────────────────────────────────────

def load_gallery():
    meta_path = GALLERY_DIR / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def _make_gallery_label(key, entry):
    """Create a human-readable label for a gallery entry."""
    source = entry.get("source", "")
    parts = key.split("_")
    if source == "BCI":
        her2_class = parts[2] if len(parts) > 2 else ""
        sample_id = parts[3] if len(parts) > 3 else ""
        return f"BCI - HER2 {her2_class} (#{sample_id})"
    else:
        stain = parts[1] if len(parts) > 1 else ""
        sample_id = parts[2] if len(parts) > 2 else ""
        return f"MIST - {stain} ({sample_id})"


def show_gallery(display_name, gallery, name_map):
    """Show a gallery example by its display name."""
    key = name_map.get(display_name)
    if not key or not gallery or key not in gallery:
        return None, None, None, None, None, None, ""
    entry = gallery[key]
    base = GALLERY_DIR / "images"
    he = Image.open(base / entry["he"]).convert("RGB") if "he" in entry else None
    gt = Image.open(base / entry["gt"]).convert("RGB") if "gt" in entry else None
    gen_her2 = Image.open(base / entry["gen_her2"]).convert("RGB") if "gen_her2" in entry else None
    gen_ki67 = Image.open(base / entry["gen_ki67"]).convert("RGB") if "gen_ki67" in entry else None
    gen_er = Image.open(base / entry["gen_er"]).convert("RGB") if "gen_er" in entry else None
    gen_pr = Image.open(base / entry["gen_pr"]).convert("RGB") if "gen_pr" in entry else None

    gt_stain = entry.get("gt_stain", "Unknown")
    info = f"**Ground truth stain:** {gt_stain}"
    return he, gt, gen_her2, gen_ki67, gen_er, gen_pr, info


# ── Build Gradio App ─────────────────────────────────────────────────

gallery_data = load_gallery()
gallery_name_map = {}  # display_name -> key
gallery_display_names = []
if gallery_data:
    for key, entry in gallery_data.items():
        label = _make_gallery_label(key, entry)
        gallery_name_map[label] = key
        gallery_display_names.append(label)

with gr.Blocks(title="UNIStainNet -- Virtual IHC Staining") as demo:

    # ── Header ────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; padding:1.5rem 1rem 0.5rem 1rem;">
        <h1 style="font-size:1.8rem; font-weight:700; margin-bottom:0.3rem;">UNIStainNet</h1>
        <p style="font-size:1.05rem; color:#555; margin-top:0.2rem;">
            Virtual Immunohistochemistry Staining from H&amp;E
        </p>
    </div>
    <p style="text-align:center; color:#555; font-size:0.95rem; margin-bottom:0.8rem;">
        Generate HER2, Ki67, ER, and PR stains from a single H&amp;E breast tissue image
        using one unified deep learning model.
    </p>
    <div style="display:flex; justify-content:center; gap:0.6rem; flex-wrap:wrap; margin-bottom:1rem;">
        <span style="display:inline-block; padding:0.25rem 0.75rem; border-radius:999px;
                     font-size:0.8rem; font-weight:600; background:#dce3f9; color:#1a3a8a;">Breast Cancer Biomarkers</span>
        <span style="display:inline-block; padding:0.25rem 0.75rem; border-radius:999px;
                     font-size:0.8rem; font-weight:600; background:#d4edda; color:#155724;">HER2 / Ki67 / ER / PR</span>
        <span style="display:inline-block; padding:0.25rem 0.75rem; border-radius:999px;
                     font-size:0.8rem; font-weight:600; background:#e8d5f5; color:#5b1a8a;">One Model, 4 Stains</span>
    </div>
    """)

    # ── Tab 1: Gallery (default — works without GPU) ────────────
    with gr.Tab("Gallery", id="gallery"):
        if not gallery_display_names:
            gr.Markdown("No pre-computed gallery available.")
        else:
            gr.Markdown(
                "Browse pre-computed virtual staining results -- **no GPU required**. "
                "Each example shows the H&E input, ground truth IHC, and all 4 generated stains from our unified model."
            )
            gallery_dropdown = gr.Dropdown(
                choices=gallery_display_names,
                value=gallery_display_names[0] if gallery_display_names else None,
                label="Select Example",
            )
            gallery_info_box = gr.Markdown(value="")

            gr.Markdown("### Input & Ground Truth")
            with gr.Row():
                gal_he = gr.Image(type="pil", label="H&E Input", height=300)
                gal_gt = gr.Image(type="pil", label="Ground Truth IHC", height=300)

            gr.Markdown("### Generated IHC Stains (all from the same H&E)")
            with gr.Row():
                gal_her2 = gr.Image(type="pil", label="Generated HER2", height=280)
                gal_ki67 = gr.Image(type="pil", label="Generated Ki67", height=280)
            with gr.Row():
                gal_er = gr.Image(type="pil", label="Generated ER", height=280)
                gal_pr = gr.Image(type="pil", label="Generated PR", height=280)

            def _show_gallery_wrapper(display_name):
                return show_gallery(display_name, gallery_data, gallery_name_map)

            gallery_dropdown.change(
                fn=_show_gallery_wrapper,
                inputs=[gallery_dropdown],
                outputs=[gal_he, gal_gt, gal_her2, gal_ki67, gal_er, gal_pr, gallery_info_box],
            )

            # Auto-load first example
            demo.load(
                fn=lambda: _show_gallery_wrapper(gallery_display_names[0]) if gallery_display_names else (None,) * 7,
                outputs=[gal_he, gal_gt, gal_her2, gal_ki67, gal_er, gal_pr, gallery_info_box],
            )

    # ── Tab 2: Single Stain ──────────────────────────────────────
    with gr.Tab("Virtual Staining", id="inference"):
        if not GPU_AVAILABLE and not HAS_SPACES:
            gr.HTML(
                '<div style="background:#fff8e1; border:1px solid #ffe082; border-radius:8px; '
                'padding:0.75rem 1rem; margin-bottom:1rem; font-size:0.9rem; color:#6d4c00;">'
                f'{NO_GPU_MSG}</div>'
            )
        else:
            gr.Markdown(
                "Upload an H&E image and select a target IHC stain to generate."
            )
        with gr.Accordion("Image upload guidelines", open=False):
            gr.Markdown(
                "- **Tissue type:** H&E-stained breast cancer tissue\n"
                "- **Magnification:** 20x recommended (trained on BCI and MIST datasets)\n"
                "- **Size:** Images are center-cropped and resized to 512x512 internally\n"
                "- **Format:** PNG, JPEG, or TIFF\n"
                "- **Best results:** Regions with invasive carcinoma; "
                "adipose or stromal tissue may produce lower quality output"
            )
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Upload H&E Image", height=380)
                stain_choice = gr.Radio(
                    choices=STAIN_NAMES, value="HER2",
                    label="Target IHC Stain",
                    info="Select which immunohistochemistry marker to generate",
                )
                guidance_slider = gr.Slider(
                    minimum=1.0, maximum=3.0, step=0.1, value=1.0,
                    label="Guidance Scale",
                    info="1.0 = standard generation, higher = stronger stain signal (CFG)",
                )
                generate_btn = gr.Button("Generate", variant="primary", size="lg")
                gen_time = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=1):
                output_image = gr.Image(type="pil", label="Generated IHC", height=380)

        generate_btn.click(
            fn=generate_single_stain,
            inputs=[input_image, stain_choice, guidance_slider],
            outputs=[output_image, gen_time],
        )

    # ── Tab 3: Cross-Stain ───────────────────────────────────────
    with gr.Tab("Cross-Stain Comparison", id="cross-stain"):
        if not GPU_AVAILABLE and not HAS_SPACES:
            gr.HTML(
                '<div style="background:#fff8e1; border:1px solid #ffe082; border-radius:8px; '
                'padding:0.75rem 1rem; margin-bottom:1rem; font-size:0.9rem; color:#6d4c00;">'
                f'{NO_GPU_MSG}</div>'
            )
        else:
            gr.Markdown(
                "Generate **all 4 IHC stains** from a single H&E input."
            )
        with gr.Accordion("Image upload guidelines", open=False):
            gr.Markdown(
                "- **Tissue type:** H&E-stained breast cancer tissue\n"
                "- **Magnification:** 20x recommended (trained on BCI and MIST datasets)\n"
                "- **Size:** Images are center-cropped and resized to 512x512 internally\n"
                "- **Format:** PNG, JPEG, or TIFF\n"
                "- **Best results:** Regions with invasive carcinoma; "
                "adipose or stromal tissue may produce lower quality output"
            )
        with gr.Row():
            with gr.Column(scale=1):
                cross_input = gr.Image(type="pil", label="Upload H&E Image", height=300)
            with gr.Column(scale=1):
                cross_guidance = gr.Slider(
                    minimum=1.0, maximum=3.0, step=0.1, value=1.0,
                    label="Guidance Scale",
                    info="1.0 = standard generation, higher = stronger stain signal",
                )
                cross_btn = gr.Button("Generate All 4 Stains", variant="primary", size="lg")
                cross_time = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("### Results")
        with gr.Row():
            cross_he_out = gr.Image(type="pil", label="H&E Input", height=250)
            cross_her2 = gr.Image(type="pil", label="HER2", height=250)
            cross_ki67 = gr.Image(type="pil", label="Ki67", height=250)
            cross_er = gr.Image(type="pil", label="ER", height=250)
            cross_pr = gr.Image(type="pil", label="PR", height=250)

        cross_btn.click(
            fn=generate_all_stains,
            inputs=[cross_input, cross_guidance],
            outputs=[cross_he_out, cross_her2, cross_ki67, cross_er, cross_pr, cross_time],
        )

    # ── Tab 4: About ─────────────────────────────────────────────
    with gr.Tab("About", id="about"):
        gr.Markdown(
            """
## UNIStainNet: Foundation-Model-Guided Virtual Staining

UNIStainNet is a deep learning model for **virtual immunohistochemistry (IHC) staining**
from standard hematoxylin & eosin (H&E) histopathology images. It translates routine H&E
slides into IHC stains for four clinically important breast cancer biomarkers:
**HER2**, **Ki67**, **ER**, and **PR**.

### Why Virtual Staining?

- **Tissue conservation** -- eliminates the need for additional serial sections
- **Faster turnaround** -- results in seconds instead of hours/days
- **Cost reduction** -- one H&E slide replaces multiple IHC tests for screening
- **Consistency** -- no batch-to-batch staining variability

### How It Works

The model uses a **SPADE-UNet generator** conditioned on dense spatial features from a
frozen [UNI](https://github.com/mahmoodlab/UNI) pathology foundation model (ViT-L/16,
pretrained on 100M+ histopathology patches). A FiLM-based stain embedding allows a
**single unified model** to generate all 4 IHC stains.

| Component | Details |
|-----------|---------|
| **Generator** | SPADE-UNet with UNI spatial conditioning + FiLM stain embeddings |
| **Foundation Model** | UNI ViT-L/16 (frozen, 303M parameters) |
| **Spatial Tokens** | 4x4 sub-crop tiling of H&E input, yielding 32x32 = 1,024 tokens |
| **Generator Parameters** | 42M |
| **Inference** | Single forward pass (~1 second on GPU) |

### Quantitative Results (MIST Dataset, Unified Model)

| Stain | FID | KID x1k | Pearson-R | DAB KL |
|-------|-----|---------|-----------|--------|
| HER2  | 34.5 | 2.2   | 0.929     | 0.166  |
| Ki67  | 27.2 | 1.8   | 0.927     | 0.119  |
| ER    | 29.2 | 1.8   | 0.949     | 0.182  |
| PR    | 29.0 | 1.1   | 0.943     | 0.171  |

### Key Innovations

- **Dense UNI spatial conditioning**: Unlike prior methods that use global image features,
  UNIStainNet extracts spatially-resolved features at 32x32 resolution, enabling the generator
  to leverage fine-grained morphological context from the pathology foundation model.
- **Misalignment-aware training**: Because H&E and IHC are cut from consecutive tissue sections
  (not the same section), there are inherent spatial shifts. Our loss suite (perceptual loss,
  DAB intensity supervision, unconditional discriminator) is designed to handle this misalignment.
- **Classifier-free guidance (CFG)**: 10% class dropout and 10% UNI dropout during training
  enables tunable generation strength at inference time.

### Links

- **Paper**: [arXiv:2603.12716](https://arxiv.org/abs/2603.12716)
- **Code**: [github.com/facevoid/UNIStainNet](https://github.com/facevoid/UNIStainNet)
- **Project Page**: [facevoid.github.io/UNIStainNet](https://facevoid.github.io/UNIStainNet/)

### Disclaimer

This is a **research tool** for exploratory analysis. It is not intended for clinical diagnosis
and has not undergone regulatory validation. Generated stains should not be used for treatment decisions.
            """
        )

    # ── Footer ───────────────────────────────────────────────────
    gr.HTML("""
    <p style="text-align:center; padding:1rem; color:#999; font-size:0.8rem;">
        UNIStainNet |
        <a href="https://arxiv.org/abs/2603.12716" style="color:#888; text-decoration:none;">arXiv</a> |
        <a href="https://github.com/facevoid/UNIStainNet" style="color:#888; text-decoration:none;">GitHub</a> |
        <a href="https://facevoid.github.io/UNIStainNet/" style="color:#888; text-decoration:none;">Project Page</a>
    </p>
    """)

if __name__ == "__main__":
    demo.launch()
