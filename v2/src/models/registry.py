"""
src/models/registry.py
----------------------
Backbone registry for all models evaluated in this project.

Design
------
Every backbone is a ModelConfig entry. build_model() dispatches on
model_type and returns an nn.Module whose forward() signature is
uniformly  model(x) -> logits  where x is (B, C, H, W).
No per-backbone dispatch needed in the training loop.

Adding a new backbone: add a ModelConfig to REGISTRY, add a handler in
build_model() only if the architecture pattern is genuinely new.

model_type dispatch
-------------------
  timm     ConvNeXt-V2, EfficientNetV2, ResNet — standard timm interface
  dinov2   DINOv2-family: HF AutoModel + CLS token + linear head
  pandem   PanDerm: HF AutoModel + CLS token + linear head (with fallback)
  siglip2  SigLIP 2: extract vision_model from full VLM + linear head
"""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
import torch.nn as nn


NUM_CLASSES = 7


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    name: str
    model_type: str        # dispatch key — see module docstring
    model_id: str          # timm model name or HuggingFace repo ID
    freeze_backbone: bool  # True = linear probe, False = full fine-tune
    num_classes: int = NUM_CLASSES
    input_size: int = 224
    # Training hyperparams bundled here so one dict drives both model + trainer
    batch_size: int = 16
    lr_head: float = 1e-3
    lr_backbone: float = 1e-5
    epochs: int = 20
    accum_steps: int = 4


# ── Registry ──────────────────────────────────────────────────────────────────

REGISTRY: dict[str, ModelConfig] = {

    # ── ConvNeXt-V2 Large ─────────────────────────────────────────────────────
    # Strongest CNN baseline. 384px full fine-tune.
    # Matches your old Phase 18 ConvNeXt-V2 config.
    "convnextv2_large": ModelConfig(
        name="convnextv2_large",
        model_type="timm",
        model_id="convnextv2_large.fcmae_ft_in22k_in1k_384",
        freeze_backbone=False,
        input_size=384,
        batch_size=8,
        lr_head=1e-3,
        lr_backbone=5e-6,
        epochs=25,
        accum_steps=8,
    ),

    # ── DINOv2+Registers Base ─────────────────────────────────────────────────
    # Linear probe — fast cross-dataset baseline for Direction B.
    "dinov2_registers_probe": ModelConfig(
        name="dinov2_registers_probe",
        model_type="dinov2",
        model_id="facebook/dinov2-with-registers-base",
        freeze_backbone=True,
        input_size=224,
        batch_size=64,
        lr_head=1e-3,
        lr_backbone=0.0,
        epochs=20,
        accum_steps=1,
    ),
    # Full fine-tune — replicates your old Phase 10, now with lesion splits.
    "dinov2_registers_ft": ModelConfig(
        name="dinov2_registers_ft",
        model_type="dinov2",
        model_id="facebook/dinov2-with-registers-base",
        freeze_backbone=False,
        input_size=224,
        batch_size=8,
        lr_head=1e-3,
        lr_backbone=5e-6,
        epochs=25,
        accum_steps=8,
    ),

    # ── PanDerm / DermLIP ─────────────────────────────────────────────────────
    # DermLIP_PanDerm (July 2025) — improved PanDerm with PubMed knowledge.
    # Publicly available on HuggingFace. Uses 256x256 input.
    # FIXED: Was "redlessone/PanDerm_base" (doesn't exist) → now "redlessone/DermLIP_PanDerm-base-w-PubMed-256"
    # GitHub: https://github.com/SiyuanYan1/PanDerm
    "pandem_probe": ModelConfig(
        name="pandem_probe",
        model_type="pandem",
        model_id="redlessone/DermLIP_PanDerm-base-w-PubMed-256",
        freeze_backbone=True,
        input_size=256,  # FIXED: was 224 → DermLIP uses 256x256
        batch_size=64,
        lr_head=1e-3,
        lr_backbone=0.0,
        epochs=20,
        accum_steps=1,
    ),
    "pandem_ft": ModelConfig(
        name="pandem_ft",
        model_type="pandem",
        model_id="redlessone/DermLIP_PanDerm-base-w-PubMed-256",
        freeze_backbone=False,
        input_size=256,  # FIXED: was 224 → DermLIP uses 256x256
        batch_size=8,
        lr_head=1e-3,
        lr_backbone=5e-6,
        epochs=25,
        accum_steps=8,
    ),

    # ── PanDerm 224 variants ──────────────────────────────────────────────────
    # A/B test: the HF model card for DermLIP_PanDerm-base-w-PubMed-256 says
    # 224x224; the "-256" in the name is the PubmedBert context length.
    # These entries let us compare 224 vs 256 without touching existing checkpoints.
    "pandem_probe_224": ModelConfig(
        name="pandem_probe_224",
        model_type="pandem",
        model_id="redlessone/DermLIP_PanDerm-base-w-PubMed-256",
        freeze_backbone=True,
        input_size=224,
        batch_size=64,
        lr_head=1e-3,
        lr_backbone=0.0,
        epochs=20,
        accum_steps=1,
    ),
    "pandem_ft_224": ModelConfig(
        name="pandem_ft_224",
        model_type="pandem",
        model_id="redlessone/DermLIP_PanDerm-base-w-PubMed-256",
        freeze_backbone=False,
        input_size=224,
        batch_size=8,
        lr_head=1e-3,
        lr_backbone=5e-6,
        epochs=25,
        accum_steps=8,
    ),

    # ── PanDerm REAL (open_clip load path) ────────────────────────────────────
    # The HF repo is an open_clip checkpoint; AutoModel can't load it. These
    # entries use RealPanDermClassifier which builds the custom BEiT vision
    # tower and loads weights directly. Resolution 224 (HF model card) +
    # CLIP normalization stats (0.4815, 0.4578, 0.4082) — see transforms.py.
    "pandem_real_probe": ModelConfig(
        name="pandem_real_probe",
        model_type="pandem_real",
        model_id="redlessone/DermLIP_PanDerm-base-w-PubMed-256",
        freeze_backbone=True,
        input_size=224,
        batch_size=64,
        lr_head=1e-3,
        lr_backbone=0.0,
        epochs=20,
        accum_steps=1,
    ),
    "pandem_real_ft": ModelConfig(
        name="pandem_real_ft",
        model_type="pandem_real",
        model_id="redlessone/DermLIP_PanDerm-base-w-PubMed-256",
        freeze_backbone=False,
        input_size=224,
        batch_size=8,
        lr_head=1e-3,
        lr_backbone=5e-6,
        epochs=25,
        accum_steps=8,
    ),

    # ── SigLIP 2 SO400M ───────────────────────────────────────────────────────
    # Google VLM vision tower (Feb 2025). Linear probe only — 400M params.
    # Verify current HF ID at: https://huggingface.co/google/siglip2-so400m-patch14-384
    "siglip2_probe": ModelConfig(
        name="siglip2_probe",
        model_type="siglip2",
        model_id="google/siglip2-so400m-patch14-384",
        freeze_backbone=True,
        input_size=384,
        batch_size=8,
        lr_head=1e-3,
        lr_backbone=0.0,
        epochs=20,
        accum_steps=8,
    ),

    # ── EfficientNetV2-M ──────────────────────────────────────────────────────
    # Best CNN from v1 (0.9685 AUC image-level). Carried forward as the
    # cross-architecture ensemble partner for ConvNeXt-V2.
    "efficientnetv2_m": ModelConfig(
        name="efficientnetv2_m",
        model_type="timm",
        model_id="tf_efficientnetv2_m",
        freeze_backbone=False,
        input_size=384,
        batch_size=8,
        lr_head=1e-3,
        lr_backbone=1e-5,
        epochs=20,
        accum_steps=4,
    ),

}


# ── Backbone nn.Module classes ────────────────────────────────────────────────

class DINOv2Classifier(nn.Module):
    """
    DINOv2-family: HuggingFace AutoModel + CLS token + linear head.
    Covers dinov2-base, dinov2-with-registers-base, and any future
    DINOv2-style model that returns last_hidden_state.
    """

    def __init__(self, model_id: str, num_classes: int, freeze_backbone: bool) -> None:
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_id)
        self.head = nn.Linear(self.backbone.config.hidden_size, num_classes)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls = self.backbone(pixel_values=x).last_hidden_state[:, 0]
        return self.head(cls)


class PanDermClassifier(nn.Module):
    """
    PanDerm / DermLIP: ViT-B pretrained on 2M dermatology images.

    Uses DermLIP_PanDerm (July 2025) — publicly available on HF.
    Handles both last_hidden_state (standard ViT) and pooler_output layouts
    for forward() resilience across model variants.

    Falls back to DINOv2+Registers if HF access fails.
    """

    def __init__(self, model_id: str, num_classes: int, freeze_backbone: bool) -> None:
        super().__init__()
        from transformers import AutoModel

        try:
            # Try loading the specified model (DermLIP_PanDerm by default)
            self.backbone = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            print(f"[OK] Loaded {model_id}")
        except Exception as e:
            # Fallback to DINOv2+Registers if the model isn't available
            print(f"[WARN] Failed to load {model_id}: {e}")
            print("  Falling back to DINOv2+Registers...")
            self.backbone = AutoModel.from_pretrained("facebook/dinov2-with-registers-base")

        self.head = nn.Linear(self.backbone.config.hidden_size, num_classes)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        # Prefer CLS token; fall back to pooler_output for variants that
        # only expose that (e.g. some PanDerm checkpoint configs).
        if hasattr(out, "last_hidden_state"):
            feat = out.last_hidden_state[:, 0]
        elif hasattr(out, "pooler_output"):
            feat = out.pooler_output
        else:
            raise RuntimeError(
                f"PanDerm output has neither last_hidden_state nor pooler_output. "
                f"Keys: {list(out.keys())}"
            )
        return self.head(feat)


class RealPanDermClassifier(nn.Module):
    """
    Real PanDerm / DermLIP vision encoder, loaded directly from open_clip weights.

    The HF repo `redlessone/DermLIP_PanDerm-base-w-PubMed-256` is an open_clip
    checkpoint, not a transformers model — so `AutoModel.from_pretrained` fails
    with "missing model_type key in config.json". The vision tower is a custom
    BEiT-Base/16: absolute pos embed, no relative pos bias, LayerScale (init_values=0.1),
    BEiT-style attention (separate q_bias / v_bias with shared qkv). CLIP normalization
    stats (NOT ImageNet, NOT SigLIP).

    Input: 224x224. Output: 768-dim CLS-token features → linear head.
    """

    def __init__(self, model_id: str, num_classes: int, freeze_backbone: bool) -> None:
        super().__init__()
        from huggingface_hub import hf_hub_download
        from timm.models.beit import Beit

        weights_path = hf_hub_download(model_id, "open_clip_pytorch_model.bin")
        full_sd = torch.load(weights_path, map_location="cpu", weights_only=False)
        visual_sd = {k.replace("visual.", "", 1): v for k, v in full_sd.items() if k.startswith("visual.")}
        visual_sd = {k: v for k, v in visual_sd.items() if not k.startswith("head.")}

        self.backbone = Beit(
            img_size=224, patch_size=16, in_chans=3,
            num_classes=0, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True,
            use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
            init_values=0.1,
            global_pool="token",
        )
        missing, unexpected = self.backbone.load_state_dict(visual_sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"RealPanDermClassifier load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        print(f"[OK] Loaded real PanDerm vision encoder from {model_id}")

        self.head = nn.Linear(768, num_classes)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)


class SigLIP2Classifier(nn.Module):
    """
    SigLIP 2: extract the vision_model from the full VLM + linear head.
    Always frozen — 400M params, full FT is VRAM-prohibitive at 8GB.
    """

    def __init__(self, model_id: str, num_classes: int) -> None:
        super().__init__()
        from transformers import AutoModel
        full_model = AutoModel.from_pretrained(model_id)
        self.vision = full_model.vision_model
        self.head = nn.Linear(self.vision.config.hidden_size, num_classes)
        for p in self.vision.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SigLIP uses mean-pooled patch tokens exposed via pooler_output
        feat = self.vision(pixel_values=x).pooler_output
        return self.head(feat)


# ── Public API ────────────────────────────────────────────────────────────────

def build_model(config: ModelConfig | str) -> nn.Module:
    """
    Instantiate and return a model from the registry.

    Args:
        config: A ModelConfig or a key into REGISTRY.

    Returns:
        nn.Module with forward(x) -> logits signature.
        Prints trainable / total param count.
    """
    if isinstance(config, str):
        if config not in REGISTRY:
            raise KeyError(
                f"Unknown model key: {config!r}. "
                f"Available: {list(REGISTRY.keys())}"
            )
        config = REGISTRY[config]

    mtype, mid, nc = config.model_type, config.model_id, config.num_classes

    if mtype == "timm":
        model = timm.create_model(mid, pretrained=True, num_classes=nc)
        if config.freeze_backbone:
            head_kw = ("head", "classifier", "fc")
            for name, p in model.named_parameters():
                if not any(k in name for k in head_kw):
                    p.requires_grad = False

    elif mtype == "dinov2":
        model = DINOv2Classifier(mid, nc, config.freeze_backbone)

    elif mtype == "pandem":
        model = PanDermClassifier(mid, nc, config.freeze_backbone)

    elif mtype == "pandem_real":
        model = RealPanDermClassifier(mid, nc, config.freeze_backbone)

    elif mtype == "siglip2":
        model = SigLIP2Classifier(mid, nc)

    else:
        raise ValueError(
            f"Unknown model_type: {mtype!r}. "
            f"Expected one of: timm, dinov2, pandem, pandem_real, siglip2"
        )

    trainable, total = count_params(model)
    print(
        f"[{config.name}] {trainable:,} / {total:,} trainable params "
        f"({100 * trainable / total:.1f}%)"
    )
    return model


def build_optimizer(model: nn.Module, cfg: ModelConfig) -> torch.optim.Optimizer:
    """
    AdamW with discriminative learning rates:
      - Head params (randomly initialised) → cfg.lr_head
      - Backbone params (pretrained)        → cfg.lr_backbone
    Frozen params (requires_grad=False) are excluded automatically.
    """
    head_kw = ("head", "classifier", "fc")
    head_params, backbone_params = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in head_kw):
            head_params.append(p)
        else:
            backbone_params.append(p)

    param_groups = []
    if backbone_params and cfg.lr_backbone > 0:
        param_groups.append({"params": backbone_params, "lr": cfg.lr_backbone})
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.lr_head})

    if not param_groups:
        raise ValueError("No trainable parameters found. Check freeze settings.")

    return torch.optim.AdamW(param_groups, weight_decay=1e-4)


def count_params(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_count, total_count)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total