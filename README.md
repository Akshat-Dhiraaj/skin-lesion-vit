# Skin Lesion Classification — Foundation Model Benchmark on HAM10000

A systematic benchmark of 7 vision foundation models for skin lesion classification, culminating in a **0.9808 AUC ensemble** on the HAM10000 dataset.

This project fine-tunes and evaluates models ranging from frozen CNN baselines to 2024–2025 self-supervised foundation models (DINOv2 with Registers, AIMv2, SigLIP), with Test Time Augmentation and ensemble combination.

---

## Results

### Full Leaderboard

| Rank | Phase | Model | Test AUC | Test Acc | Notes |
|------|-------|-------|----------|----------|-------|
| 1 | 9 | **Ensemble (EfficientNetV2 + DINOv2+Reg)** | **0.9808** | **88.4%** | Best weights: 0.60 / 0.40 |
| 2 | 8 | EfficientNetV2-M + TTA | 0.9766 | 86.9% | 8 augmented views |
| 3 | 8 | DINOv2+Registers + TTA | 0.9663 | 85.1% | 8 augmented views |
| 4 | 2 | EfficientNetV2-M | 0.9685 | 84.9% | Full fine-tune |
| 5 | 5 | DINOv2+Registers | 0.9653 | 83.9% | Linear probe — novel benchmark |
| 6 | 3 | ViT-Base | 0.9538 | 76.4% | Partial fine-tune (last 4 blocks) |
| 7 | 4 | DINOv2-Base | 0.9458 | 74.9% | Linear probe |
| 8 | 6 | AIMv2-Large | 0.9376 | 68.0% | Linear probe |
| 9 | 7 | SigLIP-SO400M | 0.9069 | 58.4% | Linear probe |
| 10 | 1 | ResNet50 (frozen) | 0.8940 | 60.4% | Head-only baseline |

### Ensemble Per-Class Report

```
              precision    recall  f1-score   support

         mel       0.66      0.72      0.69       167
          nv       0.95      0.93      0.94      1006
         bcc       0.87      0.90      0.88        77
       akiec       0.81      0.71      0.76        49
         bkl       0.74      0.79      0.76       165
          df       1.00      0.76      0.87        17
        vasc       0.91      0.95      0.93        22

    accuracy                           0.88      1503
   macro avg       0.85      0.83      0.83      1503
weighted avg       0.89      0.88      0.89      1503
```

---

## Dataset

**HAM10000** — Human Against Machine with 10,000 training images

- 10,015 dermatoscopic images across 7 skin lesion classes
- Severe class imbalance: 58x ratio (nv: 6705 vs df: 115)
- Source: [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

| Class | Full Name | Count |
|-------|-----------|-------|
| nv | Melanocytic Nevi | 6705 |
| mel | Melanoma | 1113 |
| bkl | Benign Keratosis | 1099 |
| bcc | Basal Cell Carcinoma | 514 |
| akiec | Actinic Keratoses | 327 |
| vasc | Vascular Lesions | 142 |
| df | Dermatofibroma | 115 |

---

## Methodology

### Class Imbalance Handling
- WeightedRandomSampler — rare classes sampled proportionally more during training

### Training Setup
- Mixed precision (AMP) — halves VRAM usage on RTX 4060 8GB
- Gradient accumulation — effective batch size 64 with physical batch 16
- Gradient checkpointing — additional VRAM savings for transformers
- Discriminative learning rates — 1e-3 head, 1e-5 backbone
- OneCycleLR scheduler with 10% warmup

### Augmentation Pipeline
HFlip, VFlip, RandomRotate90, ShiftScaleRotate, OpticalDistortion, ColorJitter, GaussNoise, GaussianBlur

### Test Time Augmentation (Phase 8)
8 views: original, H-flip, V-flip, rotate 90/180/270, HV-flip, zoom — averaged

### Ensemble (Phase 9)
Grid search over weights — best: EfficientNetV2 x 0.60 + DINOv2+Registers x 0.40

---

## Models Evaluated

| Model | ID | Type | Params |
|-------|----|------|--------|
| ResNet50 | timm/resnet50 | CNN | 23M |
| EfficientNetV2-M | timm/tf_efficientnetv2_m | CNN | 54M |
| ViT-Base | google/vit-base-patch16-224 | ViT | 86M |
| DINOv2-Base | facebook/dinov2-base | SSL ViT | 86M |
| DINOv2+Registers | facebook/dinov2-with-registers-base | SSL ViT | 86M |
| AIMv2-Large | timm/aimv2_large_patch14_224 | Autoregressive SSL | 307M |
| SigLIP-SO400M | google/siglip-so400m-patch14-224 | VLM Vision Tower | 400M |

---

## Key Findings

1. **Full fine-tuning beats frozen probes on small datasets** — EfficientNetV2 (full fine-tune, 0.9685) outperforms DINOv2-Base (frozen probe, 0.9458) despite DINOv2 being a stronger pretrained model.

2. **DINOv2 with Registers outperforms standard DINOv2** — 0.9653 vs 0.9458. First published HAM10000 benchmark for this model.

3. **TTA provides consistent free gains** — EfficientNetV2 improved +0.008 AUC with no retraining.

4. **CNN + ViT ensemble wins** — local features + global context at 0.60/0.40 achieves 0.9808 AUC.

5. **Melanoma recall is the critical gap** — 72% recall on mel is the primary area for future improvement.

---

## Setup

```bash
conda create -n skin-lesion python=3.11 -y
conda activate skin-lesion
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/raw --unzip
```

## Run

```bash
# Single phase
python src/engine.py --phase 1        # ResNet50 frozen
python src/engine.py --phase 2        # EfficientNetV2-M
python src/engine.py --phase 3        # ViT-Base
python src/engine.py --phase 4        # DINOv2-Base
python src/engine.py --phase 5        # DINOv2+Registers
python src/engine.py --phase 6        # AIMv2-Large
python src/engine.py --phase 7        # SigLIP
python src/tta_ensemble.py --phase 8  # TTA
python src/tta_ensemble.py --phase 9  # Ensemble

# All phases
.\run_all.bat

# MLflow dashboard
mlflow ui
```

---

## Hardware

- GPU: NVIDIA RTX 4060 Laptop (8GB VRAM)
- RAM: 32GB
- All experiments reproducible on consumer hardware

---

## Future Work

- Full fine-tuning of DINOv2+Registers (expected to exceed EfficientNetV2)
- Focal loss for melanoma recall improvement
- Magnitude pruning (Lottery Ticket Hypothesis) on best model
- GradCAM visualizations for clinical interpretability
- FastAPI inference endpoint

---

## References

- Dosovitskiy et al. (2020). An Image is Worth 16x16 Words
- Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision
- Darcet et al. (2024). Vision Transformers Need Registers
- El-Nouby et al. (2024). AIMv2: Scalable Autoregressive Image Models
- Tschandl et al. (2018). The HAM10000 dataset
- Tan & Le (2021). EfficientNetV2: Smaller Models and Faster Training
