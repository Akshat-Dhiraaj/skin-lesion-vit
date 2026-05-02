# Skin Lesion Classification — ViT Fine-tuning

Fine-tuning a Vision Transformer (ViT-Base) on the HAM10000 dataset for 7-class skin lesion classification.

## Goal
Beat published CNN baselines using a fine-tuned ViT, with GradCAM explainability and a FastAPI inference endpoint.

## Dataset
[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) — 10,015 dermatoscopic images across 7 classes.

| Class | Description |
|---|---|
| mel | Melanoma |
| nv | Melanocytic nevi |
| bcc | Basal cell carcinoma |
| akiec | Actinic keratoses |
| bkl | Benign keratosis |
| df | Dermatofibroma |
| vasc | Vascular lesions |

## Models

| Model | Val AUC | Params Trained |
|---|---|---|
| ResNet50 (frozen) | TBD | 2M |
| ResNet50 (full) | TBD | 25M |
| ViT-Base (partial) | TBD | 12M |
| ViT-Base + augmentation | TBD | 12M |

## Setup

```bash
conda create -n skin-lesion python=3.11
conda activate skin-lesion
pip install -r requirements.txt
```

## Project Structure

```
skin-lesion-vit/
├── data/
│   ├── raw/          ← HAM10000 images + metadata CSV
│   └── processed/    ← train/val/test splits
├── notebooks/        ← EDA and experiments
├── src/
│   ├── dataset.py    ← PyTorch Dataset
│   ├── transforms.py ← Augmentation pipeline
│   ├── model.py      ← ViT fine-tune setup
│   ├── train.py      ← Training loop
│   └── evaluate.py   ← Metrics + GradCAM
├── checkpoints/      ← Saved weights (gitignored)
└── logs/             ← Training logs (gitignored)
```

## Training

```bash
conda activate skin-lesion
python src/train.py
```

## Phases
- [x] Project scaffold
- [ ] Week 1-2: EDA + data pipeline
- [ ] Week 3-4: ResNet50 baseline
- [ ] Week 5-6: ViT fine-tuning
- [ ] Week 7-8: GradCAM + evaluation
- [ ] Week 9-10: FastAPI endpoint
