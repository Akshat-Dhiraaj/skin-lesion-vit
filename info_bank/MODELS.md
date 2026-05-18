# Model & Technique Compendium — Skin Lesion Project

> Every model, training technique, and dataset used across `skin-lesion-vit` (v1) and `v2` (v2): **what it is, how it works, why we used it here, the result it produced, and the paper that backs it**. All result numbers trace to `experiments/*/results.txt` artifacts. Citations were web-verified (authors, venue, arXiv ID) during the project's skeptical-review pass.

Cross-references: pedagogical depth → [`Skin_Lesion_Project_Complete_Guide.md`](Skin_Lesion_Project_Complete_Guide.md); findings/publishability → [`FINDINGS.md`](FINDINGS.md); live numbers → [`RESULTS_REPORT.md`](RESULTS_REPORT.md); agent context → [`AI_context.md`](AI_context.md).

---

## Quick index

- [Part A — Backbone models](#part-a--backbone-models)
- [Part B — Training & inference techniques](#part-b--training--inference-techniques)
- [Part C — Datasets](#part-c--datasets)
- [Part D — Full reference list](#part-d--full-reference-list)

---

## Part A — Backbone models

### Result summary (HAM10000 lesion-level test set, v2)

| Model | Paradigm | Pretraining | v2 test AUC | Role in project |
|---|---|---|---:|---|
| ConvNeXt-V2 Large | CNN (modern) | FCMAE → ImageNet-21k→1k @ 384 | **0.9660** | Best single model; ensemble anchor |
| Real PanDerm (probe) | ViT/BEiT | 2M dermatology images (SSL) | 0.9527 | Best probe; best cross-dataset transfer |
| EfficientNetV2-M | CNN | ImageNet | 0.9474 | CNN diversity in ensemble |
| SigLIP 2 SO400M (probe) | ViT (VLM tower) | WebLI image-text (sigmoid loss) | 0.9460 | Large-scale VLM features |
| DINOv2 + Registers (FT) | ViT (SSL) | LVD-142M (self-distillation) | 0.9358 | Self-supervised ViT; fairest on Fitz17k |
| ResNet50 (v1 Phase 1) | CNN (classic) | ImageNet | (v1 baseline) | Linear-probe sanity floor |
| ViT-Base (v1 Phase 3) | ViT (supervised) | ImageNet-21k | (v1) | First transformer baseline |
| AIMv2-Large (v1 Phase 6) | ViT (autoregressive SSL) | DFN image-text | (v1) | Apple autoregressive SSL probe |
| SigLIP-SO400M (v1 Phase 7) | ViT (VLM tower) | WebLI | (v1) | v1 VLM tower experiment |

---

### A1. ResNet-50

- **What it is.** The canonical deep residual CNN. 50 layers, ~25M params.
- **How it works.** Stacks "residual blocks" — each block computes `F(x) + x` (a skip connection) so gradients flow directly through the identity path. This solved the degradation problem that prevented training very deep plain CNNs. Hierarchical conv features: early layers detect edges/textures, deep layers detect object parts.
- **Why we used it.** v1 Phase 1 — a *frozen linear-probe baseline*. If a frozen ImageNet ResNet-50 + a fresh linear head already gets decent AUC, that tells you how linearly separable the task is in generic-CNN feature space. It's the sanity floor every other model must beat.
- **Result.** v1 baseline (image-level splits). Established that dermoscopy is *not* trivially linearly separable from ImageNet features — motivating fine-tuning and domain-pretrained models.
- **Paper.** He, Zhang, Ren, Sun (2015). *Deep Residual Learning for Image Recognition.* [arXiv:1512.03385](https://arxiv.org/abs/1512.03385). CVPR 2016 (best paper).

### A2. EfficientNetV2-M

- **What it is.** A CNN family co-optimised for accuracy *and* training speed. The "M" (medium) variant, ~54M params, run at 384 px here.
- **How it works.** Two ideas. (1) *Fused-MBConv* blocks in early stages — replace the depthwise-conv bottleneck with a single regular conv, which is faster on accelerators despite more FLOPs. (2) *Progressive learning* — train at increasing resolution with increasing regularisation. NAS (neural architecture search) picked the block layout.
- **Why we used it.** v1 found it the best CNN. Carried into v2 as the *cross-architecture diversity partner* for ConvNeXt-V2 — a CNN whose error pattern differs enough from the ViTs to add ensemble value. It also has BatchNorm, which made it the model that benefited most from TENT on Fitzpatrick17k.
- **Result.** v2 HAM10000 test AUC **0.9474** (TTA 0.9555). With TENT (n_steps=3) reaches the best Fitz17k dark-skin Malignant AUC of any standalone model (**0.6742**).
- **Paper.** Tan & Le (2021). *EfficientNetV2: Smaller Models and Faster Training.* [arXiv:2104.00298](https://arxiv.org/abs/2104.00298). ICML 2021.

### A3. ViT-Base (vanilla Vision Transformer)

- **What it is.** The original "image is a sequence of 16×16 patches" transformer. ~86M params.
- **How it works.** Split the image into patches, linearly embed each, add a learned position embedding and a `[CLS]` token, then run standard transformer self-attention blocks. The `[CLS]` token's final representation is fed to a classification head. No convolution — all spatial reasoning is via attention.
- **Why we used it.** v1 Phase 3 — the first transformer baseline, to measure "does attention beat convolution on this task at equal pretraining (ImageNet-21k)?"
- **Result.** v1 phase result. Established the ViT baseline that DINOv2/SigLIP/PanDerm had to beat.
- **Paper.** Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* [arXiv:2010.11929](https://arxiv.org/abs/2010.11929). ICLR 2021.

### A4. DINOv2 + Registers

- **What it is.** A self-supervised ViT (DINOv2) with the "registers" fix. Base variant, ~86M params, 224 px.
- **How it works.** *DINOv2*: self-supervised learning by self-distillation — a student network is trained to match a teacher (an EMA of the student) across different augmented crops, with no labels. Produces features that transfer extremely well. *Registers*: Darcet et al. noticed DINOv2 attention maps had high-norm "artifact" tokens in low-information background patches that corrupted dense features; adding a few extra learnable `[reg]` tokens gives the model a scratchpad so the patch tokens stay clean.
- **Why we used it.** A strong label-free pretrained ViT — tests whether self-supervised features transfer to dermoscopy better than supervised ImageNet features. The Registers variant was chosen because skin lesions are small objects on a noisy background (hair, gel, ruler marks) — exactly the regime where the artifact-token problem bites.
- **Result.** v2 HAM10000 test AUC **0.9358** (TTA 0.9408). Notably it is the **fairest model on Fitzpatrick17k** (smallest light-to-dark gap, −0.027) — its self-supervised features generalise more evenly across skin tones than the others.
- **Papers.** Oquab et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.* [arXiv:2304.07193](https://arxiv.org/abs/2304.07193). · Darcet, Oquab, Mairal, Bojanowski (2023). *Vision Transformers Need Registers.* [arXiv:2309.16588](https://arxiv.org/abs/2309.16588). ICLR 2024 (Outstanding Paper).

### A5. ConvNeXt-V2 Large

- **What it is.** A pure CNN modernised to compete with transformers, pretrained with a masked-autoencoder objective. Large variant, ~198M params, 384 px. **The best single model in the project.**
- **How it works.** *ConvNeXt* took a ResNet and applied every transformer-era trick that's compatible with convolution (large depthwise kernels, LayerNorm instead of BatchNorm, GELU, inverted bottleneck, fewer activation functions). *V2* adds **FCMAE** (Fully Convolutional Masked AutoEncoder) self-supervised pretraining + a **Global Response Normalization** layer that improves inter-channel feature competition. The checkpoint we use was FCMAE-pretrained then fine-tuned on ImageNet-21k→1k at 384 px.
- **Why we used it.** v1's strongest backbone (Phase 18, 0.9880 image-level). Modern-CNN inductive bias (locality, translation equivariance) suits dermoscopy where the diagnostic signal is local texture/pigment patterns. It's the ensemble anchor — SLSQP assigns it ~78% of the weight.
- **Result.** v2 HAM10000 test AUC **0.9660** (TTA 0.9677) — the strongest single model on the honest splits. Dominates the 4-model SLSQP ensemble that reaches 0.9731 / post-hoc 0.9733.
- **Paper.** Woo et al. (2023). *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders.* [arXiv:2301.00808](https://arxiv.org/abs/2301.00808). CVPR 2023.

### A6. AIMv2-Large

- **What it is.** Apple's autoregressive image model, large variant. v1 Phase 6.
- **How it works.** Pretrains a ViT by *autoregressively predicting image patches* (like a language model predicts tokens) jointly with a multimodal (image+text) objective. The claim: autoregressive pretraining yields features that scale cleanly with model and data size.
- **Why we used it.** v1 experiment — does autoregressive SSL beat contrastive/self-distillation SSL for dermoscopy transfer?
- **Result.** v1 phase result (one of several backbone probes).
- **Paper.** Fini et al. (2024). *Multimodal Autoregressive Pre-training of Large Vision Encoders* (AIMv2). [arXiv:2411.14402](https://arxiv.org/abs/2411.14402). CVPR 2025 (Highlight). *(Note: AIMv1 — "Scalable Pre-training of Large Autoregressive Image Models", El-Nouby et al. 2024 — is a different, earlier paper. This project uses AIMv2.)*

### A7. SigLIP-SO400M and A8. SigLIP 2 SO400M

- **What they are.** The vision tower of Google's SigLIP vision-language models. "SO400M" = a shape-optimised ~400M-parameter ViT. v1 Phase 7 used SigLIP-1; v2 uses SigLIP 2.
- **How they work.** CLIP-style image–text contrastive pretraining, but with a **sigmoid loss** instead of softmax/InfoNCE — each image-text pair is an independent binary classification, which removes the need for a global negative-normalisation across the batch and trains stably at huge batch size. SigLIP 2 adds captioning, self-distillation, and dense-prediction objectives on top. We discard the text tower and linear-probe the vision tower.
- **Why we used them.** Tests whether *web-scale image-text* pretraining (very different from ImageNet or dermatology SSL) yields transferable dermoscopy features. The SO400M tower is one of the strongest publicly available frozen feature extractors.
- **Result.** v2 `siglip2_probe` HAM10000 test AUC **0.9460** (TTA 0.9490) — competitive as a *single* model. But its errors correlate too tightly with ConvNeXt-V2, so adding it to the 4-model ensemble *hurts* (0.9731 → 0.9703). A documented negative result. **Normalization caveat:** SigLIP expects mean=std=(0.5,0.5,0.5), not ImageNet stats — handled by `get_norm_stats("siglip2")`.
- **Papers.** Zhai, Mustafa, Kolesnikov, Beyer (2023). *Sigmoid Loss for Language Image Pre-Training (SigLIP).* [arXiv:2303.15343](https://arxiv.org/abs/2303.15343). ICCV 2023. · Tschannen et al. (2025). *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features.* [arXiv:2502.14786](https://arxiv.org/abs/2502.14786).

### A9. PanDerm (real) — the dermatology foundation model

- **What it is.** A vision encoder **pretrained on ~2 million dermatology images** via self-supervision. The single most important model in v2's cross-dataset story. Architecturally it is a **custom BEiT-Base/16** (see A10), loaded from an `open_clip` checkpoint.
- **How it works.** PanDerm's authors self-supervised a ViT on a 2M-image multi-institution dermatology corpus (dermoscopy + clinical photos + pathology). The released `DermLIP_PanDerm` checkpoint additionally CLIP-aligns the vision encoder with a PubMedBERT text encoder on clinically-captioned images (the DermLIP/Derm1M work). We use **only the vision tower** with a linear probe.
- **Why we used it.** The core hypothesis of v2: *does dermatology-domain pretraining beat generic (ImageNet/web) pretraining for skin lesion classification and — crucially — for cross-dataset transfer?* It directly tests the value of a domain foundation model.
- **Result.** **Yes, decisively, in linear-probe mode.**
  - HAM10000 test AUC **0.9527** (probe) — beats fully fine-tuned EfficientNetV2-M (0.9474) and DINOv2 (0.9358).
  - ISIC 2019 cross-dataset macro AUC **0.8845** — beats every fine-tuned ImageNet model (best was ConvNeXt-V2 at 0.8546, a **+0.030** gap). Domain pretraining transfers far better.
  - DDI absolute Malignant AUC **0.6821** — best of all models.
  - **4× sample efficiency**: the probe at 25% of HAM10000 training data ≈ EfficientNetV2-M at 100%.
  - **Counter-intuitive**: full fine-tuning *hurts* (`pandem_real_ft` 0.9295 < probe 0.9527). The pretraining is too task-relevant to disturb on only 7k images — see technique B7.
- **The bug story.** The HF repo is an `open_clip` checkpoint (no `config.json`); `transformers.AutoModel.from_pretrained` raised and a silent `try/except` substituted `facebook/dinov2-with-registers-base`. Every `pandem_*` checkpoint *without* `_real` is therefore DINOv2, not PanDerm. The fix (`RealPanDermClassifier`) builds the matching BEiT and loads the visual sub-state-dict with 0 missing / 0 unexpected keys. This is the project's headline engineering find.
- **Papers.** Yan et al. (2025). *A multimodal vision foundation model for clinical dermatology* (PanDerm). [Nature Medicine](https://www.nature.com/articles/s41591-025-03747-y). · Yan et al. (2025). *Derm1M: A Million-scale Vision-Language Dataset Aligned with Clinical Ontology Knowledge for Dermatology* (DermLIP). [arXiv:2503.14911](https://arxiv.org/abs/2503.14911). ICCV 2025 (Highlight). Two distinct papers, both Yan-first.

### A10. BEiT (the architecture inside PanDerm)

- **What it is.** Not a model we chose directly — it's the *architecture* PanDerm's vision encoder turned out to be, which we had to reverse-engineer to load the weights. BEiT-Base/16.
- **How it works.** BEiT pretrains a ViT with a **masked image modelling** objective (BERT-for-images): tokenise the image with a discrete VAE, mask patches, predict the missing visual tokens. Architecturally it differs from a vanilla ViT in three ways the loader must match: (1) **separated attention bias** — learned `q_bias` and `v_bias`, implicit zero `k_bias`, shared `qkv` weight; (2) **LayerScale** — per-channel learnable scalars `gamma_1`, `gamma_2` initialised near 0 so each block starts as a near-identity (stabilises deep-transformer training); (3) PanDerm's variant uses **absolute** position embeddings, not BEiT's default relative position bias.
- **Why it matters here.** Matching `timm.models.beit.Beit(use_abs_pos_emb=True, use_rel_pos_bias=False, init_values=0.1, global_pool='token')` is exactly what produced a clean weight load. Understanding the architecture was the difference between "PanDerm works" and "silent DINOv2 fallback".
- **Papers.** Bao, Dong, Piao, Wei (2022). *BEiT: BERT Pre-Training of Image Transformers.* [arXiv:2106.08254](https://arxiv.org/abs/2106.08254). ICLR 2022. · LayerScale: Touvron et al. (2021). *Going deeper with Image Transformers (CaiT).* [arXiv:2103.17239](https://arxiv.org/abs/2103.17239). ICCV 2021.

---

## Part B — Training & inference techniques

### B1. Patient/lesion-level splitting (`StratifiedGroupKFold`)

- **What & why.** HAM10000 has ~7,470 lesions across 10,015 images — most lesions photographed multiple times. Random *image-level* splitting leaks near-duplicate images of the same lesion across train/test, inflating the metric. We split by `lesion_id` so a lesion's images never span splits.
- **How it works.** Two-pass `sklearn.model_selection.StratifiedGroupKFold` — pass 1 separates test, pass 2 separates val — keeping class proportions roughly equal while keeping `lesion_id` groups intact. `assert_no_group_leakage()` enforces disjointness, asserted in pytest.
- **Improvement / honesty cost.** Same ConvNeXt-V2: v1 image-level **0.9880** → v2 lesion-level **0.9660**. The 0.022 gap *is* the leakage v1 was unknowingly reporting. This is the single most important methodological correction in the project.
- **Reference.** [`sklearn.StratifiedGroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html); pitfall context: Roberts et al. (2021), *Common pitfalls… ML for COVID-19*, [Nature Machine Intelligence](https://www.nature.com/articles/s42256-021-00307-0).

### B2. Discriminative learning rates + OneCycleLR

- **What & why.** A randomly-initialised head needs a high LR; pretrained backbone weights need a tiny LR or they're destroyed. We use head LR 1e-3, backbone LR ~5e-6 (≈100–200× ratio), scheduled by OneCycleLR (warmup → cosine decay).
- **How it works.** `build_optimizer` puts head and backbone params in separate AdamW groups with different LRs. OneCycleLR ramps LR up for the first 10% of steps then cosine-decays — the "super-convergence" recipe.
- **References.** Howard & Ruder (2018), *ULMFiT* (discriminative LRs), [arXiv:1801.06146](https://arxiv.org/abs/1801.06146). · Smith (2018), *Super-Convergence / OneCycle*, [arXiv:1708.07120](https://arxiv.org/abs/1708.07120).

### B3. 8 GB VRAM tricks — AMP, gradient accumulation, gradient checkpointing

- **What & why.** ConvNeXt-V2 Large at 384 px doesn't fit a useful batch in 8 GB. Three tricks make it fit.
- **How they work.** *AMP* (automatic mixed precision): compute in fp16/bf16, keep fp32 master weights — ~half the activation memory, with a `GradScaler` to prevent fp16 underflow. *Gradient accumulation*: physical batch 8, accumulate 8 steps → effective batch 64 without 64-image memory. *Gradient checkpointing*: don't store intermediate activations; recompute them in the backward pass — trades compute for memory.
- **References.** Micikevicius et al. (2017), *Mixed Precision Training*, [arXiv:1710.03740](https://arxiv.org/abs/1710.03740). · Chen et al. (2016), *Training Deep Nets with Sublinear Memory Cost* (gradient checkpointing), [arXiv:1604.06174](https://arxiv.org/abs/1604.06174).

### B4. Test-Time Augmentation (TTA)

- **What & why.** Average the model's softmax over 8 deterministic views of each test image (original, H/V flip, 90/180/270 rotation, HV-flip, slight zoom). Reduces prediction variance — dermoscopy has no canonical orientation, so all dihedral views are valid.
- **Improvement.** ~+0.01–0.02 macro AUC, free, no retraining. E.g. ConvNeXt-V2 0.9660 → 0.9677.
- **Reference.** Established practice since Krizhevsky et al. (2012, AlexNet 10-crop). Implementation: `src/tta.py`.

### B5. SLSQP ensembling (global & per-class & per-target)

- **What & why.** Averaging models equally is suboptimal when they differ in accuracy. SLSQP (Sequential Least-SQuares Programming) finds the weight vector maximising AUC under a sum-to-1, [0.01,0.99] simplex constraint. Per-class SLSQP fits an independent weight vector per class; per-target SLSQP refits for each external dataset.
- **How it works.** `scipy.optimize.minimize(method="SLSQP")` with *multi-restart* from uniform + each one-hot corner — necessary because AUC is non-smooth and a single uniform-init run gets stuck on the plateau.
- **Improvement.** 4-model SLSQP **0.9731** vs best single 0.9660. Per-class post-hoc **0.9733**. Per-target SLSQP exposes the accuracy–fairness frontier (B8).
- **References.** Kraft (1988), *A Software Package for Sequential Quadratic Programming.* · [`scipy.optimize` SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html). Code: `src/ensemble.py`, `src/cross_dataset_ensemble.py`.

### B6. Post-hoc calibration — Platt sigmoid + F2 thresholds

- **What & why.** Raw probabilities aren't calibrated and a 0.5 cutoff isn't clinically optimal. Three stages, fit on val: per-class SLSQP weights → per-class Platt sigmoid calibration → per-class F2-optimal thresholds.
- **How it works.** *Platt*: a 1-parameter logistic regression per class — strictly monotonic, so it **preserves AUC** while fixing the probability scale (rows deliberately not renormalised). *F2*: F-beta with β=2 weights recall 4× over precision — correct when missing a melanoma is far worse than a false alarm. **Why not isotonic:** on tiny rare-class val populations (`df=16`, `vasc=20`) isotonic collapses to <30 unique values and tanks macro OvR AUC by ~0.04.
- **Improvement.** Lifts mean ensemble 0.9713 → **0.9733** final.
- **References.** Platt (1999), *Probabilistic Outputs for SVMs.* · Niculescu-Mizil & Caruana (2005), *Predicting Good Probabilities with Supervised Learning* (Platt vs isotonic), [ICML](https://www.cs.cornell.edu/~caruana/niculescu.scldbst.crc.rev4.pdf). Code: `src/post_hoc.py`.

### B7. Linear probe vs full fine-tune (and why FT can hurt)

- **What & why.** A *probe* trains only a linear head on frozen features (tests feature quality); *full FT* unfreezes the backbone. For very strong, task-relevant pretrained features on small data, FT can **distort** the pretrained representation and underperform the probe.
- **Evidence here.** `pandem_real_probe` 0.9527 > `pandem_real_ft` 0.9295 on HAM10000, and 0.8845 > 0.8185 on ISIC 2019. PanDerm's 2M-image dermatology pretraining is too good to disturb with 7k images.
- **Reference.** Kumar et al. (2022), *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*, [arXiv:2202.10054](https://arxiv.org/abs/2202.10054). ICLR 2022.

### B8. TENT — test-time entropy minimisation

- **What & why.** A *training-free* domain-adaptation method: at inference, adapt only the normalisation-layer affines to each test batch by minimising prediction entropy. Used to close Fitzpatrick skin-tone disparities under domain shift (HAM10000 → DDI/Fitz17k).
- **How it works.** Freeze everything except BatchNorm/LayerNorm `weight`/`bias` (BN also drops running stats so each batch self-normalises). Per test batch: forward → mean softmax entropy → one Adam step → emit adapted prediction.
- **Improvement.** DDI dark-skin Malignant AUC: `pandem_real_probe` 0.5919 → **0.6931** (+0.10); TENT-aware SLSQP ensemble reaches **0.6571 with light-to-dark gap −0.0004 (full closure)**. Fitz17k fairness-tuned `__best` ensemble reaches **0.6802**. **Honest limitation:** TENT *hurts* `pandem_real_probe` on Fitz17k under every (lr, n_steps) tried — a recorded negative result.
- **Reference.** Wang et al. (2021), *TENT: Fully Test-Time Adaptation by Entropy Minimization*, [arXiv:2006.10726](https://arxiv.org/abs/2006.10726). ICLR 2021. Survey: Liang et al. (2023), [arXiv:2303.15361](https://arxiv.org/abs/2303.15361). Code: `src/tent.py`.

### B9. Binary focal loss (mel specialist)

- **What & why.** A dedicated melanoma-vs-rest head, trained with focal loss + a 50/50 balanced sampler, to maximise mel recall.
- **How it works.** Focal loss `−α_t (1−p_t)^γ log p_t` down-weights easy examples (γ) and class-balances (α). **The α trap we hit:** Lin et al.'s α=0.25 *down-weights* the positive class — correct for object detection (rare foreground vs 10⁴ background anchors) but *backwards* with a 50/50 balanced sampler. Switching to α=0.5 + longer schedule recovered ConvNeXt-V2 from a 0-recall collapse (test mel AUC 0.7671 → 0.8942).
- **Result (negative).** Even fixed, the binary specialist (0.8942 AUC, 0.22 recall@P=0.80) does **not** beat the 7-way model with an F2-tuned mel threshold (recall 0.566). v1 Phase 19's 0.83 recall was an artefact of leaky image-level splits.
- **Reference.** Lin, Goyal, Girshick, He, Dollár (2017), *Focal Loss for Dense Object Detection*, [arXiv:1708.02002](https://arxiv.org/abs/1708.02002). ICCV 2017. Code: `src/losses.py`, `scripts/train_mel_binary.py`.

### B10. Sample-efficiency curves

- **What & why.** Subsample training lesions to {10,25,50,100}% (stratified per class), retrain, plot test AUC vs fraction. A good pretrain saturates with less labelled data — directly relevant to clinical deployment where labels are scarce.
- **Result.** `pandem_real_probe`: 0.907 → 0.940 → 0.946 → 0.953 (flat — saturated by 25%). `efficientnetv2_m`: 0.864 → 0.898 → 0.922 → 0.950 (still climbing). ≈ 4× data efficiency from dermatology pretraining.
- **Reference.** Methodology after Kolesnikov et al. (2020), *Big Transfer (BiT)*, [arXiv:1912.11370](https://arxiv.org/abs/1912.11370). Code: `scripts/sample_efficiency.py`.

### B11. Grad-CAM (v1, interpretability)

- **What & why.** Visualises which image regions drove a prediction (gradient-weighted class activation map). Used in v1 for clinical-trust sanity checks (is the model looking at the lesion or at a ruler/hair artefact?).
- **Reference.** Selvaraju et al. (2017), *Grad-CAM*, [arXiv:1610.02391](https://arxiv.org/abs/1610.02391). ICCV 2017. Library: [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam).

---

## Part C — Datasets

### C1. HAM10000 (training + in-distribution test)

- **What.** 10,015 dermoscopic images, 7 classes (`mel, nv, bcc, akiec, bkl, df, vasc`), ~7,470 lesions, severe class imbalance (nv:df ≈ 58:1). Austrian + Australian sources.
- **Why.** The standard academic benchmark for multi-class dermoscopy; lets results be compared to the literature.
- **Reference.** Tschandl, Rosendahl, Kittler (2018), *The HAM10000 dataset*, [Scientific Data 5:180161](https://www.nature.com/articles/sdata2018161).

### C2. ISIC 2019 (cross-dataset, same modality)

- **What.** 15,316 dermoscopic images after removing the HAM10000 overlap; 8 classes (adds SCC, mapped to akiec). Includes the BCN20000 set (different hospital/camera than HAM10000).
- **Why.** Same modality, different acquisition — a clean "does it generalise to other dermoscopy?" check.
- **Reference.** Combalia et al. (2019), *BCN20000: Dermoscopic Lesions in the Wild*, [arXiv:1908.02288](https://arxiv.org/abs/1908.02288).

### C3. DDI — Diverse Dermatology Images (fairness)

- **What.** 656 **biopsy-proven** clinical photographs, balanced across Fitzpatrick skin-tone groups, from Stanford. Gold-standard labels.
- **Why.** The reference dataset for auditing dermatology-AI fairness across skin tones. Clinical photos (not dermoscopy) → also a hard modality-shift test.
- **Reference.** Daneshjou et al. (2022), *Disparities in dermatology AI performance on a diverse, curated clinical image set*, [Science Advances 8:eabq6147](https://www.science.org/doi/10.1126/sciadv.abq6147).

### C4. Fitzpatrick17k (fairness, larger/noisier)

- **What.** Up to 16,577 web-sourced clinical images with Fitzpatrick labels and 114 condition labels (3,730 usable here — ~23% URL liveness after link rot).
- **Why.** A larger, noisier Fitzpatrick-stratified fairness test; complements DDI's small-but-clean set.
- **Reference.** Groh et al. (2021), *Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset*, [CVPR Workshops 2021](https://openaccess.thecvf.com/content/CVPR2021W/ISIC/papers/Groh_Evaluating_Deep_Neural_Networks_Trained_on_Clinical_Images_in_Dermatology_CVPRW_2021_paper.pdf).

---

## Part D — Full reference list

**Backbone architectures**
- He et al. (2015). Deep Residual Learning (ResNet). [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Tan & Le (2021). EfficientNetV2. [arXiv:2104.00298](https://arxiv.org/abs/2104.00298)
- Dosovitskiy et al. (2020). ViT. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Oquab et al. (2023). DINOv2. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- Darcet et al. (2023). Vision Transformers Need Registers. [arXiv:2309.16588](https://arxiv.org/abs/2309.16588)
- Fini et al. (2024). AIMv2. [arXiv:2411.14402](https://arxiv.org/abs/2411.14402)
- Zhai et al. (2023). SigLIP. [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)
- Tschannen et al. (2025). SigLIP 2. [arXiv:2502.14786](https://arxiv.org/abs/2502.14786)
- Woo et al. (2023). ConvNeXt V2. [arXiv:2301.00808](https://arxiv.org/abs/2301.00808)
- Bao et al. (2022). BEiT. [arXiv:2106.08254](https://arxiv.org/abs/2106.08254)
- Touvron et al. (2021). CaiT / LayerScale. [arXiv:2103.17239](https://arxiv.org/abs/2103.17239)

**Dermatology foundation models**
- Yan et al. (2025). PanDerm. [Nature Medicine](https://www.nature.com/articles/s41591-025-03747-y)
- Yan et al. (2025). Derm1M / DermLIP. [arXiv:2503.14911](https://arxiv.org/abs/2503.14911)

**Techniques**
- Howard & Ruder (2018). ULMFiT (discriminative LR). [arXiv:1801.06146](https://arxiv.org/abs/1801.06146)
- Smith (2018). Super-Convergence / OneCycle. [arXiv:1708.07120](https://arxiv.org/abs/1708.07120)
- Micikevicius et al. (2017). Mixed Precision Training. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- Chen et al. (2016). Gradient Checkpointing. [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)
- Lin et al. (2017). Focal Loss. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- Platt (1999). Probabilistic Outputs for SVMs. *Advances in Large Margin Classifiers*, MIT Press.
- Niculescu-Mizil & Caruana (2005). Predicting Good Probabilities. [ICML PDF](https://www.cs.cornell.edu/~caruana/niculescu.scldbst.crc.rev4.pdf)
- Kraft (1988). Sequential Quadratic Programming (SLSQP).
- Wang et al. (2021). TENT. [arXiv:2006.10726](https://arxiv.org/abs/2006.10726)
- Liang et al. (2023). Test-Time Adaptation survey. [arXiv:2303.15361](https://arxiv.org/abs/2303.15361)
- Kumar et al. (2022). Fine-Tuning Distorts Pretrained Features. [arXiv:2202.10054](https://arxiv.org/abs/2202.10054)
- Kolesnikov et al. (2020). Big Transfer (BiT). [arXiv:1912.11370](https://arxiv.org/abs/1912.11370)
- Selvaraju et al. (2017). Grad-CAM. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
- Roberts et al. (2021). ML pitfalls (data leakage). [Nature Machine Intelligence](https://www.nature.com/articles/s42256-021-00307-0)

**Datasets**
- Tschandl et al. (2018). HAM10000. [Scientific Data](https://www.nature.com/articles/sdata2018161)
- Combalia et al. (2019). BCN20000 / ISIC 2019. [arXiv:1908.02288](https://arxiv.org/abs/1908.02288)
- Daneshjou et al. (2022). DDI. [Science Advances](https://www.science.org/doi/10.1126/sciadv.abq6147)
- Groh et al. (2021). Fitzpatrick17k. [CVPRW 2021](https://openaccess.thecvf.com/content/CVPR2021W/ISIC/papers/Groh_Evaluating_Deep_Neural_Networks_Trained_on_Clinical_Images_in_Dermatology_CVPRW_2021_paper.pdf)

**Tools**
- [PyTorch](https://pytorch.org/) · [HF Transformers](https://huggingface.co/docs/transformers) · [timm](https://github.com/huggingface/pytorch-image-models) · [OpenCLIP](https://github.com/mlfoundations/open_clip) · [Albumentations](https://albumentations.ai/) · [MLflow](https://mlflow.org/) · [scikit-learn](https://scikit-learn.org/) · [SciPy](https://scipy.org/) · [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

---

*Compiled 2026-05-16. Result numbers verified against `experiments/*/results.txt`. Citation metadata (authors, venue, arXiv ID) web-verified during the project's skeptical-review pass — if a detail looks off, the arXiv/DOI link is authoritative over this summary.*
