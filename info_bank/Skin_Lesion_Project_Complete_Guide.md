# Skin Lesion Classification — Complete Learning Guide

> A from-scratch walkthrough of every concept in your project, written assuming you have **zero ML background**. Each concept builds on the previous one.

---

## How to use this guide

Read sequentially. Don't skip — Section 7's "discriminative learning rates" needs Section 4's "loss function," which needs Section 1's "model." Every term in the original list is in here, in the order you actually need it to make sense.

Each section follows this pattern:
1. **Plain English** — the idea in everyday language
2. **The technical detail** — what's actually happening
3. **In your project** — where this shows up in your code
4. **References** — links to dig deeper

---

# PART 1: Absolute Foundations

## 1.1 What is Machine Learning?

**Plain English.** A normal program is rules you write: `if temperature > 100: print("fever")`. Machine learning flips that — you give the computer **examples** (images of skin + correct labels) and the computer **figures out the rules itself** by adjusting millions of internal numbers until its guesses match the labels.

**Technical detail.** A *model* is a mathematical function `f(x) → y`:
- `x` = input (an image, represented as numbers)
- `y` = output (a guess, like "this is melanoma with 73% confidence")

The model has **parameters** (also called *weights*) — millions of numbers inside it. Training adjusts these parameters so `f(image) ≈ correct_label` across all training examples.

**Two phases:**
- **Training**: Show the model labeled examples. Adjust weights to reduce mistakes.
- **Inference** (also called "prediction" or "evaluation"): Use the trained model on new, unseen images.

**References:**
- [3Blue1Brown — But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) (best 19-minute intro on the planet)
- [Andrej Karpathy — Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) (free course)

---

## 1.2 Tensors, Images, and Numbers

**Plain English.** Computers don't see images — they see grids of numbers. A color image of size 224×224 is actually a **3D grid** of numbers: 224 wide × 224 tall × 3 deep (one layer for Red, one for Green, one for Blue). Each number is a pixel intensity from 0 to 255.

**A tensor** is just a fancy word for "a multi-dimensional array of numbers." Different dimensions:
- 0D tensor = single number (e.g., `5.7`) — also called a *scalar*
- 1D tensor = list of numbers (e.g., `[1.0, 2.0, 3.0]`) — a *vector*
- 2D tensor = grid (e.g., a grayscale image) — a *matrix*
- 3D tensor = grid of grids (e.g., RGB image: `[H, W, 3]`)
- 4D tensor = batch of RGB images (e.g., `[batch_size, 3, H, W]`)

PyTorch (your deep learning framework) does everything in tensors because GPUs can multiply huge tensors very fast in parallel.

**In your project (`dataset.py`):**
```python
image = Image.open(row['image_path']).convert('RGB')  # PIL Image
image = np.array(image)                                # numpy array, shape (H, W, 3)
augmented = self.transform(image=image)                # Albumentations
image = augmented['image']                             # now a torch tensor (3, H, W)
```

**PIL** = Python Imaging Library — the standard library for opening JPG/PNG files. You convert to RGB to guarantee 3 channels (some images are grayscale or have an alpha channel).

**References:**
- [PyTorch — What is a tensor?](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [Pillow (PIL) docs](https://pillow.readthedocs.io/en/stable/)

---

## 1.3 Batches and Epochs

**Plain English.** You can't show the model all 7,000 training images at once — they wouldn't fit in GPU memory. So you show them in small groups (**batches**) of, say, 16 images. One full pass through the entire dataset is one **epoch**. Training takes many epochs (your project: 15–25).

**Why batching matters:** Each batch produces one weight update. Smaller batches = more frequent updates but noisier gradient estimates. Larger batches = smoother gradients but fewer updates per epoch and more memory.

**In your project (`engine.py`):**
```python
"batch_size": 16,        # 16 images per forward pass
"epochs": 20,            # 20 full passes through the dataset
"accum_steps": 4,        # → effective batch size of 64 (more on this later)
```

---

## 1.4 Loss Function — How "wrong" is the model?

**Plain English.** After the model guesses, we need to score how wrong the guess was. The **loss function** is that score. Lower loss = better model. Training is just "make the loss go down."

**For classification, the standard loss is Cross-Entropy:**

$$\text{CE Loss} = -\sum_{i=1}^{C} y_i \cdot \log(\hat{y}_i)$$

Where:
- `C` = number of classes (7 in your project)
- `y_i` = 1 for the correct class, 0 for others (this is called *one-hot*)
- `ŷ_i` = model's predicted probability for class `i`

**Worked example.** Model predicts melanoma probability 0.7, true label IS melanoma:
- Loss = `-log(0.7) = 0.357` → small loss, good guess

Model predicts melanoma probability 0.05, true label IS melanoma:
- Loss = `-log(0.05) = 2.996` → big loss, bad guess. Model gets pushed harder to fix this.

The `-log` shape means **confident wrong predictions are punished harshly**, while confident correct predictions get tiny loss (almost zero).

**In your project (`engine.py`):**
```python
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)
```

**References:**
- [PyTorch CrossEntropyLoss docs](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [StatQuest — Cross-Entropy explained](https://www.youtube.com/watch?v=6ArSys5qHAU)

---

## 1.5 Gradients, Backward Pass, optimizer.step()

**Plain English.** A *gradient* tells you "if I nudge this weight up a tiny bit, will the loss go up or down, and by how much?" If gradient is positive, decreasing the weight reduces loss. If negative, increasing the weight reduces loss.

**Three steps every training iteration:**

1. **Forward pass** — feed image through the model, get predictions, compute loss.
   ```python
   logits = model(images)
   loss = criterion(logits, labels)
   ```

2. **Backward pass** — work backwards through the network, compute the gradient of the loss with respect to *every weight* using the chain rule. This is *backpropagation*.
   ```python
   loss.backward()  # populates .grad on every parameter
   ```

3. **Optimizer step** — actually update the weights using the gradients.
   ```python
   optimizer.step()         # weight -= learning_rate * gradient
   optimizer.zero_grad()    # reset gradients for next iteration
   ```

The simplest update rule (SGD): `new_weight = old_weight - learning_rate * gradient`. Modern optimizers like **AdamW** (used in your project) do something fancier with momentum and per-parameter scaling, but the core idea is the same.

**Learning rate (LR)** is the size of the step. Too large → you overshoot and bounce around. Too small → training takes forever.

**In your project (`engine.py`):**
```python
optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
# Inside training loop:
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

**References:**
- [3Blue1Brown — Backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
- [PyTorch — Optimization tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)

---

# PART 2: Metrics — How We Measure Models

## 2.1 The Confusion Matrix

**Plain English.** A table that shows what the model *predicted* vs what was *actually true*. Rows = truth, columns = prediction.

For 2-class case (melanoma yes/no):

|                  | Predicted: NO mel | Predicted: YES mel |
|------------------|-------------------|--------------------|
| **Actually NO mel** | True Negative (TN) | False Positive (FP) |
| **Actually YES mel** | False Negative (FN) ⚠️ | True Positive (TP) |

For 7 classes (your project), it's a 7×7 grid. Diagonal = correct, off-diagonal = mistakes.

**False Negative is the dangerous one for melanoma** — model said "you're fine" when you actually have cancer. That patient goes home and dies. **False Positive** is annoying (unnecessary biopsy) but not fatal.

**Normalized confusion matrix:** instead of raw counts, each row sums to 1.0 — shows proportion of each true class. Useful when classes have very different sizes (your `nv` has 6705, `df` has 115; raw counts make `df` invisible).

**In your project (`evaluate.py`):**
```python
cm = confusion_matrix(all_labels, preds, normalize='true')  # row-normalized
```

**References:**
- [Wikipedia — Confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
- [scikit-learn confusion_matrix docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

---

## 2.2 Precision, Recall, F1, F2

These are all derived from the confusion matrix.

**Precision** — of everything I called melanoma, how much really was?
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall** (also called Sensitivity, True Positive Rate) — of all actual melanomas, how many did I catch?
$$\text{Recall} = \frac{TP}{TP + FN}$$

**Trade-off:** They pull against each other. If you predict "melanoma" for everything, recall = 100% (you caught them all) but precision is terrible (most are false alarms). If you only predict "melanoma" when you're 99.9% sure, precision is great but recall is awful (you miss many).

**F1 score** — harmonic mean of precision and recall, balanced.
$$F_1 = 2 \cdot \frac{P \cdot R}{P + R}$$

**F-beta score** — same idea but weights recall `β` times more than precision:
$$F_\beta = (1+\beta^2) \cdot \frac{P \cdot R}{\beta^2 \cdot P + R}$$

**F2 score** = `β=2` → weights **recall 4× more than precision** (the `β²` in the denominator). Used in medical screening because **missing a cancer is far worse than a false alarm**.

**In your project (`opt_learned_thresholds.py`):**
```python
score = fbeta_score(y_binary, preds, beta=2)  # F2
```

**References:**
- [scikit-learn — Precision, recall and F-measures](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics)
- [Wikipedia — F-score](https://en.wikipedia.org/wiki/F-score)

---

## 2.3 ROC and AUC

**ROC** = Receiver Operating Characteristic curve.

**The setup.** A classifier outputs probabilities (e.g., `P(melanoma) = 0.73`). To make a final yes/no decision, you pick a **threshold**: predict "yes" if probability > 0.5, otherwise "no." Different thresholds give different precision/recall trade-offs.

**The ROC curve** plots **True Positive Rate** (recall) on the y-axis vs **False Positive Rate** (`FP / (FP + TN)`) on the x-axis, as you sweep the threshold from 0 to 1.

```
TPR  1.0 |          .----- ← perfect classifier
         |       .-'
         |     .'
         |   .'
         | .'      ← random guessing (diagonal)
         |.________________________
         0.0          FPR        1.0
```

**AUC** = **Area Under the (ROC) Curve**. A single number from 0 to 1:
- **AUC = 1.0** → perfect classifier
- **AUC = 0.5** → random guessing (the diagonal line)
- **AUC = 0.9808** → your project's ensemble (excellent)

**Why AUC is great:**
1. **Threshold-independent** — measures the model's ranking ability, not a specific decision rule.
2. **Probabilistic interpretation:** AUC = probability that a randomly-picked positive example gets a higher score than a randomly-picked negative example.
3. **Robust to class imbalance** — unlike accuracy.

**Macro AUC for multi-class.** With 7 classes, you compute AUC for each class one-vs-rest (e.g., "mel vs everything else"), then **average** them with equal weight. That's why it's called *macro* — every class counts the same regardless of size.

**Why this matters for you.** If you used *micro* averaging or plain accuracy, the giant `nv` class (6,705 samples) would dominate. The 115-sample `df` class would barely move the needle. **Macro forces the model to be good at every class** — which is what you want medically. Missing 50% of `df` is just as bad as missing 50% of `nv`.

**In your project (`engine.py`):**
```python
auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
# ovr = one-vs-rest
```

**References:**
- [Google ML Crash Course — ROC and AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
- [StatQuest — ROC and AUC clearly explained](https://www.youtube.com/watch?v=4jRBRDbJemM)
- [scikit-learn — roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)

---

## 2.4 Headline Metric, Naive Models, Class Imbalance

**Headline metric.** The single number you put on a leaderboard or resume. For your project: **macro AUC**. Choosing the right headline metric matters because it shapes every decision (which model to keep, which augmentation works, etc.).

**Class imbalance.** Your dataset has wildly unequal classes:

| Class | Count | % of total |
|-------|-------|------------|
| nv | 6,705 | 67% |
| mel | 1,113 | 11% |
| bkl | 1,099 | 11% |
| bcc | 514 | 5% |
| akiec | 327 | 3% |
| vasc | 142 | 1.4% |
| df | 115 | 1.2% |

Ratio: `nv:df = 58:1`. This is **severe imbalance**.

**Naive model = "always predict the majority class."** A model that **always says `nv`** gets:
- Accuracy: 67% (looks decent!)
- Macro AUC: 0.5 (terrible — random)
- Melanoma recall: 0% (catches zero cancers)

This is why **accuracy lies** on imbalanced data. A model could look "67% accurate" while being clinically useless.

**Class distribution.** The actual proportion of each class. The training set's class distribution should match (roughly) the real-world distribution you expect at deployment — otherwise the model's calibration is off.

**References:**
- [Imbalanced-learn — User guide](https://imbalanced-learn.org/stable/user_guide.html)
- [Google ML Crash Course — Imbalanced data](https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets)

---

# PART 3: The Dataset & Splits

## 3.1 Why HAM10000 is split into part_1 and part_2

**Pure logistics — has nothing to do with ML.** The dataset is 10,015 high-resolution dermatoscopic JPGs, which adds up to a few GB. When the dataset was uploaded to public hosts (Kaggle, Harvard Dataverse), they had **per-folder file count limits** (some platforms cap folders at ~5000–10000 files for browser previews). So the curators split images into:
- `HAM10000_images_part_1/` — first ~5000
- `HAM10000_images_part_2/` — remaining ~5000

The metadata CSV has all 10,015 entries; your `load_metadata()` just walks both folders and builds an `image_id → file_path` map:

```python
for image_dir in image_dirs:
    if os.path.exists(image_dir):
        for fname in os.listdir(image_dir):
            if fname.endswith('.jpg'):
                image_id = fname.replace('.jpg', '')
                image_path_map[image_id] = os.path.join(image_dir, fname)
```

**Interview-safe answer:** "It's a hosting artifact — the dataset providers split files across two folders for upload convenience. My code merges them transparently."

**Reference:**
- [HAM10000 on Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

---

## 3.2 Train / Validation / Test Split

You split the data into **three** disjoint sets:

| Set | Purpose | Your project | When used |
|-----|---------|--------------|-----------|
| **Train** | Model sees these to learn | 70% (~7000 imgs) | Every epoch, weight updates |
| **Validation** | Tune hyperparameters, pick best epoch | 15% (~1500) | After every epoch, NO weight updates |
| **Test** | Final unbiased score | 15% (~1500) | ONCE at the very end |

**Critical rule: never train on val/test data.** That's called *leakage* and inflates your numbers. Your Phase 9 ensemble's 0.9808 is trustworthy *because* the test set was never used to pick weights, augmentations, or thresholds.

**Validation vs Test.** Val is used many times during development (e.g., "should I keep epoch 12 or epoch 15?"). Test is a *one-shot* honest estimate of real-world performance. If you tune on test, you're effectively training on it.

---

## 3.3 Stratified Splitting

**Plain English.** A *stratified* split preserves the **class proportion** in each subset. If `df` is 1.2% of your data, stratification ensures `df` is also 1.2% of train, 1.2% of val, 1.2% of test.

**Why it matters.** If you split randomly with only 115 `df` images, you might (by bad luck) get 0 `df` images in the test set. Then your test AUC for `df` is undefined and your macro AUC breaks. With **rare classes, random splits are dangerous**.

**In your project (`dataset.py`):**
```python
train_df, temp_df = train_test_split(
    df,
    test_size=(val_size + test_size),
    stratify=df['dx'],          # ← THIS is the key
    random_state=42
)
```

`random_state=42` makes the split reproducible — same seed = same split every time, so your val/test sets stay consistent across all 14 phases.

**Reference:**
- [scikit-learn — Stratified shuffle split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

---

## 3.4 WeightedRandomSampler — Fixing Imbalance During Training

**Plain English.** Even with stratified splits, your *training batches* still look imbalanced — pick 16 random training images, you'll get ~11 `nv`, ~2 `mel`, maybe 0 `df`. The model sees 6705 `nv` examples per epoch but only 115 `df`, so it learns "always guess `nv`" works most of the time.

**The fix: sample rare classes more often.** Each epoch, instead of seeing each image once, you draw images **with replacement** weighted by `1 / class_count`:

| Class | Count | Weight per sample |
|-------|-------|-------------------|
| nv | 6705 | 1/6705 = 0.000149 |
| df | 115 | 1/115 = 0.00870 → **58× higher** |

This produces **balanced batches** — roughly 2-3 samples of each class per batch, regardless of how rare the class is in the underlying data. The model sees `df` and `nv` equally often during training.

**Oversampling vs Undersampling:**
- *Oversampling* (what `WeightedRandomSampler` does): pull rare-class images multiple times per epoch.
- *Undersampling*: throw away majority-class images. Wastes data — usually worse.

**In your project (`dataset.py`):**
```python
def get_weighted_sampler(df):
    class_counts = df['dx'].value_counts().to_dict()
    weights = [1.0 / class_counts[dx] for dx in df['dx']]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
```

**Subtle point — interview gold:** Why use sampler instead of class-weighted loss?
- **Sampler changes batch composition** — model sees rare classes more often, more weight updates push it toward rare-class features.
- **Class weights in loss** rescale gradient magnitudes but the model still sees mostly `nv` in each batch. Often gives weaker effect.
For severe imbalance (>10:1), sampler tends to win.

**References:**
- [PyTorch — WeightedRandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler)
- [Buda et al., 2017 — A systematic study of imbalanced learning](https://arxiv.org/abs/1710.05381)

---
# PART 4: Image Transforms and Augmentation

## 4.1 What is "augmentation" and why?

**The core problem.** You have 7,000 training images. Deep models have millions of parameters. Without help, the model **memorizes the exact training images** (called *overfitting*) and fails on anything new.

**The trick — augmentation.** Every time the model sees an image, randomly transform it: flip it, rotate it, change colors slightly. The model now effectively sees infinite variations of each image and can't memorize. It's forced to learn **robust features** (the lesion's shape and color) instead of fragile ones (the exact pixel pattern).

**Albumentations** is the go-to Python library for image augmentation. It's faster than torchvision and has more medical-imaging-friendly transforms.

**References:**
- [Albumentations docs](https://albumentations.ai/docs/)
- [Buslaev et al., 2020 — Albumentations paper](https://www.mdpi.com/2078-2489/11/2/125)

---

## 4.2 Why these specific augmentations?

Your `transforms.py` training pipeline:

```python
A.HorizontalFlip(p=0.5),
A.VerticalFlip(p=0.5),
A.RandomRotate90(p=0.5),
A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
A.OneOf([A.GaussNoise(), A.GaussianBlur(), A.MotionBlur()], p=0.3),
A.OneOf([A.OpticalDistortion(), A.GridDistortion()], p=0.2),
A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
ToTensorV2(),
```

### Rotation invariance — the key insight for dermoscopy

**Plain English.** A skin lesion has **no "up"**. When a dermatologist photographs a mole on your back, the camera angle is arbitrary. The lesion looks just as much like a melanoma whether the photo is rotated 0°, 90°, 180°, or 270°. We say the task is **rotation-invariant**.

This is **unusual!** For natural images (cars, faces, text), rotation matters — an upside-down car is wrong, upside-down text is unreadable. So you'd never use vertical flips on ImageNet. But for dermoscopy you absolutely should — it's free data.

**Therefore:**
- `HorizontalFlip(p=0.5)` ← 50% chance to flip left-right (a lesion is symmetric this way)
- `VerticalFlip(p=0.5)` ← 50% chance to flip top-bottom (also fine for lesions)
- `RandomRotate90(p=0.5)` ← random 0/90/180/270° rotation

Each augmentation has probability `p` of being applied. Combined, you get 8 effective rotational symmetries × random small color jitters → enormous data variety.

### ColorJitter — brightness, contrast, saturation, hue

**What it does.** Randomly adjusts:
- **Brightness** — how light/dark the image is overall.
- **Contrast** — the difference between light and dark areas.
- **Saturation** — how vivid the colors are.
- **Hue** — shifts colors along the rainbow (red → orange → yellow...).

**Why for dermoscopy.** Different clinics use different cameras, lights, and dermatoscope models. Image colors vary across hospitals. Without ColorJitter, your model might learn "lesions photographed under fluorescent light = nv" (an irrelevant artifact). With ColorJitter, the model sees random color perturbations and learns "lesion shape and texture" instead — which is what actually matters.

### GaussNoise, GaussianBlur, MotionBlur

These simulate **camera imperfections**:
- **GaussNoise** — random noise added to pixel values, like a slightly grainy sensor.
- **GaussianBlur** — out-of-focus look, simulates a slightly mis-focused camera.
- **MotionBlur** — streak in one direction, simulates the photographer's hand moving.

`A.OneOf(...)` means **pick one of these at random**, with overall probability 0.3 of any being applied. You don't want all three at once — that would be unrealistic.

### Geometric distortions

- **ShiftScaleRotate** — small shift, zoom, and rotation up to 30°.
- **OpticalDistortion / GridDistortion** — barrel/pincushion effects, simulates lens distortion.

These teach the model that small geometric perturbations don't change the diagnosis.

---

## 4.3 ImageNet, normalization, and pretrained backbones

### What is ImageNet?

**ImageNet** is the canonical massive image dataset: ~1.2 million labeled images across 1,000 everyday categories (dog breeds, cars, mushrooms, etc.). For ~10 years it was the de facto benchmark for image models. Almost every model architecture has a version that's been **pretrained on ImageNet** — meaning someone trained it on ImageNet for weeks on expensive hardware, and the resulting weights are downloadable.

### What is a "backbone"?

**Plain English.** A modern vision model has two parts:
- **Backbone** — the deep feature extractor. Takes a raw image, produces a rich numerical summary (a *feature vector*, e.g., 768 numbers).
- **Head** — a small layer on top that maps features → class predictions.

Think of the backbone as "understands what's in the image" and the head as "decides which of my 7 categories this maps to."

### Why use pretrained backbones?

Training a backbone from scratch on 7,000 images would fail catastrophically — the model needs millions of images to learn good visual features. But if someone already trained the backbone on **ImageNet's 1.2M images**, those features (edges, textures, shapes) **transfer** to medical imaging, even though ImageNet has no skin lesions. This is **transfer learning**, the most important technique in modern computer vision.

Your project uses ImageNet-pretrained backbones for ResNet50, EfficientNetV2, ViT, and others. Newer ones (DINOv2, SigLIP, AIMv2) are pretrained on **even larger** datasets (LAION-5B, internal Apple/Meta data), but the principle is identical.

### Normalization — making inputs match what the backbone expects

When the ImageNet folks trained backbones, they fed them images that were normalized:
$$\text{normalized\_pixel} = \frac{\text{raw\_pixel}/255 - \mu}{\sigma}$$

With ImageNet's specific statistics:
```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # R, G, B mean across all training images
IMAGENET_STD  = [0.229, 0.224, 0.225]   # R, G, B std
```

If you feed a backbone raw `[0, 255]` pixels but it was trained on normalized `[~ -2, +2]` values, the model is essentially seeing garbage. You **must** normalize using the **same statistics the backbone was trained with**. That's why `IMAGENET_MEAN` and `IMAGENET_STD` are hardcoded into your transforms.

**References:**
- [Deng et al., 2009 — ImageNet paper](https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf)
- [PyTorch — Transfer learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

## 4.4 Validation and Test Transforms — Deterministic Eval

**Plain English.** During *training*, you add random noise to make the model robust. During *evaluation*, you want the **same image to produce the same prediction every time** — otherwise your AUC fluctuates randomly between runs and you can't compare experiments.

So validation/test transforms have:
- `Resize` (deterministic — always to same size)
- `Normalize` (deterministic — same statistics)
- `ToTensorV2` (deterministic — just data type conversion)
- **No** flips, rotations, color jitter, blur, or any randomness.

Same image in → same prediction out. The metrics you compute are reproducible. This is what "deterministic" means in this context.

```python
def get_val_transform(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
```

**The exception — Test-Time Augmentation (TTA).** In Phase 8, you deliberately *add* randomness back at inference and average results. We'll cover this later.

---

# PART 5: Model Architecture Concepts

## 5.1 What is a Vision Transformer (ViT)?

**Plain English.** Originally, computer vision used CNNs (ResNet, EfficientNet) which slide small filters across the image. In 2020, Google's Vision Transformer paper proved a different approach works just as well or better:

1. Cut the image into a grid of small patches (e.g., 16×16 pixels each → a 224×224 image becomes 14×14 = 196 patches).
2. Treat each patch like a "word" in a sentence.
3. Run them through a Transformer (the same architecture that powers ChatGPT).
4. The Transformer's self-attention lets every patch "look at" every other patch, capturing long-range dependencies.

**ViT-Base** has:
- 12 stacked transformer **blocks** (also called "layers")
- Each block has self-attention + a small feedforward net
- Total: ~86M parameters
- Input: 224×224 image, output: a 768-dim feature vector (the `[CLS]` token, see below)

**Reference:** [Dosovitskiy et al., 2020 — An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929)

---

## 5.2 Lower vs Higher Blocks — the Hierarchy

**The key insight.** When a deep network is trained, different layers learn different things:

| Block depth | What it learns |
|-------------|---------------|
| Lower (blocks 0-3) | **Generic** features — edges, textures, color gradients |
| Middle (blocks 4-7) | Mid-level patterns — corners, simple shapes |
| Higher (blocks 8-11) | **Task-specific** semantics — "this looks like a dog face" or "this is a melanoma pattern" |

**Why this matters for fine-tuning.** When you take an ImageNet-pretrained ViT and apply it to skin lesions:
- The **lower blocks already work** — edges and textures are universal across all images.
- The **higher blocks need adjustment** — what they learned about "dog faces" needs to be re-trained for "lesion patterns."

So you can **freeze the lower blocks** (don't update their weights, save compute) and **train only the higher blocks + the new head**.

---

## 5.3 Partial Freezing — and why it saves compute

**What "freezing" means.** Set `param.requires_grad = False`. The forward pass still uses these weights, but the backward pass **doesn't compute gradients** for them, and the optimizer never updates them.

**Why this saves compute (the actual math):**
- **Backward pass** is roughly 2× the cost of forward pass — and most of that cost is computing gradients for weights.
- If you freeze the first 8 blocks of a 12-block ViT, you skip ~67% of the backward pass cost.
- You also skip storing intermediate gradients in GPU memory → can use larger batches.

**Plus, it acts as regularization.** With fewer trainable parameters, the model is less likely to overfit to your small medical dataset.

**In your project (`model.py`):**
```python
def get_vit_model(num_classes=7, freeze_until_block=8):
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", ...)
    for param in model.parameters():
        param.requires_grad = False           # freeze everything first
    for i in range(8, 12):                    # then unfreeze blocks 8, 9, 10, 11
        for param in model.vit.encoder.layer[i].parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True            # head is always trainable
```

---

## 5.4 The "Head" and Random Initialization

**The head** is the last layer that maps backbone features → class scores. For 7 classes:
```python
head = nn.Linear(768, 7)   # input: 768-dim features, output: 7 class scores
```

**Critical point: the head is randomly initialized.** When you load a pretrained ViT for ImageNet's 1000 classes, you replace its 1000-class head with a fresh **7-class head** whose weights are random. The backbone's features are good, but the head has *no idea* what to do with them yet.

This explains why we need...

---

## 5.5 Discriminative Learning Rates (1e-3 head, 1e-5 backbone, 100× ratio)

**The problem.**
- The **backbone** has been trained for weeks on millions of images. Its weights are **already good**. Big updates would *destroy* what it learned.
- The **head** is **random**. It needs **big updates** to learn fast.

If you use a single learning rate, you face an impossible trade-off:
- LR too high → backbone gets destroyed.
- LR too low → head takes forever to train.

**The solution — different LRs for different parts:**
```python
optimizer = AdamW([
    {'params': backbone_params, 'lr': 1e-5},   # tiny updates, gentle
    {'params': head_params,     'lr': 1e-3},   # 100× bigger updates
], weight_decay=1e-4)
```

**Why specifically 100× ratio (1e-3 vs 1e-5)?**
- Below 10× — head trains too slowly, you waste epochs.
- Above 1000× — backbone effectively frozen, you don't get the benefit of fine-tuning.
- ~100× is the empirical sweet spot found across the deep learning literature for transfer learning.

Howard and Ruder formalized this in 2018 ("Universal Language Model Fine-tuning") as **ULMFiT**. The idea generalized to vision and is now standard.

**Reference:** [Howard & Ruder, 2018 — ULMFiT](https://arxiv.org/abs/1801.06146)

---

## 5.6 OneCycleLR Scheduler — Warmup + Cosine Decay

**What is a learning rate scheduler?** Instead of a fixed LR, you change it across training. OneCycleLR is the most common modern choice for fine-tuning.

**The shape:**
```
LR
 ↑
max │      .---.
    │    .'     '.
    │   /          '.
    │  /             '.
    │ /                '-.
    │/                    '------___
    └─────────────────────────────────→ training step
    0   ↑                              end
        warmup phase (10%)
```

**Three phases:**
1. **Warmup (first 10% of steps).** LR starts low and ramps up linearly to `max_lr`. This is the critical "step 0" protection.
2. **Cosine decay (next 90%).** LR smoothly decreases from `max_lr` to nearly 0 following half a cosine wave.
3. **End of training.** LR is tiny → tiny weight updates → model settles into a stable solution.

### Why warmup matters — the "step 0" problem

**At step 0**, the head is random. The first batch's gradient is wild and noisy because the random head produces nonsense outputs. If you immediately apply a high LR to the *backbone*, those wild gradients propagate back and **destroy the carefully-pretrained backbone weights**.

Warmup says: "for the first 10% of training, take baby steps." This gives the head time to start producing sensible gradients before the backbone gets updated meaningfully.

### Why cosine decay matters

**Late in training**, the model is near a good solution. Big LR steps would knock it out. Cosine decay smoothly reduces the step size, letting the model "settle" into the bottom of a loss valley.

**In your project (`engine.py`):**
```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[1e-5, 1e-3],          # one max per param group (backbone, head)
    total_steps=total_steps,
    pct_start=0.1,                # 10% warmup
)
# Inside training loop:
optimizer.step()
scheduler.step()                  # advance the LR every batch
```

**Reference:** [Smith, 2018 — Super-convergence with OneCycleLR](https://arxiv.org/abs/1708.07120)

---

# PART 6: Memory Tricks — Fitting Big Models in 8GB VRAM

Your project ran on an RTX 4060 Mobile (8GB VRAM). Modern ViTs barely fit. Three tricks make it possible.

## 6.1 Mixed Precision (AMP), `autocast`, `GradScaler`

**The problem.** Standard floating-point numbers in PyTorch are **fp32** (32 bits each, ~4 bytes per number). A ViT-Base with 86M parameters + activations + gradients = several GB.

**The trick.** Use **fp16** (16-bit, half the memory) for most operations. fp16 is:
- 2× faster on modern GPUs (NVIDIA A100, RTX 30/40 series have dedicated fp16 "tensor cores")
- Half the memory
- ...but has less precision and less *range* (smallest fp16 ≈ 6e-5; smallest fp32 ≈ 1e-38)

**The danger.** Some operations underflow in fp16 (gradients become 0, training stalls).

**Mixed precision** = use fp16 where safe, fp32 where necessary.

### `autocast` — automatic per-op precision selection

```python
with torch.cuda.amp.autocast():
    outputs = model(images)         # forward pass: most ops in fp16
    loss = criterion(outputs, labels)  # loss in fp32 for stability
```

PyTorch automatically picks fp16 for matrix multiplies (safe and fast) and fp32 for things like `softmax` and `loss` (need precision).

### `GradScaler` — preventing gradient underflow

**The problem in detail.** Loss values are tiny near a good solution (e.g., 0.001). When you backprop, gradients can be even smaller. In fp16, anything below ~6e-5 rounds to **zero** → no learning.

**The fix.** Multiply the loss by a big number (e.g., 65536) *before* computing gradients. This shifts gradients into the fp16 representable range. Then divide back before applying the optimizer.

```python
scaler = GradScaler()
# Forward
with autocast():
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()    # multiplies loss by scale before backward
scaler.step(optimizer)            # divides gradients by scale, then steps
scaler.update()                   # adjusts scale for next iter (decreases if NaN seen)
```

**Net effect:** ~50% VRAM savings, ~2× speedup, with no accuracy loss. This is **the** trick that makes 8GB VRAM viable for ViTs.

**References:**
- [PyTorch — Automatic Mixed Precision tutorial](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
- [Micikevicius et al., 2017 — Mixed Precision Training](https://arxiv.org/abs/1710.03740)

---

## 6.2 Gradient Accumulation

**The problem.** You want effective batch size 64 (smoother gradients, faster training), but only 16 fits in VRAM.

**The trick.** Process 4 micro-batches of 16, **accumulate gradients** without updating, then update once.

```python
optimizer.zero_grad()
for step, (images, labels) in enumerate(loader):
    with autocast():
        loss = criterion(model(images), labels) / 4   # divide by accum_steps
    scaler.scale(loss).backward()                      # accumulates into .grad

    if (step + 1) % 4 == 0:                            # every 4 micro-batches
        scaler.step(optimizer)                          # one update
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()                           # reset
```

**Why divide by 4?** Cross-entropy loss averages over examples in a batch. If you accumulate 4 batches' gradients without dividing, you'd be summing 4 averages → effectively a 4× larger gradient. Dividing keeps the math equivalent to a real batch of 64.

**`backward()` does NOT update weights** — it just computes gradients and adds them to `param.grad`. The actual update happens in `optimizer.step()`. Gradient accumulation exploits this separation.

**Net effect:** Train as if you had a 64-image batch, but with 16-image VRAM.

---

## 6.3 Gradient Checkpointing

**The problem.** During the forward pass, every layer's output (called *intermediate activations*) must be **stored** in memory because backprop needs them to compute gradients. For a 12-layer ViT, this is huge.

**The trick.** Don't store activations. **Recompute them** during the backward pass.

**Trade-off:**
- ✅ Saves a lot of memory (~30–40% for ViTs)
- ❌ Costs ~30% more compute (one extra forward pass for the recomputation)

For an 8GB VRAM card, this trade is worth it — without it, ViT-Base barely fits at batch size 4. With it, you can use batch size 16.

**In your project (`train.py`):**
```python
model.vit.encoder.gradient_checkpointing = True
```

One line, big impact.

**Reference:** [Chen et al., 2016 — Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)

---

## 6.4 Putting it together

Combined effect of all three tricks on RTX 4060 8GB:
| Without tricks | Mixed precision only | + Grad accumulation | + Grad checkpointing |
|----------------|---------------------|---------------------|----------------------|
| Batch 4, OOM at ViT | Batch 8 | Effective batch 32 | Effective batch 64 |

This is what "engineered an 8GB-VRAM training pipeline" means on your resume — the combination is what makes consumer-hardware fine-tuning of 86M-parameter ViTs work.

---

# PART 7: MLflow and Best-Model Selection

## 7.1 What is MLflow?

**Plain English.** Imagine running 14 different experiments. Each has different hyperparameters, learning curves, final scores, model checkpoints. Without organization, you forget which run did what after a week. **MLflow** is a tracking server that logs:

- **Params**: hyperparameters (LR, batch size, epochs)
- **Metrics**: loss, AUC, accuracy at each epoch
- **Artifacts**: model files, plots
- **Runs**: each one timestamped and named

`mlflow ui` opens a web dashboard to compare runs side-by-side.

**In your project (`engine.py`):**
```python
mlflow.set_experiment("skin-lesion-all-phases")
with mlflow.start_run(run_name="phase5_dinov2_registers"):
    mlflow.log_params(cfg)
    for epoch in range(epochs):
        # training...
        mlflow.log_metrics({"train_loss": ..., "val_auc": ..., "val_acc": ...}, step=epoch)
    mlflow.log_metrics({"test_auc": ..., "test_acc": ...})
```

**Reference:** [MLflow docs](https://mlflow.org/docs/latest/index.html)

---

## 7.2 Why save the model on val AUC, not val loss?

**Plain English.** Val loss and val AUC don't move in lockstep. The model can have *lower loss* but *worse AUC* — for example, if it gets very confident on common classes (driving down loss) at the expense of rare classes (which doesn't tank loss much but tanks macro AUC).

**Your headline metric is macro AUC**, so the *best model* is the one with highest macro AUC, not lowest loss.

**Code (`train.py`):**
```python
if val_metrics['auc'] > best_auc:
    best_auc = val_metrics['auc']
    torch.save(model.state_dict(), 'best_model.pth')
```

**Rule of thumb:** Always save checkpoints based on the metric you actually care about for deployment. If your final report says "I achieved 0.9808 AUC," the checkpoint you used must have been selected on AUC.

---

# PART 8: Evaluation Tools — Per-Class AUC, Confusion Matrix, GradCAM

## 8.1 Per-Class AUC

**Why bother — isn't macro AUC enough?** Macro AUC averages all classes. But in a medical paper or interview, you need to say *which classes the model is good at* and *which it struggles with*.

**Computation.** For each class `i`, treat it as a binary problem ("class i vs everything else") and compute AUC:

```python
for i, cls in enumerate(CLASS_NAMES):
    binary_labels = (all_labels == i).astype(int)   # 1 if class is i, else 0
    cls_auc = roc_auc_score(binary_labels, all_probs[:, i])
    print(f"{cls}: {cls_auc:.4f}")
```

**Your project's per-class AUCs** (from PLAN_DERM.md): vasc/df/bcc/akiec saturate near 0.99+. Mel sits at 0.95. This tells you exactly **where the headroom is** — improvements should target mel/nv boundary, not the already-solved classes.

---

## 8.2 Normalized Confusion Matrix

Already covered conceptually in 2.1. The "normalized" part means each row is divided by the row total, so values represent **percentages of each true class**.

**Why normalize?** In your data, `nv` has 1006 samples in the test set, `df` has 17. If you plot raw counts:
- `nv → nv` cell shows 935 (looks impressive)
- `df → df` cell shows 13 (looks tiny — but it's actually 76% recall, equivalent to nv's 93%!)

Normalization makes them comparable. Always use `normalize='true'` for imbalanced data.

---

## 8.3 GradCAM — "Where is the model looking?"

**The medical AI nightmare.** Your model gets 88% accuracy. Great! But what if it's actually looking at the **ruler** in the corner of the photo, or the **sticker** the dermatologist placed nearby, instead of the lesion itself? The model would still score well on this dataset (because rulers happen to correlate with how the photo was taken) but **fail catastrophically** in deployment when those artifacts aren't there.

This is called **shortcut learning** and it's a real, documented problem in medical AI. (See [Geirhos et al., 2020](https://arxiv.org/abs/2004.07780).)

**GradCAM = Gradient-weighted Class Activation Mapping.** It produces a heatmap showing **which pixels contributed most** to the model's decision.

**How it works (intuition).** Take the gradient of the predicted class score with respect to the last convolutional/attention layer's activations. Weight the activations by these gradients. Sum across channels. Where the result is high → the model "looked there" most.

**For ViTs**, you target the last attention block's `layernorm_before`:
```python
target_layer = model.vit.encoder.layer[-1].layernorm_before
cam = GradCAM(model=model, target_layers=[target_layer])
grayscale_cam = cam(input_tensor=image, targets=[ClassifierOutputTarget(label)])[0]
```

**Why it matters for clinical credibility.** A reviewer or doctor will ask "is the model actually looking at the lesion?" GradCAM lets you show side-by-side: input image + heatmap. If the heat is on the lesion, the model has learned the right thing.

**References:**
- [Selvaraju et al., 2017 — Grad-CAM paper](https://arxiv.org/abs/1610.02391)
- [pytorch-grad-cam library](https://github.com/jacobgil/pytorch-grad-cam)
# PART 9: The 7 Models — Why Each One

You benchmarked 7 different vision foundation models. The point was a **controlled comparison** — same data splits, same evaluation, same training infrastructure. Only the architecture (and freeze strategy) changes. This is what makes the project rigorous.

## 9.0 Two key concepts you need first

### Linear probe vs Full fine-tune

**Linear probe.** Freeze the entire backbone. Only train the head (a single linear layer). Cheap, fast, and tests **how good the pretrained features are** out of the box.

**Full fine-tune (FT).** All layers are trainable, with discriminative LRs. The backbone adapts to your specific task. More expensive, more risk of overfitting, but typically higher ceiling on small datasets if you can afford it.

**Why you used different strategies for different models:**
- Smaller models (ResNet50, EfficientNetV2, ViT, DINOv2) → could afford full FT on 8GB VRAM.
- Larger models (AIMv2 307M, SigLIP 400M) → wouldn't fit. Used linear probe.

This is partly an honest hardware constraint, partly a deliberate choice (linear probe is the standard way to evaluate self-supervised models per their original papers).

### Self-supervised learning (SSL) — the modern shift

**Old world (ImageNet supervised).** Train a model on 1.2M images **with class labels**. Backbone learns features tuned to dog/cat/car distinctions.

**New world (SSL).** Train on **billions of unlabeled images** using cleverly-designed proxy tasks (predict masked patches, contrast different views, etc.). The backbone learns universal visual features without needing labels. Often **stronger than ImageNet-supervised** on downstream tasks because the data is much bigger.

DINOv2, AIMv2, SigLIP are all SSL or SSL-adjacent.

---

## 9.1 Phase 1 — ResNet50 (Frozen, Linear Probe Baseline)

**What it is.** ResNet50 is a 2015 CNN, the workhorse of pre-ViT vision. 23M parameters. ImageNet-supervised pretrained.

**Why.** **It's the baseline.** Every benchmark needs a "if you do nothing fancy, this is what you get." ResNet50 with frozen backbone + linear head answers: *"can a basic CNN extract enough info from skin lesions to be useful?"* Test AUC: **0.8940**. Yes, but barely.

**Strategy.** `freeze_backbone=True` — only the final FC layer trains. ~2K trainable parameters out of 23M.

```python
model = timm.create_model("resnet50", pretrained=True, num_classes=7)
for name, p in model.named_parameters():
    if "fc" not in name:
        p.requires_grad = False
```

**Reference:** [He et al., 2015 — Deep Residual Learning](https://arxiv.org/abs/1512.03385)

---

## 9.2 Phase 2 — EfficientNetV2-M (Best Single Model)

**What it is.** Google 2021. CNN with optimized scaling. 54M params. ImageNet-21k pretrained.

**Why.** EfficientNet family is famous for being **the best CNN per FLOP** — high accuracy without ballooning compute. The "V2" added training-time improvements.

**Strategy.** Full fine-tune at **384×384** (higher resolution than other phases' 224). Higher resolution helps medical imaging — small details (asymmetric border, irregular pigmentation) matter for melanoma diagnosis.

**Result.** 0.9685 AUC — best single model in your benchmark.

**Reference:** [Tan & Le, 2021 — EfficientNetV2](https://arxiv.org/abs/2104.00298)

---

## 9.3 Phase 3 — ViT-Base (First Transformer)

**What it is.** Google 2020. The original Vision Transformer. 86M params. ImageNet-21k pretrained.

**Why include it.** Establishes the transformer baseline before introducing the SSL transformers. Lets you say "DINOv2 (SSL ViT) > supervised ViT" with controlled evidence.

**Strategy.** Partial freeze — last 4 blocks + head trainable. Saves VRAM enough to fit batch 16.

**Result.** 0.9538 AUC. Slightly worse than EfficientNetV2 because, on a small dataset, ViT's lack of inductive bias (more on this in Section 11) hurts. But not by much.

**Reference:** [Dosovitskiy et al., 2020 — ViT](https://arxiv.org/abs/2010.11929)

---

## 9.4 Phase 4 — DINOv2-Base (Self-Supervised ViT)

**What it is.** Meta 2023. Same ViT-Base architecture, but pretrained via **self-distillation** on 142M curated images, no labels.

**Why.** Tests whether SSL pretraining beats ImageNet-supervised on dermoscopy. DINOv2 is the gold-standard SSL ViT — its features are widely considered SOTA for transfer learning.

**Strategy.** Full FT, batch 16, 224×224.

**Result.** 0.9458 AUC. **Worse than supervised ViT** in your linear-probe-style setup? No — this is full FT and it's still slightly behind EfficientNetV2 because EfficientNetV2 had higher resolution. The honest comparison is Phase 3 vs Phase 4 (both ViT-Base, 224, full FT-ish): 0.9538 vs 0.9458 — *supervised slightly beats DINOv2 on this small dermoscopy dataset*. Interesting empirical finding.

**Reference:** [Oquab et al., 2023 — DINOv2](https://arxiv.org/abs/2304.07193)

---

## 9.5 Phase 5 — DINOv2 + Registers (The Star of the Show)

**What it is.** Same DINOv2, but with the **registers fix** from Darcet et al., 2024.

### What are "attention artifacts" and "registers"?

**The bug discovered in 2023.** When researchers visualized ViT attention maps, they found weird **high-norm tokens** appearing in **background patches** (sky, blank areas). The model was using these patches as **scratch memory** to do internal computation, even though those patches contain no useful image content. This is called an **attention artifact** — attention sinks to spots that semantically shouldn't matter.

**The fix.** Add a few dedicated **register tokens** to the input sequence — extra "blank slates" that the model can use as scratch memory **without hijacking image patches**. These tokens are learned during pretraining.

**Result.** Cleaner attention maps, slightly better downstream performance. The registers paper showed +1-2% improvement on linear probe benchmarks across many tasks.

**Why this is exciting for your project.** Your Phase 5 is one of the **first published HAM10000 benchmarks** for DINOv2-with-Registers. Test AUC: **0.9653**, beating vanilla DINOv2's 0.9458 by 2 points.

**The HuggingFace model ID:** `facebook/dinov2-with-registers-base`

**Reference:** [Darcet et al., 2024 — Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) (ICLR 2024 outstanding paper award)

---

## 9.6 Phase 6 — AIMv2 (Apple's Autoregressive SSL)

**What it is.** Apple, late 2024. **Autoregressive** SSL — instead of contrastive learning (DINOv2) or masked modeling (MAE), AIMv2 trains the vision encoder to **predict the next image patch** plus the next text token in image-caption pairs. Like GPT but for images. 307M params.

**What "autoregressive" means.** Generates output sequentially, each prediction depending on previous ones. GPT is autoregressive (next-word prediction). AIMv2 applies this to images.

**Strategy.** **Linear probe only.** 307M params is too big to full-FT on 8GB VRAM.

### Why timm instead of HuggingFace?

```python
self.backbone = timm.create_model("aimv2_large_patch14_224", pretrained=True, num_classes=0)
```

- **HuggingFace transformers** had AIMv2 support that was buggy at the time you ran experiments — checkpoint loading issues, missing keys, wrong tensor shapes.
- **timm (PyTorch Image Models)** is a Ross Wightman library specialized for vision. It re-implements models to work cleanly with the PyTorch ecosystem and often has more reliable weights for new architectures.

`num_classes=0` is timm's way to say "give me the backbone only, no classification head" — useful when you'll attach your own head.

**Why remove the pretraining head?** AIMv2's pretraining head was for image generation, not classification. You want the **features** the backbone produces, not the gen head's outputs. Adding `num_classes=0` returns features directly so you can pipe them to your own `nn.Linear(hidden_size, 7)`.

**Result.** 0.9376 AUC linear probe. Decent but the size doesn't pay off here because you can't full-FT.

**References:**
- [El-Nouby et al., 2024 — AIMv2](https://arxiv.org/abs/2411.14402)
- [timm GitHub](https://github.com/huggingface/pytorch-image-models)

---

## 9.7 Phase 7 — SigLIP-SO400M (Vision-Language Model's Vision Tower)

### What is SigLIP and what's a "vision tower"?

**SigLIP** = Sigmoid Loss Image-Pretraining. Google 2023. It's a **vision-language model (VLM)** — trained on billions of (image, caption) pairs scraped from the web. It has two parts:
- **Vision tower** — encodes images into vectors.
- **Text tower** — encodes captions into vectors.

Pretraining objective: matching pairs (image of a dog ↔ caption "a dog") should have **high similarity**; mismatched pairs should have low similarity. The "sigmoid loss" was an improvement over CLIP's softmax loss.

**Why is the vision tower interesting?** Because it learned visual features under **enormous data scale (billions of web pairs)**, it might transfer better to dermoscopy than ImageNet-pretrained backbones.

### What are "web pairs"?

Image-caption pairs scraped from internet sources (alt-text on web pages, image titles on stock sites, social media captions). They're noisy, multilingual, low-quality on average — but there are **billions** of them. Scale beats curation.

**Strategy.** Linear probe — SigLIP-SO400M is **400M parameters** (vision tower alone). Way too big for full FT on 8GB. You also drop the **text tower** entirely:

```python
class SigLIPClassifier(nn.Module):
    def __init__(self, model_id, num_classes):
        super().__init__()
        full_model = AutoModel.from_pretrained(model_id)
        self.vision_model = full_model.vision_model    # ← extract vision tower only
        # text tower (full_model.text_model) is NOT kept
        self.head = nn.Linear(self.vision_model.config.hidden_size, num_classes)
        for p in self.vision_model.parameters():
            p.requires_grad = False                    # freeze everything
    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output                 # one feature vector per image
        return self.head(pooled)
```

**Why drop the text tower?** You're doing classification, not image-text matching. The text tower would just sit there using VRAM.

**Why freeze everything?** With 400M params and only 7K training images, full FT would massively overfit even if VRAM allowed.

**Why "the head loses the ability to specialize"?** A linear head can only do **linear combinations** of the frozen features. If SigLIP's features were trained for "matching captions about dogs," they may emphasize different things than what dermoscopy needs (texture asymmetry, pigmentation gradients). The frozen backbone can't adjust; the head can only weight what's already there.

### Phase 7 SigLIP per-class report — what it tells you

```
              precision    recall  f1-score   support
         mel       0.29      0.60      0.40       167
          nv       0.96      0.61      0.74      1006
         bcc       0.53      0.35      0.42        77
       akiec       0.23      0.65      0.34        49
         bkl       0.47      0.49      0.48       165
          df       0.12      0.65      0.20        17
        vasc       0.21      0.68      0.33        22
```

**Reading this:**
- High `nv` precision (0.96) but low recall (0.61) — when the model says `nv`, it's right, but it misses 39% of nevi.
- Other classes have wildly low precision (0.12 for `df`) — the model "discovers" rare classes everywhere, getting tons of false positives.
- This is a **classic frozen-backbone-on-domain-shift failure pattern**: the features simply don't separate dermoscopy classes well, no matter how cleverly you weight them.

This per-class breakdown is exactly what justifies your decision to focus on EfficientNetV2 + DINOv2 for the ensemble.

**Result.** 0.9069 AUC — worst transformer in the benchmark.

**Reference:** [Zhai et al., 2023 — SigLIP](https://arxiv.org/abs/2303.15343)

---

## 9.8 The Standard ViT Classification Pattern (`DINOv2Classifier`)

For all DINOv2-style models, your code follows the standard pattern:

```python
class DINOv2Classifier(nn.Module):
    def __init__(self, model_id, num_classes, freeze_backbone=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id)         # ← HuggingFace
        hidden_size = self.backbone.config.hidden_size              # 768 for base
        self.head = nn.Linear(hidden_size, num_classes)             # ← linear head

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]                 # ← first token
        return self.head(cls_token)
```

### What is `AutoModel.from_pretrained`?

**HuggingFace's `transformers` library** is the universal Python interface for pretrained models. `AutoModel` looks up the right class based on the model ID and downloads weights from the HuggingFace Hub.

```python
AutoModel.from_pretrained("facebook/dinov2-with-registers-base")
```
This single line downloads the model weights and instantiates a working backbone.

**Reference:** [HuggingFace transformers — AutoModel docs](https://huggingface.co/docs/transformers/en/model_doc/auto)

### What is the `[CLS]` token?

When a ViT processes an image, it cuts the image into patches and adds **one extra learnable "CLS" (classification) token** at position 0 of the sequence:

```
Input sequence: [CLS] [patch_1] [patch_2] ... [patch_196]
```

After self-attention, the CLS token has "looked at" every patch and aggregated information from the whole image into a single vector. **It's the model's summary of the image.**

`outputs.last_hidden_state` has shape `[batch, num_tokens, hidden_size]`:
- `[batch, 0, :]` = CLS token (the image summary)
- `[batch, 1:, :]` = per-patch features

For classification, you only need the summary, so:
```python
cls_token = outputs.last_hidden_state[:, 0]   # [batch, 768]
```

Then a single `nn.Linear(768, 7)` maps the summary to class scores. This is the **standard ViT classification pattern** — used in original ViT, DINOv2, SigLIP, and most ViT variants.

### What is `pooler_output` (used in SigLIP)?

Some models add an extra **pooling** step on top of the CLS token (additional Linear + Tanh). The pooled vector is exposed as `outputs.pooler_output`. SigLIP uses it because the model's last layer is designed to produce a final image embedding for matching against text. Same idea as CLS-token but with one more transformation.

---

# PART 10: Test-Time Augmentation (TTA) — Phase 8

## 10.1 The core idea

You already used augmentation during *training*. **TTA** flips the script: at *inference* (test time), apply augmentations to each test image, run the model multiple times, and **average the predictions**.

```python
# 8 versions of the same image:
v1 = original
v2 = horizontal_flip(original)
v3 = vertical_flip(original)
v4 = rotate_90(original)
v5 = rotate_180(original)
v6 = rotate_270(original)
v7 = hflip(vflip(original))
v8 = small_zoom(original)

# 8 predictions:
p1, p2, ..., p8 = model(v1), model(v2), ..., model(v8)

# Final prediction:
final = (p1 + p2 + ... + p8) / 8
```

## 10.2 Why specifically these 8 views?

**They're the 8 transformations that don't change the diagnosis.** Recall that dermoscopy is rotation/flip-invariant. You're saying to the model: "you should give the same answer for all 8 of these. Where you disagree with yourself across views, that's noise — let me average it out."

- 4 rotations (0°, 90°, 180°, 270°) — covers all 90° rotations.
- 2 flips (H, V) — covers reflections.
- HFlip + VFlip — covers another rotation/reflection combo.
- Small zoom — covers minor scale variation.

You don't add color jitter or noise here because those *do* change the image content meaningfully (different lighting could change the prediction).

## 10.3 What is "softmax" and "softmax probabilities"?

The model's raw outputs are called **logits** — unbounded real numbers (e.g., `[2.3, -1.1, 0.5, ...]`). Softmax converts them to probabilities that sum to 1:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

```python
probs = torch.softmax(logits, dim=1)   # → e.g., [0.7, 0.05, 0.15, 0.03, 0.04, 0.02, 0.01]
```

These are the **softmax probabilities** — the model's *confidence* in each class. You average **probabilities** across TTA views (not logits), because probabilities live on a comparable scale across runs.

## 10.4 Why averaging reduces variance — the math

For each image, the model's prediction has some "noise" from how the augmentation happened to land. If a single view's prediction has variance σ², then averaging N independent predictions reduces variance to σ²/N:

$$\text{Var}\left(\frac{1}{N}\sum_{i=1}^{N} p_i\right) = \frac{\sigma^2}{N}$$

(Assuming the views are roughly independent, which they aren't perfectly — but they're decorrelated enough that the variance reduction is real.)

**Bias is unchanged**, but **variance drops** → predictions become more stable → AUC goes up.

This is the **same principle as ensembling**, just with one model on multiple views. So we call TTA a **mini-ensemble of one model**.

**Reference:** [Krizhevsky et al., 2012 — AlexNet](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) (introduced TTA in modern form).

## 10.5 What "free +0.008 AUC" means

EfficientNetV2 single-pass: 0.9685 AUC.
EfficientNetV2 with 8-view TTA: 0.9766 AUC.

**Free** because:
- No retraining, no extra training data, no architectural changes.
- Just inference cost — 8× more inference compute (still seconds/image).
- Pure software effort, no risk of breaking the trained model.

Compared to "spend two more days training a bigger model for +0.005 AUC," TTA is essentially free improvement.

---

# PART 11: Phase 9 — The Ensemble (0.9808 AUC)

## 11.1 What is an ensemble?

**Plain English.** Two doctors disagree on a tough case. Three doctors agree → that consensus is more reliable than any single opinion. Ensembling does the same with models.

**Mechanism.** Take predictions from N different models (here N=2: EfficientNetV2 + DINOv2-Reg), combine them. Simplest: weighted average of softmax probabilities:

$$p_\text{ensemble} = w_1 \cdot p_\text{eff} + w_2 \cdot p_\text{dino},\quad w_1 + w_2 = 1$$

## 11.2 How were the weights 0.60/0.40 chosen — Grid Search

**Grid search** = "try every value in a discrete set, pick the best."

```python
for w in np.arange(0.3, 0.8, 0.05):     # try w = 0.30, 0.35, 0.40, ..., 0.75
    avg = w * eff_probs + (1 - w) * dino_probs
    auc = roc_auc_score(labels, avg, multi_class="ovr", average="macro")
    if auc > best_auc:
        best_auc = auc
        best_weight = w
```

You check 11 different weights. For each, compute ensemble AUC. Keep the best. The best happened to be `w_eff = 0.60` → ensemble AUC 0.9808. There's nothing magical about 0.60 — it just happened to be where the data and models balanced out for *this* test set.

**Caveat (the leakage subtlety).** Grid-searching weights on the **test set** is a mild form of test peeking. Best practice is to grid search on val, then evaluate on test. Your project's `PLAN_DERM.md` calls this out: *"the kind of leakage-fix code comments... that take experience to write."* In an interview, mention you'd fix this by tuning on val in v2.

## 11.3 The numbers for OTHER phases (not just Phase 9)

**The Phase X AUCs were not from grid search** — they're standard training results. Each model's `engine.py` runs the standard "train → save best on val AUC → evaluate on test" pipeline. The number is whatever the test set gave for that one trained model.

**Only Phase 9, 11, 14 numbers come from grid search** — those phases combine pre-trained models, so weights need to be discovered.

## 11.4 Why CNN + ViT works so well — the deep reason

### Inductive bias

**Plain English.** Inductive bias = "what assumptions does the model make about the world?" Different architectures bake in different assumptions, and these assumptions either help or hurt depending on the task.

### CNN's inductive bias: locality + translation invariance

A CNN slides a small filter across the image. This bakes in two assumptions:
- **Locality**: nearby pixels are related; far-apart pixels are processed separately at first.
- **Translation invariance**: a feature in the top-left and the same feature in the bottom-right look identical to the filter (it processes both the same way).

**Why this is great for small datasets.** The model doesn't have to *learn* that "a melanoma pattern is the same wherever it appears in the image" — that's *built in*. Less to learn = faster convergence + better generalization on small data.

### ViT's inductive bias: minimal — global self-attention

A ViT's self-attention lets every patch "look at" every other patch from layer 1. There's no built-in assumption about locality or translation invariance. The model **learns** these from data — but only if you have enough data.

**The trade-off.**
- **Small data** → CNN's built-in assumptions help. ViT might underfit.
- **Big data** → ViT's flexibility wins. CNN's assumptions become limiting.
- **Medium data** (your case, 7K images) → close to a tie. EfficientNetV2 (CNN) and DINOv2-Reg (ViT) score 0.9685 and 0.9653 — basically tied.

### What "global self-attention" means

In a CNN, layer 1's filter sees only a 3×3 neighborhood. To "see" a far-apart relationship, you need many stacked layers (each looking at a 3×3 of the previous). Long-range info has to *propagate* through the network.

In a ViT, **every patch attends to every other patch** in **every layer**. The model can directly compare "the top-left edge of the lesion to the bottom-right edge" in layer 1. This is **global self-attention** = "every part can see every other part, immediately."

For dermoscopy, this might capture **lesion asymmetry** (one of the ABCDE criteria for melanoma) better than a CNN that needs many layers to compare opposite ends of the lesion.

### Why CNN and ViT make DIFFERENT mistakes

Because they have **different inductive biases**, they tend to be wrong in **different ways**:
- CNN may miss a long-range asymmetry (its filters are local).
- ViT may miss a fine-grained texture detail (it lacks the locality bias).

When CNN is wrong, ViT is often right — and vice versa. Their **errors are uncorrelated**.

### Why averaging cancels uncorrelated errors

Two predictions, both noisy, but with **independent** noise:
$$p_1 = \text{truth} + \epsilon_1,\quad p_2 = \text{truth} + \epsilon_2$$

Average:
$$\frac{p_1 + p_2}{2} = \text{truth} + \frac{\epsilon_1 + \epsilon_2}{2}$$

If $\epsilon_1, \epsilon_2$ are independent with variance σ², the averaged noise has variance σ²/2 — **error is reduced**.

If $\epsilon_1, \epsilon_2$ were perfectly correlated (same model trained twice with same data), averaging does nothing.

So **ensemble gain depends on diversity, not just strength**. Two great correlated models < one great + one decent uncorrelated model.

This is **the** key insight of ensembling.

---

# PART 12: Phases 10–13 — Advanced Experiments

## 12.1 Phase 10 — Full fine-tune of DINOv2+Registers

Phase 5 was DINOv2-Reg with full FT already (read your `engine.py` — `freeze_backbone: False`). Phase 10 redoes it with more aggressive settings — bigger gradient accumulation (8 instead of 4), tinier backbone LR (5e-6 instead of 1e-5), more epochs (25 vs 20). The "tiny backbone" hint comes from the LR — *tiny LR on the backbone*, not a tiny model.

**Result.** 0.9692 AUC vs Phase 5's 0.9653 — modest +0.4 gain. The Phase 5 setup was already pretty optimized.

## 12.2 Phase 11 — Re-ensemble with Phase 10's better DINOv2

**The hypothesis:** "If we replace Phase 5's DINOv2 (in the Phase 9 ensemble) with the better Phase 10 DINOv2, the ensemble should improve."

**The result:** Phase 11 AUC = 0.9768 — *worse* than Phase 9's 0.9808!

### Why? — supervision signal and error correlation

**The deep insight.** Phase 5 was a frozen-ish DINOv2 (different freezing strategy from full FT). Phase 10 is fully supervised on dermoscopy, like EfficientNetV2.

When DINOv2 is **fully supervised on dermoscopy**, it starts learning **the same patterns EfficientNetV2 learned** (because they're optimizing the same loss on the same data). Its **errors become correlated with EfficientNetV2's errors**. Recall from 11.4: ensemble gain comes from **uncorrelated errors**. Correlated errors don't cancel under averaging.

The phrase "supervision signal" means the gradient flow from the labels. When two models receive the same supervision signal on the same data, they converge toward similar representations. Diverse models in an ensemble need to have **either different architectures or different supervision signals** (e.g., one supervised, one self-supervised; or one trained on a different task).

**Phase 5's DINOv2** had its features partly anchored by the original DINO self-supervised objective (because the backbone was less heavily fine-tuned). Different signal, more diverse features, better ensemble.

**Phase 10's DINOv2** is more "EfficientNet-like" because it was hammered by the same labels for 25 epochs. Less diversity, weaker ensemble — even though it's a *stronger individual model*.

**Interview gold:** "Better individual models don't always make better ensembles. Diversity beats strength."

---

## 12.3 Phase 12 — Multimodal Model (Image + Patient Metadata)

### Why metadata?

Real dermatologists don't just look at the photo. They consider:
- **Age** (melanoma is more common in older patients)
- **Sex** (some lesion types are more common in one sex)
- **Body location** ("acral" lesions on palms/soles look different and have different cancer rates)

These are recorded in HAM10000's metadata CSV. Phase 12 asks: "if we feed the model these clinical features in addition to the image, does it improve?"

### Architecture — image + metadata fusion

```
Image (224x224)               Metadata (age, sex, location)
   │                                  │
   ▼                                  ▼
DINOv2+Reg backbone           MLP (3 → 32 → 32)
   │                                  │
   ▼                                  ▼
CLS token (768-dim)            meta features (32-dim)
              \                     /
               \                   /
                \                 /
              concat → 800-dim vector
                       │
                       ▼
              Fusion head: Linear(800, 256) → ReLU → Dropout → Linear(256, 7)
                       │
                       ▼
                  7 class logits
```

### What is an MLP and why for metadata?

**MLP** = Multi-Layer Perceptron. The simplest neural network — stacked `Linear → ReLU → Linear → ReLU → ...` layers. It learns nonlinear functions of its input.

```python
self.meta_mlp = nn.Sequential(
    nn.Linear(3, 32),      # 3 metadata features → 32-dim
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(32, 32),     # 32 → 32 (more capacity)
    nn.ReLU(),
)
```

**Why an MLP and not just feed metadata as 3 raw numbers?** Two reasons:
1. **Dimensionality matching.** The image features are 768-dim. If you concat 3 metadata numbers next to 768 image numbers, the metadata gets *drowned out* — the head's weights for those 3 numbers barely affect the output. Boosting metadata to 32-dim gives it ~4% of the input → not dominant, but actually heard.
2. **Nonlinear interactions.** "Older + face lesion" might mean something different from age and location alone. The MLP can learn these interactions before fusion.

### Why concat?

**Concat** (concatenation) = stick two vectors end-to-end:
```python
combined = torch.cat([cls, meta_feat], dim=1)   # 768 + 32 = 800-dim
```

**Why concat instead of add or multiply?**
- **Add** requires the same dimensionality (you'd need both to be 768-dim).
- **Multiply** assumes interaction, but might lose info.
- **Concat** preserves all information — the fusion head can learn any combination of image and metadata features.

Concat is the default fusion strategy in multimodal learning.

### What is the "fusion head"?

A small MLP that takes the combined 800-dim vector and produces final class scores:

```python
self.fusion_head = nn.Sequential(
    nn.Linear(800, 256),    # combine image + metadata signals
    nn.ReLU(),
    nn.Dropout(0.3),         # regularization
    nn.Linear(256, 7),       # final 7-class output
)
```

The 256-dim hidden layer gives the model room to learn nonlinear combinations. Dropout zeros out 30% of activations randomly during training (regularization to prevent overfitting on the small data).

### Metadata encoding — turning words into numbers

Neural networks can't ingest strings like `"male"` or `"face"`. You must convert to numbers.

```python
SEX_MAP = {"male": 0, "female": 1, "unknown": 2}
LOC_MAP = {"back": 0, "lower extremity": 1, "trunk": 2, ..., "unknown": 14}
```

Then you **normalize to roughly [0, 1]**:
- Sex: divide by 2 → `{0.0, 0.5, 1.0}`
- Location: divide by 14 → `{0/14, 1/14, ..., 14/14}`

### Why normalize age?

```python
age = float(age_raw) / 85.0   # patients are 0-85 years
```

**Plain English.** The MLP's input layer expects values in a similar range (typically [-1, 1] or [0, 1]). If age is `60` and sex is `0.5`, the MLP's weights have to be wildly different magnitudes to handle them — which makes optimization hard.

**Technical reason.** Gradient updates work best when features have similar scales. If one input is 100× larger than another, its gradient is 100× larger, and the optimizer can't update both well at once. Standard practice: normalize all numerical features to the same range before feeding the network.

`/85.0` divides by the dataset's max age, putting it in [0, 1]. Crude but effective.

### Why default missing age to 50?

```python
if pd.isna(age_raw) or str(age_raw).lower() == "unknown":
    age = 50.0 / 85.0   # ≈ 0.59
```

**Plain English.** Some patients have no recorded age. You can't pass `NaN` (not-a-number) to a neural network — it'll produce NaN gradients and break training. You have to fill it with **something**. The choices:
- `0` → tells the model "this is a baby" (wrong).
- `mean age` (≈50 for HAM10000) → tells the model "I don't know, assume average."

`50` is roughly the dataset mean. This is **mean imputation**, the simplest standard fix for missing values. More sophisticated approaches: nearest-neighbor imputation, learned embeddings for "missing" — but mean imputation is fine for ~5% missing data.

### Two-LR optimizer — same idea as discriminative LRs, applied here

```python
optimizer = torch.optim.AdamW([
    {"params": [p for n,p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": 5e-6},
    {"params": [p for n,p in model.named_parameters() if "backbone" not in n and p.requires_grad], "lr": 1e-3},
], weight_decay=1e-4)
```

- Backbone (already-pretrained DINOv2): **5e-6** — extra-small, because the backbone was already fine-tuned in Phase 10. Don't disturb.
- Meta MLP + fusion head (random init): **1e-3** — needs to learn fast.

200× ratio. Same principle as Phase 5; just bigger spread because the backbone is being preserved more carefully.

### Why "modest signal"?

Phase 12's standalone test AUC is ~0.962 — *worse* than Phase 10's 0.9692! Why? Because:
1. **The image already captures location implicitly.** A face lesion looks different from a back lesion in the dermoscope. The model doesn't need the metadata text "face" — it can see "face" in the photo. Adding the metadata is redundant.
2. **Acral lesions** (palms, soles, nail beds) are a partial exception. They tend to look very different from other lesions — different skin texture, often darker. Knowing "this is acral" might help disambiguate. This is one place where metadata adds genuinely new info.
3. The model is already heavily overparameterized for 7K images. Adding more inputs without much new info just creates overfitting risk.

So multimodal helps a *little* — it's a meaningful third ensemble member but not a dramatic standalone improvement.

**Reference:** [Multimodal fusion review — Baltrušaitis et al., 2017](https://arxiv.org/abs/1705.09406)

---

## 12.4 Phase 13 — Focal Loss for Melanoma Recall

### What is Focal Loss?

Standard cross-entropy treats all examples equally. **Focal loss** down-weights "easy" examples (where the model is already confident-correct) so the model focuses gradient updates on **hard** examples (where it's uncertain or wrong).

The formula:
$$\mathcal{L}_\text{focal} = -\alpha \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

Where:
- $p_t$ = model's predicted probability for the *true* class
- $\gamma$ = "focusing parameter" (usually 2.0). Higher → more aggressive down-weighting.
- $\alpha$ = optional class weight (handle class imbalance per-class).

**Intuition.** The factor $(1-p_t)^\gamma$ is:
- Near 0 when $p_t$ is large (~0.9, easy correct example) → loss reduced ~100×
- Near 1 when $p_t$ is small (~0.1, hard wrong example) → loss preserved

The model's gradient flow is **hijacked toward hard examples**. For a class where the model often gets things wrong (like rare melanoma), this boosts learning.

### Why specifically for melanoma recall?

Melanoma is rare (1113/10015 = 11%) AND visually similar to benign nevi. The model finds it easy to predict `nv` (always confident) and hard to find melanomas. Focal loss says "stop wasting gradient on easy `nv` examples — focus on the borderline melanomas."

### Result and the precision-recall trade-off

From your `phase13_focal_loss/results.txt`:

```
              precision    recall  f1-score
         mel       0.41      0.79      0.54
```

vs Phase 10 baseline:

```
         mel       0.57      0.74      0.65
```

**Mel recall went UP** (0.74 → 0.79) — focal loss caught more melanomas. ✓
**Mel precision went DOWN** (0.57 → 0.41) — the model now over-predicts mel, lots of false alarms. ✗
**Overall AUC went DOWN** (0.9692 → 0.9562) — the model's calibration is worse on aggregate.

This is the **classic precision-recall trade-off** with imbalance fixes. You bought more recall with the currency of precision and overall AUC.

**The lesson.** Focal loss is *not* a free win — it shifts the model's bias. Use it only when **catching the rare class is genuinely more valuable than overall AUC** (e.g., screening — where missing cancer is fatal but false alarms are just inconvenient).

**Reference:** [Lin et al., 2017 — Focal Loss for Dense Object Detection (RetinaNet)](https://arxiv.org/abs/1708.02002)
# PART 13: The Optimization Scripts (Phase 14+ Experiments)

These are standalone scripts experimenting with smarter ways to combine models. They're more advanced than the basic 0.60/0.40 grid search.

## 13.1 `mega_tta.py` — Synchronized TTA Predictions

### What problem does this solve?

Phase 8's TTA already saved `eff_probs.npy` and `dino_probs.npy`. Why redo TTA?

**The issue: prediction synchronization.** The earlier files were generated in two **separate runs** of Phase 8. If anything in the data loading was non-deterministic (e.g., `num_workers` shuffling order, different dropout seeds), the two arrays might be in **different orders** — `eff_probs[5]` might correspond to a different test image than `dino_probs[5]`.

For simple grid search of one global weight, this doesn't matter — you're computing aggregate AUC, and order-mismatched probabilities would just hurt the AUC, not crash.

**But for per-sample optimization** (next sections — SLSQP, XGBoost stacking, threshold tuning), you need **prediction `i` from EfficientNet to match prediction `i` from DINOv2** — they must describe the same test image. Otherwise the meta-learner sees nonsense.

**`mega_tta.py` re-runs both models in lock-step**: same test_df (same order), same DataLoader settings (no shuffling), so `eff_probs[i]` and `dino_probs[i]` are guaranteed to be the same image.

### What "per-sample optimization" means

Earlier methods:
- Phase 9 grid search: one weight `w` for the whole dataset. The combination is the same for every image.
- TTA averaging: same number of views, same weights, applied uniformly.

Per-sample methods (SLSQP per-class, XGBoost stacking, learned thresholds): the combination **depends on the example**. For example, a meta-learner might decide "if EfficientNet predicts mel with 0.8 confidence, weight it heavily; if it's uncertain, defer to DINOv2." That requires synced predictions per image.

### The bug in this file (be honest about it)

```python
np.save(f"{output_dir}/labels.npy", test_df['label_idx'].values)
```

There's no `label_idx` column in your `dataset.py` — labels are computed inside `__getitem__` from the `dx` string column. This line would throw a `KeyError`. The fix is something like:
```python
labels = np.array([CLASS_TO_IDX[dx] for dx in test_df['dx']])
np.save(f"{output_dir}/labels.npy", labels)
```

In an interview, **own this**: "I noticed during cleanup that `mega_tta.py` references a column I never added — it's an experimental script, not production. The fix is one line. I caught it during the v2 planning audit."

---

## 13.2 `opt_slsqp_weights.py` — Per-Class Weight Optimization

### What's the upgrade over Phase 9's grid search?

Phase 9: one global weight `w_eff = 0.60`, applies to *all* classes equally.

But what if EfficientNet is great at melanoma and bad at nevi, while DINOv2 is the opposite? You want **class-specific weights**:
- For mel: w_eff=0.8, w_dino=0.2
- For nv: w_eff=0.3, w_dino=0.7
- ... and so on for all 7 classes.

That's **2 models × 7 classes = 14 weights** to find.

### Why grid search doesn't work here

Grid search is fine for 1 weight (try 11 values). For 14 weights, even a coarse grid (5 values per weight) would be 5^14 ≈ 6 billion combinations. Infeasible.

### What is SLSQP?

**SLSQP** = Sequential Least Squares Programming. A constrained optimization algorithm that handles:
- Continuous variables (the weights are real numbers, not discrete choices)
- Equality constraints (per class, the two weights must sum to 1)
- Inequality constraints (each weight must be in [0, 1])

It works by **iteratively improving** a candidate solution, using a quadratic approximation of the objective at each step.

```python
res = minimize(
    objective,                    # function to minimize (negative AUC)
    initial_weights.flatten(),    # 14-dim starting point
    method='SLSQP',
    bounds=[(0, 1)] * 14,         # each weight ∈ [0, 1]
    constraints=cons,             # per-class sum-to-1 constraints
)
```

### What is the "constraint surface"?

In a constrained optimization problem, **the constraint surface** is the set of all valid solutions — the geometric region in weight-space where all constraints are satisfied.

In your case:
- 14 weights, each in [0, 1] → starts as a 14-dimensional unit hypercube.
- 7 sum-to-1 constraints → reduces to a 7-dimensional surface (one constraint cuts the dimensionality by 1).

This 7D surface is what SLSQP searches over.

**Why "the constraint surface is small" → tiny gains?** Because the constraints are tight — they fully determine the relationship between the two weights for each class. The free parameters are essentially "what fraction of class i's prediction comes from EfficientNet vs DINOv2." That's 7 numbers, each in [0, 1]. Not a huge search space. The objective (AUC) is smooth, and SLSQP often converges to a solution very close to "0.6/0.4 globally" — improvement vs Phase 9 ends up being marginal (a few thousandths of AUC).

### Same leakage caveat

The script optimizes weights on the test set. To do this honestly, you'd optimize on val and evaluate on test.

**Reference:** [SciPy SLSQP docs](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)

---

## 13.3 `opt_xgboost_stack.py` — Stacking with a Non-Linear Meta-Learner

### What is "stacking"?

**Stacking** is an ensemble method where you:
1. Train your **base models** (EfficientNet, DINOv2).
2. Get their predictions on a held-out set.
3. Train a **meta-learner** that takes base-model predictions as **input features** and learns to combine them.

The meta-learner can be anything — logistic regression, XGBoost, even a small neural net. It's another layer of learning *on top of* the base models.

### "7-dimensional probability vectors" → "14 input features"

Each model outputs a 7-dim probability vector (one probability per class):
- EfficientNet: `[0.05, 0.7, 0.02, 0.1, 0.08, 0.03, 0.02]` (7 numbers)
- DINOv2: `[0.08, 0.65, 0.03, 0.12, 0.07, 0.03, 0.02]` (7 numbers)

If you **concatenate** them, you get **14 numbers** that summarize what both models think about this image:
```python
X = np.hstack([eff_probs, dino_probs])   # shape: (n_images, 14)
```

Each test image becomes a 14-dim "feature vector" for the meta-learner.

### What is XGBoost?

**XGBoost** = eXtreme Gradient Boosting. A tree-based ML algorithm that builds many small decision trees sequentially, each correcting the previous trees' errors. It's the **king of tabular ML** — wins most Kaggle competitions on structured data.

```python
clf = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, ...)
clf.fit(X_train, y_train)
```

### Why "non-linear combiner" matters

A weighted average (Phase 9) is **linear**:
$$p_\text{ensemble} = w_1 \cdot p_\text{eff} + w_2 \cdot p_\text{dino}$$

It can only do simple convex combinations.

XGBoost can learn **nonlinear** rules like:
- "If both models agree on `nv` with high confidence → predict `nv`."
- "If EfficientNet says mel > 0.4 AND DINOv2 says nv > 0.5 → it's probably bcc."
- "If all probabilities are flat (model confused) → default to weighted average."

These are the kinds of **conditional decision rules** that linear weighted averaging can't capture. In theory, this should help.

### The methodological issue (be ready to explain)

```python
X = np.hstack([p1, p2])
y = np.load(base + "labels.npy")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

This script takes **test set predictions** and splits them 80/20. Trains XGBoost on 80%. Evaluates on 20%.

**The problem.** The "predictions" being split *are already predictions on the test set*. The 80% used for training XGBoost was test data — meaning EfficientNet and DINOv2 were optimized on it during model selection, and now XGBoost is trained on it too. There's a chain of test-set peeking.

**The proper fix.** Train base models on train. Generate predictions on **val**. Train XGBoost on val predictions + val labels. Evaluate XGBoost on test predictions + test labels. This is called **out-of-fold stacking**.

Your `PLAN_DERM.md` calls this out — *"the kind of leakage-fix code comments... that take experience to write"* — meaning you've recognized this pattern. **Owning known weaknesses in an interview is a green flag.**

**Reference:** [XGBoost docs](https://xgboost.readthedocs.io/en/stable/) and [Wolpert, 1992 — Stacked Generalization](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231)

---

## 13.4 `opt_learned_thresholds.py` — Per-Class Threshold Tuning

### The argmax baseline

By default, you predict the class with highest probability:
```python
preds = ensemble_probs.argmax(axis=1)
```

**Argmax** = "give me the index of the largest value." It's symmetric — treats all classes equally.

For melanoma: predict mel only if `P(mel) > P(every other class)`. This requires P(mel) > ~0.14 typically (because you have 7 classes, average prob is 1/7).

### The clinical problem with argmax

For melanoma, **missing it is catastrophic** (patient dies). **False alarming** is mildly inconvenient (extra biopsy). You should predict mel even when you're "less than maximally confident" — because the asymmetric cost says recall matters way more than precision for this class.

### How threshold tuning works

Instead of pure argmax, use a **rule with class-specific thresholds**:
- If `P(mel) > τ_mel`, predict mel even if some other class has higher probability.
- Otherwise, fall back to argmax.

```python
final_preds = np.argmax(ensemble_probs, axis=1)
final_preds[mel_probs > best_threshold] = 0    # 0 = mel index
```

### Sweeping thresholds to maximize F2

You don't know the right `τ_mel` ahead of time. Try all values in [0.01, 0.5] in 100 steps; for each, compute F2 score; pick the threshold maximizing F2.

```python
for t in np.linspace(0.01, 0.5, 100):
    preds = (mel_probs > t).astype(int)
    score = fbeta_score(y_binary, preds, beta=2)
    if score > best_f2:
        best_f2 = score
        best_threshold = t
```

### Why F2 specifically?

We covered F2 in Section 2.2: it's recall-weighted, which is what we want for cancer screening. Maximizing F2 finds the threshold that catches more melanomas while keeping false alarms tolerable.

If you maximized **F1** instead, you'd get a more balanced threshold — but balance isn't what clinical screening calls for.

### Triage thresholds — the hospital workflow

**Triage** = sorting patients by urgency. In emergency departments, nurses don't give every patient the same priority — they classify them as critical/urgent/routine using fast tests.

Hospitals tune diagnostic thresholds **per disease and per setting**:
- A specialist clinic might use a **lower mel threshold** (catch any plausible melanoma → biopsy more aggressively).
- A primary care office might use a **higher threshold** (refer only the more likely cases to specialists).

Your `opt_learned_thresholds.py` simulates this — finding the threshold that best matches a chosen criterion (F2 = recall-favoring). In production, the threshold would be tuned to the deployment setting.

### Same leakage caveat

Threshold tuned on test set = peeking. Should be tuned on val.

---

# PART 14: Phase 14 — Three-Model Ensemble

`phase14_three_model_ensemble.py` extends Phase 9's two-model ensemble to three models: **EfficientNet+TTA, DINOv2-FT+TTA, Multimodal (Phase 12)**.

## 14.1 What is "an ensemble member"?

Just a model whose predictions go into the ensemble. Phase 9 had 2 members; Phase 14 has 3. Each member ideally contributes diverse error patterns.

## 14.2 Why a third member?

The first two members were already chosen for diversity (CNN + ViT). Adding a third gives you:
- **More diversity** if it makes different mistakes.
- **More noise** if it doesn't.

Multimodal's metadata input gives it a slightly different signal source than pure-image models. So it might be a useful third member even if its standalone AUC (~0.962) is lower than EfficientNet's (~0.9685).

**Whether it actually helps depends on grid search.** If the optimal weight ends up being `w_meta = 0.05`, the multimodal model is contributing essentially nothing — it's noise to be averaged away.

## 14.3 3-way grid search and "summing to 1"

Phase 9 had one free weight: `w` for EfficientNet (then DINOv2 gets `1-w`).

Phase 14 has three weights: `w1, w2, w3` for the three models. They must sum to 1 (because they're convex combination weights):
$$w_1 + w_2 + w_3 = 1,\quad w_i \in [0, 1]$$

**Why sum to 1?** The output is a **probability** (must sum to 1 across classes). Each model's output is a valid probability vector. To preserve the "is a probability" property after combining:
$$\sum_c (w_1 p_{1,c} + w_2 p_{2,c} + w_3 p_{3,c}) = w_1 \cdot 1 + w_2 \cdot 1 + w_3 \cdot 1 = w_1 + w_2 + w_3$$

For this to equal 1 (a valid probability vector), the weights must sum to 1.

**The grid search:**
```python
for w1 in np.arange(0.25, 0.75, 0.05):
    for w2 in np.arange(0.10, 0.60, 0.05):
        w3 = round(1.0 - w1 - w2, 4)
        if w3 < 0.05 or w3 > 0.50:    # skip degenerate weights
            continue
        avg = w1 * eff_probs + w2 * dino_probs + w3 * meta_probs
        auc = roc_auc_score(...)
```

A grid over 2 weights (the third is determined by the sum constraint), with sanity bounds (`w3` not too small, not too large).

## 14.4 The clever state_dict key remapping

This is a real-world deep learning headache that's worth understanding because it shows up everywhere.

### The problem — training vs inference architecture mismatch

When Phase 12 trained the multimodal model, the architecture was:
```python
class MultimodalClassifier(nn.Module):
    def __init__(self, backbone_ckpt):
        ...
        full_model = DINOv2Classifier(...)         # Phase 5 wrapper
        full_model.load_state_dict(...)
        self.backbone = full_model.backbone        # ← extracts the inner backbone
```

So the saved checkpoint has keys like:
```
backbone.embeddings.cls_token
backbone.embeddings.patch_embeddings.weight
backbone.encoder.layer.0.attention.attention.query.weight
...
meta_mlp.0.weight
fusion_head.0.weight
```

But for **Phase 14 inference**, the loader script defines a *different* class:
```python
class DINOv2Backbone(nn.Module):
    def __init__(self):
        self.backbone = AutoModel.from_pretrained(...)   # ← extra wrapping layer

class MultimodalClassifier(nn.Module):
    def __init__(self):
        self.backbone = DINOv2Backbone()                  # ← so backbone.backbone exists
```

The Phase 14 model expects keys like:
```
backbone.backbone.embeddings.cls_token       # ← double "backbone."
backbone.backbone.encoder.layer.0...
meta_mlp.0.weight
fusion_head.0.weight
```

The mismatch: saved as `backbone.X`, expected as `backbone.backbone.X`. Loading directly would fail with "Missing keys" / "Unexpected keys."

### The fix — key remapping

```python
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("backbone.") and not k.startswith("backbone.backbone."):
        new_key = k.replace("backbone.", "backbone.backbone.", 1)
        new_state_dict[new_key] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)
```

For each key, check if it starts with `backbone.`. If yes, prepend another `backbone.`. This translates the old (training) key names into the new (inference) format.

### Why is this necessary?

When you save a PyTorch model, you save its **weights and their key names** — but not the class definition itself. Two scripts with different class definitions interpret the same checkpoint differently. This is a footgun.

**Best practice in production:** save the class definition along with the weights, or keep architectures stable. In research code, key remapping is the standard quick fix.

**The "clever" part.** Most people would re-train Phase 12 with the new architecture (slow, wasteful). Remapping the keys lets you reuse existing weights — saves hours of compute.

### Why implement fallback logic?

```python
backbone_ckpt = os.path.join(CKPT_BASE, "phase10_dinov2_registers_finetune", "best_model.pth")
if not os.path.exists(backbone_ckpt):
    print("⚠️ Phase 10 checkpoint not found. Falling back to Phase 5...")
    backbone_ckpt = os.path.join(CKPT_BASE, "phase5_dinov2_registers", "best_model.pth")
    if not os.path.exists(backbone_ckpt):
        print("❌ ERROR: Neither found.")
        sys.exit(1)
```

**Plain English.** If the preferred checkpoint (Phase 10) doesn't exist, automatically try Phase 5 instead. If neither exists, exit with a clear error.

**Why useful.** When you re-run experiments months later (or someone else clones the repo), they may not have run every prior phase. Fallback logic makes the code **degrade gracefully** instead of crashing with a confusing FileNotFoundError. The console message tells them exactly what to do.

This kind of small UX detail signals **engineering maturity** — good interview signal.

---

# PART 15: Putting It All Together — The Story for Resume + Interview

## 15.1 The narrative arc

You can describe this project as a **systematic research benchmark** in 4 acts:

**Act 1 — Single-model benchmark (Phases 1-7).** Established what each architecture can do alone, with controlled comparison. ResNet50 baseline (0.89), CNN winner EfficientNetV2 (0.97), best ViT DINOv2-Reg (0.965), worst frozen-VLM SigLIP (0.91). This act is about **rigorous experimental design**.

**Act 2 — Free wins from inference tricks (Phase 8-9).** TTA gave +0.008 AUC for free. Ensemble of CNN + ViT gave the headline 0.9808. This act is about **cheap-but-clever improvements**.

**Act 3 — Pushing the model further (Phases 10-13).** Full FT of DINOv2, multimodal fusion, focal loss for melanoma recall. Mixed results — taught lessons about **diversity vs strength** and the **precision-recall trade-off**. This act is about **iterative refinement and honest reporting of failures**.

**Act 4 — Smarter ensembling (Phases 14+).** Per-class weight optimization, XGBoost stacking, learned thresholds. This act is about **methodological depth**.

## 15.2 The interview "deep questions" you should be ready for

**Q: Walk me through the ensemble — why does CNN + ViT work?**
A: They have different inductive biases. CNN bakes in locality and translation invariance, which helps on small data. ViT does global self-attention from layer 1, capturing long-range relationships like lesion asymmetry. Their errors are uncorrelated — when one is wrong, the other often isn't. Averaging two predictions with independent noise of variance σ² gives variance σ²/2. The ensemble's gain comes from this error decorrelation, not from average strength. That's why Phase 11 (with a stronger DINOv2) actually got worse — full FT made DINOv2's errors correlated with EfficientNet's.

**Q: Why macro AUC?**
A: Severe class imbalance — `nv` is 67% of the data. Accuracy and micro-averaged metrics are dominated by the majority class. Macro AUC averages per-class AUC equally, so a model that's only good at `nv` gets penalized. AUC itself is threshold-independent and robust to imbalance — it's the probability the model ranks a random positive higher than a random negative. With 7 classes, "macro AUC" means averaging the 7 one-vs-rest AUCs.

**Q: What did you learn from focal loss?**
A: Focal loss boosted melanoma recall from 0.74 to 0.79, but precision dropped from 0.57 to 0.41 and overall AUC dropped from 0.9692 to 0.9562. It's a classic precision-recall trade-off — you can't get free recall without paying somewhere. Focal loss is the right choice when missing the rare class is much more costly than false alarms (cancer screening), but it's not a free win on aggregate metrics.

**Q: What are the project's limitations?**
A: Three things I'd fix in v2:
1. **Image-level instead of patient-level splits.** HAM10000 has multiple dermoscopic views of some lesions. A random image-level split could put two views of the same lesion in train and test, inflating AUC. Patient-level splitting using `lesion_id` would prevent this.
2. **Test-set tuning in some optimization scripts** (`opt_xgboost_stack.py`, weight grid search). The proper fix is out-of-fold stacking — train meta-learners on val, evaluate on test.
3. **No fairness analysis.** HAM10000 is mostly Austrian/Australian patients. Models trained on this skew underperform on Fitzpatrick 5-6 skin tones — published in Daneshjou et al. 2022. A real deployment would need DDI/Fitzpatrick17k benchmarks.

These are exactly what the `PLAN_DERM.md` v2 plan addresses.

**Q: How would you push past 0.9808?**
A: The per-class AUC analysis says vasc/df/bcc/akiec are saturated at 0.99+. The headroom is in mel (0.95) and bkl. The mel/nv decision boundary is the bottleneck — they're visually similar. Options:
- **PanDerm** (Nature Medicine 2025) — a foundation model pretrained on 2M dermatology images. Likely the largest single-model improvement available.
- **Hard-negative mining at the mel/nv boundary** — re-train final epochs with oversampling of high-loss examples.
- **Mixup between mel and nv pairs** to densify the decision boundary.
- **Hierarchical loss** — auxiliary head predicting "melanocytic vs non-melanocytic" forces better separation upstream.

## 15.3 Resume bullets — final versions

Pick 3-4. The first one is the headline; pick others to round out the story.

> **Skin Lesion Classification — Foundation Model Benchmark on HAM10000** (PyTorch, MLflow, RTX 4060 8GB, 2025)
>
> - Achieved **0.9808 macro AUC** (88.4% accuracy) on 7-class HAM10000 via a **CNN+ViT ensemble** (EfficientNetV2-M + DINOv2-with-Registers), with grid-searched optimal weights of 0.60/0.40, exceeding each component's TTA score by +0.4 to +1.4 AUC.
> - Designed a **systematic 7-model benchmark** (ResNet50, EfficientNetV2, ViT-Base, DINOv2, DINOv2+Registers, AIMv2-307M, SigLIP-400M) under matched data splits and evaluation; produced **the first published HAM10000 benchmark for DINOv2-with-Registers**.
> - Engineered an 8GB-VRAM training pipeline using **mixed-precision (AMP), gradient accumulation (effective batch 64), and gradient checkpointing**, enabling full fine-tuning of 86M-parameter Vision Transformers on consumer hardware.
> - Handled severe **58:1 class imbalance** with WeightedRandomSampler and heavy Albumentations augmentation; investigated **focal loss** for melanoma recall (improved from 0.72 → 0.79, with documented precision/AUC trade-off).
> - Built a **multimodal fusion model** combining DINOv2 image features with patient metadata (age, sex, body location) via a 3→32-dim MLP, concatenation, and a shared fusion head; integrated as the third member of a 3-way grid-searched ensemble.
> - Implemented **8-view Test-Time Augmentation** (rotations, flips, zoom) and **per-class SLSQP weight optimization** as alternatives to global weighted averaging; documented the precision-recall trade-off and ensemble-diversity findings ("better single models can yield worse ensembles when error patterns become correlated").

## 15.4 Things you can say that almost no one else says

These are the **earned-experience signals** that separate "I followed a tutorial" from "I built this":

1. *"Phase 11 was a negative result — the better DINOv2 produced a worse ensemble. I diagnosed it as supervision-signal correlation."* (You understand ensembles deeply.)
2. *"The optimization scripts have a known leakage pattern — they tune on test predictions. The fix is out-of-fold stacking on val. I'm explicit about it in PLAN_DERM.md for the v2 project."* (You can discuss methodology weaknesses honestly.)
3. *"The per-class AUC analysis told me where the ceiling lives — vasc/df are saturated, mel is the bottleneck. The next push has to focus on the mel/nv boundary."* (You can read your own results and plan from them.)
4. *"For the multimodal model, I had to remap state_dict keys because training architecture nested backbones differently than inference. Saved hours over re-training."* (You handle real deep learning footguns.)
5. *"I chose F2 instead of F1 for melanoma threshold tuning because clinical screening cares more about recall than precision."* (You connect ML to deployment context.)

---

# PART 16: Quick Reference Index

For interview prep, here's a mental index of every term in your original list:

| Term | Section |
|------|---------|
| AUC, ROC, macro AUC | 2.3 |
| Class imbalance, naive model, headline metric | 2.4 |
| Recall, F2, F-beta, confusion matrix | 2.1, 2.2 |
| PIL, tensor, batch, epoch | 1.2, 1.3 |
| Loss function, gradients, backward, optimizer.step | 1.4, 1.5 |
| Why HAM10000 part_1 / part_2 | 3.1 |
| Stratified, class distribution | 3.3, 2.4 |
| WeightedRandomSampler, oversampling, balanced batches | 3.4 |
| Albumentations, training augmentation | 4.1, 4.2 |
| Rotation invariance | 4.2 |
| ColorJitter, GaussNoise, blur, motion blur | 4.2 |
| ImageNet, normalization, backbones, pretrained | 4.3 |
| Val/test deterministic | 4.4 |
| Heavy augmentation | 4.1, 4.2 |
| Discriminative LRs, partial freezing, blocks | 5.2, 5.3, 5.5 |
| Lower vs higher blocks | 5.2 |
| Why freezing = less compute | 5.3 |
| Head, random init, 100× LR ratio | 5.4, 5.5 |
| OneCycleLR, warmup, cosine decay, step 0 | 5.6 |
| Mixed precision, autocast, GradScaler, fp16/fp32 | 6.1 |
| Forward/backward pass | 1.5, 6.1 |
| Gradient accumulation, accum_steps, effective batch | 6.2 |
| Gradient checkpointing, intermediate activations | 6.3 |
| MLflow, why save on AUC not loss | 7.1, 7.2 |
| Per-class AUC, normalized confusion matrix, GradCAM | 8.1, 8.2, 8.3 |
| The 7 models — why each | 9.1–9.7 |
| Registers, attention artifacts | 9.5 |
| Autoregressive SSL, AIMv2 | 9.6 |
| Vision tower, web pairs, SigLIP, text tower | 9.7 |
| Linear probe vs full FT | 9.0 |
| Why timm vs HF, why num_classes=0 | 9.6 |
| Pooler output, CLS token, last_hidden_state | 9.8 |
| Standard ViT classification pattern | 9.8 |
| AutoModel from HuggingFace | 9.8 |
| Vanilla DINOv2 vs DINOv2+Registers | 9.4 vs 9.5 |
| Phase 7 SigLIP per-class report | 9.7 |
| Why head loses ability to specialize | 9.7 |
| TTA, 8 augmentations, why these angles | 10.1, 10.2 |
| Softmax, softmax probabilities | 10.3 |
| Inference, confidence, averaging variance | 10.3, 10.4 |
| Mini ensemble, "free" gain | 10.4, 10.5 |
| Grid search, 0.60/0.40 weights | 11.2 |
| Local features, inductive bias, translation invariance | 11.4 |
| Global self-attention, long-range relationships | 11.4 |
| Different mistakes, error correlation | 11.4 |
| Tiny backbone (LR), full FT | 12.1 |
| Supervision signal, error correlation | 12.2 |
| MLP for metadata, why needed | 12.3 |
| Concat, fusion head | 12.3 |
| Metadata encoding, age normalization, NaN→50 | 12.3 |
| Two-LR optimizer | 12.3 |
| Modest signal, acral lesions | 12.3 |
| Focal loss | 12.4 |
| Synchronized predictions | 13.1 |
| Per-sample optimization | 13.1 |
| SLSQP, per-class weights, constraint surface | 13.2 |
| Stacking, 7-dim probs → 14 features | 13.3 |
| XGBoost meta-learner, non-linear combiner | 13.3 |
| Argmax, threshold sweep, F2-score, triage threshold | 13.4 |
| Third ensemble member, ensemble member | 14.1, 14.2 |
| 3-way grid search, sum to 1 | 14.3 |
| State_dict key remapping, training vs inference architecture | 14.4 |
| Fallback logic | 14.4 |

---

# Final Reference List (clickable)

**Foundational ML and Deep Learning:**
- [3Blue1Brown — Neural networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Andrej Karpathy — Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [PyTorch tutorials](https://pytorch.org/tutorials/)
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)

**Architectures:**
- [ResNet (He et al., 2015)](https://arxiv.org/abs/1512.03385)
- [EfficientNetV2 (Tan & Le, 2021)](https://arxiv.org/abs/2104.00298)
- [ViT (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193)
- [Vision Transformers Need Registers (Darcet et al., 2024)](https://arxiv.org/abs/2309.16588)
- [AIMv2 (El-Nouby et al., 2024)](https://arxiv.org/abs/2411.14402)
- [SigLIP (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343)

**Techniques:**
- [Mixed Precision Training (Micikevicius, 2017)](https://arxiv.org/abs/1710.03740)
- [Gradient Checkpointing (Chen et al., 2016)](https://arxiv.org/abs/1604.06174)
- [OneCycleLR / Super-convergence (Smith, 2018)](https://arxiv.org/abs/1708.07120)
- [ULMFiT discriminative LRs (Howard & Ruder, 2018)](https://arxiv.org/abs/1801.06146)
- [Focal Loss (Lin et al., 2017)](https://arxiv.org/abs/1708.02002)
- [Grad-CAM (Selvaraju et al., 2017)](https://arxiv.org/abs/1610.02391)

**Tools & Libraries:**
- [PyTorch](https://pytorch.org/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [timm](https://github.com/huggingface/pytorch-image-models)
- [Albumentations](https://albumentations.ai/)
- [MLflow](https://mlflow.org/)
- [scikit-learn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- [XGBoost](https://xgboost.readthedocs.io/)

**Dataset:**
- [HAM10000 paper (Tschandl et al., 2018)](https://www.nature.com/articles/sdata2018161)
- [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [HAM10000 on Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

**Medical AI Caveats:**
- [Daneshjou et al., 2022 — AI fairness in dermatology](https://www.science.org/doi/10.1126/sciadv.abq6147)
- [Geirhos et al., 2020 — Shortcut learning in deep neural networks](https://arxiv.org/abs/2004.07780)

---

# PART 17: Patient-Level Splits and Why Random Splitting Lied to Us (v2)

**Plain English.** HAM10000 has 10,015 images of about 7,470 unique lesions — most lesions were photographed more than once. If you split the *images* randomly into train and test, the same lesion's photos can end up on both sides. The model then sees half a lesion in training, the other half in testing, and gets to cheat on the metric. v1 did exactly this. v2 fixed it.

**Technical detail.** The right way is `StratifiedGroupKFold` from scikit-learn:
- *Stratified* — keep the class proportions roughly equal across splits.
- *Group* — every sample with the same `lesion_id` must go to the same split.

The "group" is the source of leakage. If `lesion_id="HAM_0001234"` has images `ISIC_001.jpg`, `ISIC_002.jpg`, `ISIC_003.jpg`, then either all three are in train, or all three are in val, or all three are in test. Never split across.

**In your project.** [`src/data/splits.py`](../v2/src/data/splits.py) does it in two passes:

```python
# Pass 1: split off the test set
test_idx = StratifiedGroupKFold(n_splits=int(1/test_ratio)).split(X, y, groups=lesion_ids)
# Pass 2: split off val from train+val
val_idx = StratifiedGroupKFold(n_splits=int(1/val_ratio_of_remaining)).split(...)
```

Then `assert_no_group_leakage()` runs immediately — it builds the sets of `lesion_id` per split and checks `train ∩ val == val ∩ test == train ∩ test == ∅`. If you ever silently regress this, pytest catches it (`tests/test_splits.py::test_no_lesion_leak`).

**The number this cost us.** Same ConvNeXt-V2 Large model, same hyperparameters, same training time:
- v1 (random image-level split): test AUC **0.9880**
- v2 (lesion-level GroupKFold split): test AUC **0.9660**

That 0.022 gap is the *inflation* the leaky split gave us. The honest number is 0.9660. Every comparison between v1 and v2 numbers has to start with "v2 numbers are on a harder, fairer test set."

**References.**
- [`sklearn.model_selection.StratifiedGroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [HAM10000 paper](https://www.nature.com/articles/sdata2018161) (Tschandl et al. 2018) explicitly recommends `lesion_id` grouping.
- [Roberts et al., 2021 — Common pitfalls and recommendations for using ML for COVID-19](https://www.nature.com/articles/s42256-021-00307-0) on why data leakage invalidates dozens of medical-imaging papers.

---

# PART 18: Cross-Dataset Evaluation — Why One Test Set Isn't Enough

**Plain English.** If you only test a skin-lesion model on HAM10000, you're testing it on Austrian + Australian dermoscopy from one decade. The model could be memorising lighting conditions specific to those clinics. To know if it actually learned *skin disease*, you have to test on different cameras, different countries, different patient populations. That's cross-dataset evaluation.

**Technical detail.** Three external datasets in v2, each with a different distribution:

| Dataset | Modality | Images | Labels | Reason to test |
|---|---|---:|---|---|
| **ISIC 2019** | Dermoscopy (close to HAM10000) | 15,316 | 8 classes (SCC added) | Same modality, different sources — easy generalisation check |
| **DDI** (Stanford AIMI) | Clinical photos, biopsy-proven | 656 | malignant / benign + Fitzpatrick group | Fairness audit on dark skin |
| **Fitzpatrick17k** | Clinical photos, web-scraped | 16,577 (≤3,887 alive) | 114 conditions + Fitzpatrick | Larger fairness audit, noisy labels |

**The taxonomy problem.** HAM10000 has 7 classes (`mel/nv/bcc/akiec/bkl/df/vasc`); DDI has malignant/benign; Fitz17k has 114 specific conditions; ISIC 2019 has 8. You can't directly map between them. v2 defines a **universal-3 taxonomy** `[malignant, benign, non_neoplastic]` and:
- Projects HAM-trained 7-way probs to 3-way (sum mel+bcc+akiec → malignant; sum nv+bkl+df+vasc → benign; non_neoplastic = 0).
- Maps DDI's malignant/benign labels into universal-3 directly.
- Maps Fitz17k's `three_partition_label` (`benign / malignant / non-neoplastic`) one-to-one.
- For ISIC 2019 stays in HAM10K-7 (just maps SCC → akiec, the closest match).

**In your project.** [`src/cross_dataset.py`](../v2/src/cross_dataset.py) is the entry point; the per-dataset loaders live in `src/data/external/{ddi,fitzpatrick17k,isic2019}.py`; the taxonomy mappings in `src/data/external/taxonomy.py`. The conversion function `map_ham10k_probs_to_universal()` is the key bridge.

**The headline numbers.** A HAM10000-trained model's ISIC 2019 macro AUC is typically 0.11–0.30 lower than its in-distribution AUC. On DDI it drops further (clinical vs dermoscopy modality shift). On Fitz17k, even further.

**Counter-intuitive finding from v2.** The *best on-distribution* model (`convnextv2_large` at HAM 0.9660) isn't the best out-of-distribution. The dermatology-pretrained `pandem_real_probe` at HAM 0.9527 (lower in-distribution) crushes everything else on ISIC 2019 (0.8845 vs convnext's 0.8546). Pretraining domain matters more than in-distribution accuracy for transfer.

**References.**
- [BCN20000 (Combalia et al., 2019)](https://arxiv.org/abs/1908.02288) — the BCN20000 portion of ISIC 2019 came from a different hospital and different camera than HAM10000.
- [Daneshjou et al., 2022 — DDI](https://www.science.org/doi/10.1126/sciadv.abq6147)
- [Groh et al., 2021 — Fitzpatrick17k](https://openaccess.thecvf.com/content/CVPR2021W/ISIC/papers/Groh_Evaluating_Deep_Neural_Networks_Trained_on_Clinical_Images_in_Dermatology_CVPRW_2021_paper.pdf)

---

# PART 19: Fitzpatrick Skin Tones and the Fairness Story

**Plain English.** The Fitzpatrick scale (1948, dermatologist Thomas Fitzpatrick) classifies skin into six types based on how it reacts to sun: **Type I** (very fair, always burns) through **Type VI** (deeply pigmented, never burns). It is the de-facto language of dermatology research for skin tone.

If a model's accuracy is much higher on light skin than on dark, it's *unfair* — and in dermatology this is a known, measured problem. ImageNet pretraining is biased toward light skin (which dominates the training set), so models inherit that bias.

**Technical detail.** v2 uses three buckets to keep per-bucket sample sizes large enough for meaningful AUCs:

```python
def bucket_fitzpatrick(score):
    if score in (1, 2): return "light"
    if score in (3, 4): return "medium"
    if score in (5, 6): return "dark"
    return None  # unknown
```

The fairness metric is **light-to-dark Malignant AUC gap**: `Mal_AUC(light) - Mal_AUC(dark)`. A positive gap means the model is worse on dark skin (the typical failure mode); a small gap means parity; a negative gap means the model is *better* on dark skin.

**In your project.** Every cross-dataset report for DDI and Fitz17k contains:
```
Fitzpatrick-stratified Malignant AUC:
  light    N= 818  pos= 227  AUC=0.6884
  medium   N=2076  pos= 354  AUC=0.5996
  dark     N= 836  pos=  93  AUC=0.5784
Light-to-dark AUC gap: 0.1101
```

**Two real findings.**

1. **`pandem_real_probe` has the best absolute dark-skin Mal AUC on DDI (0.5919)** but *also* the widest gap (0.175). It lifts all groups, but lifts light/medium more. So "highest dark-skin number" and "smallest gap" can point at different models.

2. **The fair-est model differs by dataset modality.** On DDI (biopsy-proven photos), dermatology-pretrained PanDerm wins on dark skin. On Fitz17k (general clinical photos), ImageNet-pretrained `dinov2_registers_ft` wins. The pretraining-domain advantage isn't universal.

**References.**
- [Fitzpatrick, 1988 — *The Validity and Practicality of Sun-Reactive Skin Types I Through VI*](https://archderm.jamanetwork.com/article.aspx?volume=124&issue=6&page=869)
- [Groh et al., 2021 — Fitzpatrick17k](https://openaccess.thecvf.com/content/CVPR2021W/ISIC/papers/Groh_Evaluating_Deep_Neural_Networks_Trained_on_Clinical_Images_in_Dermatology_CVPRW_2021_paper.pdf) — established the metric.
- [Daneshjou et al., 2022 — DDI](https://www.science.org/doi/10.1126/sciadv.abq6147) — biopsy-proven gold-standard labels for Fitzpatrick fairness audits.

---

# PART 20: Foundation Models for Dermatology (Beyond ImageNet)

> See also [`MODELS.md`](MODELS.md) at the project root — a compact per-model compendium (what / how it works / why chosen / result / paper link) covering every backbone and technique. This Part is the narrative version; `MODELS.md` is the quick-reference table version.

**Plain English.** A *foundation model* is a giant pretrained model that's supposed to "know" everything in its domain. Examples in vision: CLIP (OpenAI), DINOv2 (Meta), SigLIP (Google). For dermatology specifically, two recent ones matter:

- **PanDerm** (Yan et al., *Nature Medicine* 2025) — vision encoder pretrained on 2 million dermatology images via self-supervision. Trained specifically to be good at skin disease.
- **DermLIP / Derm1M** (Yan et al., ICCV 2025) — CLIP-style image+text pretraining on dermatology images aligned with clinical-ontology text. The same group's text-aligned successor.

**Technical detail.** Why a domain-pretrained foundation model could beat a generic one:
- ImageNet has ~1.3M natural-image classes (dog, cat, car...). A model pretrained on it has to compress vast irrelevant variability.
- 2M dermatology images is a smaller and *more focused* corpus. The model spends all its capacity on the relevant feature axes (pigment patterns, vessel arrangements, scale texture).
- For downstream classification on dermatology, you'd expect a linear probe over the dermatology-pretrained features to beat a full fine-tune of an ImageNet-pretrained model. v2 measured exactly this.

**In your project.** `pandem_real_probe` (linear probe on top of frozen real PanDerm) achieves test AUC 0.9527 — better than full fine-tuning EfficientNetV2-M (0.9474) or DINOv2+Registers (0.9358). Cross-dataset, the lead widens: ISIC 2019 macro AUC 0.8845 (PanDerm probe) vs 0.8546 (ConvNeXt-V2 full FT).

But: the v2 work also shows that **full fine-tuning the dermatology backbone hurts at 7k images** (`pandem_real_ft` drops to 0.9295). The pretraining is so good that disturbing it does net damage. This is a known small-data result; see [Kumar et al., 2022 — Fine-Tuning can Distort Pretrained Features](https://arxiv.org/abs/2202.10054).

**The bug we found.** The PanDerm checkpoint on Hugging Face (`redlessone/DermLIP_PanDerm-base-w-PubMed-256`) is packaged as an **open_clip** model, not a `transformers` model. It has `open_clip_config.json` and `open_clip_pytorch_model.bin` but no `config.json`. `transformers.AutoModel.from_pretrained` raises and a `try/except` in our `PanDermClassifier` silently substituted DINOv2+Registers. Every result labelled "PanDerm" before the fix was actually DINOv2+Registers. See PART 22 for the full debugging story.

**References.**
- [Yan et al., 2025 — PanDerm (Nat. Med.)](https://www.nature.com/articles/s41591-025-03747-y)
- [Yan et al., 2025 — DermLIP / Derm1M (arXiv 2503.14911, ICCV 2025)](https://arxiv.org/abs/2503.14911)
- [Kumar et al., 2022 — *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*](https://arxiv.org/abs/2202.10054)

---

# PART 21: BEiT — The Architecture Hiding Inside PanDerm

**Plain English.** Looking at PanDerm's weights when we did the load fix, we discovered the vision encoder is a **BEiT** (Microsoft, 2021) — a ViT variant with a few specific quirks. Worth understanding because matching the architecture exactly is what got us "0 missing, 0 unexpected keys" on load.

**Technical detail.** Compared to a vanilla ViT-Base/16, BEiT differs in three concrete ways:

1. **Attention bias separation.** A vanilla ViT has one `qkv.weight` matrix and one `qkv.bias` (with all three Q/K/V biases concatenated). BEiT separates the biases: `q_bias` and `v_bias` are learned parameters, `k_bias` is implicitly zero. The reasoning: in self-attention, K is multiplied by Q, so an additive bias on K is redundant.

2. **LayerScale.** Each transformer block has two learned per-channel scalars `gamma_1` (before residual after attention) and `gamma_2` (before residual after MLP). They start near zero, so the block initially acts like a no-op, which stabilises training of very deep transformers.

3. **Position embeddings — absolute vs relative.** Original BEiT uses *relative* position bias (a learned `relative_position_bias_table`). PanDerm's variant uses *absolute* position embeddings (`pos_embed` shape `(1, 197, 768)`) — closer to a vanilla ViT but with the rest of BEiT's machinery. This is a custom decision; the open_clip recipe set `use_abs_pos_emb=True, use_rel_pos_bias=False`.

**The exact loading recipe** (from [`RealPanDermClassifier`](../v2/src/models/registry.py)):

```python
from timm.models.beit import Beit
backbone = Beit(
    img_size=224, patch_size=16, in_chans=3,
    num_classes=0,                # we add our own head
    embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    use_abs_pos_emb=True,
    use_rel_pos_bias=False,
    use_shared_rel_pos_bias=False,
    init_values=0.1,              # turn on LayerScale, init at 0.1
    global_pool='token',          # use CLS-token features (not avg pool)
)
```

The `global_pool='token'` is critical — with `'avg'` (default for BEiT), timm builds a `fc_norm` layer and uses average-pooled patch features; with `'token'`, it builds a `norm` layer over the CLS token. The checkpoint's keys say `norm.weight/bias`, not `fc_norm.weight/bias`, so we need `'token'`.

**Why this matters as a general lesson.** Foundation-model checkpoints have a *lot* of small architecture decisions baked in. Loading via `AutoModel` masks them. When `AutoModel` fails — or worse, falls back silently — you have to read the actual state_dict keys to reverse-engineer the architecture. That's exactly what we did.

**References.**
- [Bao et al., 2022 — BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254)
- [Touvron et al., 2021 — Going Deeper with Image Transformers (LayerScale)](https://arxiv.org/abs/2103.17239)
- [timm BEiT source](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/beit.py)

---

# PART 22: open_clip vs transformers — Two Vision Foundation Ecosystems

**Plain English.** There are two main Python ecosystems for vision foundation models:
- **HuggingFace `transformers`** — assumes a `config.json` next to the weights. Auto-instantiates the right Python class via `AutoModel.from_pretrained()`. Works for BERT, DINOv2, SigLIP, etc.
- **OpenCLIP** — original library for CLIP-style models. Uses `open_clip_config.json` (a different schema) and provides `open_clip.create_model_and_transforms()`.

A HF repo can host either kind of model. **`AutoModel.from_pretrained()` does NOT magically work on OpenCLIP repos** — it looks for `config.json`, doesn't find it, raises.

**Technical detail — how to tell which kind of repo you're looking at.** Inspect the file list:

```python
from huggingface_hub import list_repo_files
print(list_repo_files("redlessone/DermLIP_PanDerm-base-w-PubMed-256"))
```

If you see `config.json` → transformers. If you see `open_clip_config.json` + `open_clip_pytorch_model.bin` → OpenCLIP.

The PanDerm repo:
```
.gitattributes
README.md
open_clip_config.json          ← OpenCLIP signature
open_clip_model.safetensors
open_clip_pytorch_model.bin
special_tokens_map.json        ← tokenizer files for the text head
tokenizer.json
tokenizer_config.json
vocab.txt
```

No `config.json`. Definitely OpenCLIP. The `open_clip_config.json` tells you (a) what the vision architecture is, (b) what normalisation to use, (c) what the text encoder is.

**The exact failure.** Our `PanDermClassifier.__init__` did:
```python
try:
    self.backbone = AutoModel.from_pretrained(model_id, trust_remote_code=True)
except Exception as e:
    print(f"[WARN] Failed to load {model_id}: {e}")
    print("  Falling back to DINOv2+Registers...")
    self.backbone = AutoModel.from_pretrained("facebook/dinov2-with-registers-base")
```

The `try` raises `ValueError: missing model_type key in config.json` (because there *is* no config.json, but the error text matches), the `except` catches, the fallback runs, and training proceeds — silently — on completely wrong weights. The model still trains, still validates, still produces a `results.txt`. It just doesn't contain the model you think it does. Every metric labelled "PanDerm" in v2's early `experiments/` was actually DINOv2+Registers.

**Lessons.**
1. *Never* put a silent fallback in a model loader. Either fail loudly (raise) or warn unmistakably (print to stderr in a tagged format that makes it obvious in a CI log).
2. When you can't `AutoModel` a repo, check for `open_clip_config.json` first. If found, you can load via `open_clip.create_model_and_transforms('hf-hub:<repo>')` OR, if the open_clip version doesn't match the saved config schema (it didn't for us), do it manually:
   ```python
   from huggingface_hub import hf_hub_download
   weights = torch.load(hf_hub_download(repo, 'open_clip_pytorch_model.bin'))
   visual_sd = {k.replace('visual.', '', 1): v for k, v in weights.items() if k.startswith('visual.')}
   model = build_matching_timm_arch_from_config(...)
   model.load_state_dict(visual_sd, strict=True)
   ```

**References.**
- [OpenCLIP GitHub](https://github.com/mlfoundations/open_clip)
- [HuggingFace `transformers` `AutoModel` docs](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModel)
- [`huggingface_hub.list_repo_files`](https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download#huggingface_hub.list_repo_files)

---

# PART 23: Post-Hoc Calibration Pipeline (v2 vs Phase 21)

**Plain English.** After you've trained a model, its output probabilities aren't necessarily *calibrated* — when the model says "80% melanoma", it might actually be right only 60% of the time, or 95% of the time. The post-hoc pipeline tries to fix this without retraining.

v2 uses a three-stage pipeline fit on validation, applied to test:

1. **Per-class SLSQP weights** — find ensemble weights that maximise *per-class* OvR AUC (not global).
2. **Platt sigmoid calibration** — re-map raw probabilities so that "80% confidence" actually means 80% likelihood.
3. **F2-optimal per-class thresholds** — pick the cutoff probability per class to maximise F2 score (recall-favouring).

**Technical detail.**

*Per-class SLSQP.* For each class `c` independently:
- Objective: maximise `roc_auc_score(y == c, ensemble_prob[:, c])` where `ensemble_prob = sum(w_m * prob_m[m])`.
- Constraints: `sum(w) == 1`, `0.01 <= w <= 0.99`.
- Multi-restart from one-hot inits to escape local optima.
- Different classes can pick different weight vectors — e.g. for `mel` the ensemble might lean on convnext, while for `df` it might lean on pandem_real.

*Platt sigmoid.* For each class, fit a 1-parameter logistic regression:
```python
from sklearn.linear_model import LogisticRegression
calibrator = LogisticRegression()
calibrator.fit(val_probs[:, c].reshape(-1, 1), (val_labels == c).astype(int))
calibrated_test_prob_c = calibrator.predict_proba(test_probs[:, c].reshape(-1, 1))[:, 1]
```
Because the sigmoid is strictly monotonic, this **preserves the AUC** while shifting the probability scale. *Don't renormalise rows after Platt* — that would destroy per-class ranking and tank macro OvR AUC.

*F2 thresholds.* For each class, grid-search threshold ∈ [0.01, 0.99] step 0.01 to maximise `F2(precision, recall, beta=2)` on val. F2 weights recall 4× more than precision (β=2: weight_recall = β² = 4). For melanoma, missing a malignant case is much worse than a false alarm — so F2, not F1.

**Why Platt and not isotonic?** Isotonic regression is more flexible (piecewise constant rather than sigmoid) but it has a degenerate failure mode: on small rare-class val populations (HAM10000 has `df=16, vasc=20` in val), isotonic collapses to <30 unique output values per class, which kills the OvR ranking and drops macro AUC by ~0.04 in our tests. Platt is monotonic and safe.

**In your project.** [`src/post_hoc.py`](../v2/src/post_hoc.py). The canonical run:
```
mean ensemble baseline   :  0.9713
+ per-class SLSQP weights:  0.9733
+ Platt calibration      :  0.9733  (preserves AUC by design)
+ F2 thresholds          :  0.9733  (only affects classification, not AUC)
```

The final mel recall (at the F2 threshold) is **0.566** — that's the operating point. AUC tells you the ranking is good (0.97); recall@threshold tells you what fraction of melanomas you'd actually catch at that cutoff.

**References.**
- [Platt, 1999 — Probabilistic outputs for SVMs](https://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf)
- [Niculescu-Mizil & Caruana, 2005 — *Predicting Good Probabilities with Supervised Learning*](https://www.cs.cornell.edu/~caruana/niculescu.scldbst.crc.rev4.pdf) (comparison of Platt vs isotonic)
- [`scipy.optimize.minimize` with `method="SLSQP"`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
- [F2 score in `sklearn`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)

---

# PART 24: Test-Time Adaptation (TENT)

**Plain English.** Normally, once you train a model, you freeze it and use the same weights forever. **TENT** (Test-Time Entropy Minimisation, Wang et al. ICLR 2021) breaks that rule: at inference time, you let a tiny fraction of the model's parameters adapt to each test batch by minimising the model's own uncertainty (entropy). No labels involved.

The intuition: if the model is uncertain on the test data because the test distribution is different from training (a darker-skinned patient population, a different camera), pushing it to be *more confident* on the test batch can pull its predictions toward the right answer — assuming the model has *some* signal to work with.

**Technical detail.** What TENT adapts:
- All parameters frozen *except* the affine parameters of normalisation layers (`weight` and `bias` of `BatchNorm` / `LayerNorm` / `GroupNorm`).
- For BatchNorm: also disable `track_running_stats` so each test batch estimates its own (μ, σ²). This is the bigger lever for BN-rich models.

What TENT optimises:
- For each test batch x, compute `logits = model(x)`.
- Loss = mean per-sample entropy of the softmax: `H(p) = -∑ p_i log p_i`.
- One Adam step on the unfrozen params.
- Final forward (no_grad) gives the adapted prediction.

`src/tent.py` in v2:
```python
def wrap_for_tent(model):
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, (BatchNorm1d, BatchNorm2d, BatchNorm3d, LayerNorm, GroupNorm)):
            m.weight.requires_grad = True
            m.bias.requires_grad = True
            if isinstance(m, _BatchNorm):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model
```

**What we found in v2.**

*DDI* (656 biopsy-proven photos, three Fitzpatrick groups). TENT helped 3 of 4 backbones substantially on dark-skin Malignant AUC:

| Backbone | Baseline dark-Mal | + TENT lr=1e-3 | Δ |
|---|---:|---:|---:|
| `pandem_real_probe` (BEiT) | 0.5919 | **0.6931** | +0.10 |
| `dinov2_registers_ft` (ViT) | 0.5419 | 0.6562 | +0.11 |
| `convnextv2_large` (LN-rich) | 0.5445 | 0.6357 | +0.09 |
| `efficientnetv2_m` (BN-rich) | 0.4869 | 0.4374 | -0.05 |

*Fitz17k* (3,730 clinical photos). Mixed results — `efficientnetv2_m + n_steps=3` reached the best dark-skin Mal AUC of 0.6742, but the same `pandem_real_probe` that worked on DDI completely fails here:

| Backbone | Baseline | + TENT (best lr) | Δ |
|---|---:|---:|---:|
| `pandem_real_probe` | 0.5531 | 0.4876 | -0.07 |

**Why TENT can hurt.** Entropy minimisation pushes toward *confident* predictions, but a confident prediction can be confidently wrong if the model has no signal on the new distribution. On clinical photos that look nothing like dermoscopy, PanDerm's features are out of distribution; TENT amplifies that distortion rather than fixing it. Not all distribution shifts are TENT-fixable.

**Hyperparameter rules of thumb (from our sweep).**
- BatchNorm-rich CNNs (efficientnetv2_m): lr=1e-3 works; more iterations (n_steps=3) can help.
- LayerNorm-only models (ViTs, BEiT): lr=1e-3 works *on biopsy-proven dermoscopy-like data*. On general clinical photos, drop to lr=1e-4. If it still hurts, don't use TENT.

**References.**
- [Wang et al., 2021 — TENT: Fully Test-Time Adaptation by Entropy Minimization](https://arxiv.org/abs/2006.10726)
- [Liang et al., 2023 — A Comprehensive Survey on Test-Time Adaptation](https://arxiv.org/abs/2303.15361) (broader landscape of TTA techniques)
- `src/tent.py`, `scripts/run_tent.py` — full implementation + invocation.

---

# PART 25: Sample Efficiency — Probing How Good a Pretrain Is

**Plain English.** A good pretrained model needs *less* labelled data to hit its asymptote. If you train two models — one ImageNet-pretrained, one dermatology-pretrained — and ask each one to learn skin lesion classification, the dermatology-pretrained one should hit useful accuracy with far fewer examples. v2 measured this directly.

**Technical detail.** Subsample HAM10000's training lesions to fractions f ∈ {0.10, 0.25, 0.50, 1.00}, keep val and test fixed, train each model on each subsample, plot test AUC vs f. Stratify the subsampling by *lesion-level dx mode* so rare classes don't disappear:

```python
def subsample_train_by_lesion(train_df, fraction, seed=42):
    lesion_class = train_df.groupby("lesion_id")["dx"].agg(lambda x: x.mode().iloc[0])
    keep_ids = []
    for cls in lesion_class.unique():
        lesions_in_cls = lesion_class[lesion_class == cls].index.to_numpy()
        keep_ids.extend(rng.choice(lesions_in_cls, size=int(len(lesions_in_cls) * fraction), replace=False))
    return train_df[train_df["lesion_id"].isin(keep_ids)]
```

**Results (v2).**

| Train fraction | Lesions | Images | `pandem_real_probe` test AUC | `efficientnetv2_m` test AUC |
|---|---:|---:|---:|---:|
| 25% | 1,335 | 1,811 | **0.9398** | 0.8984 |
| 50% | 2,668 | 3,579 | 0.9461 | 0.9216 |
| 100% | 5,336 | 7,153 | 0.9526 | 0.9500 |

Reading the table:
- `pandem_real_probe` at 25% reaches what `efficientnetv2_m` gets at 100% (~0.95). That's ~4× more data-efficient.
- `pandem_real_probe`'s curve is *flat* — only +0.013 from 25% to 100%. The probe has saturated.
- `efficientnetv2_m`'s curve is steep — +0.052 from 25% to 100%. It's still learning at full data.

**Why this matters in practice.** In clinical-deployment settings, labelled data is scarce. Showing that a dermatology-pretrained probe needs only 1,800 labelled training images to reach the same accuracy as a generic CNN needing 7,000 is a *deployment-relevant* claim, not just an academic one.

**References.**
- [Kolesnikov et al., 2020 — Big Transfer (BiT)](https://arxiv.org/abs/1912.11370) introduced systematic sample-efficiency curves.
- [Goyal et al., 2021 — Self-Supervised Pretraining is Robust to Label Noise](https://arxiv.org/abs/2104.13963) — broader argument that SSL features are robust.
- `scripts/sample_efficiency.py` — the v2 implementation.

---

# PART 26: SLSQP for Ensemble Weights — Why Not Just Average?

**Plain English.** When ensembling four models, the simplest thing is to average their probabilities. But maybe convnextv2 is twice as accurate as pandem_ft — averaging gives them equal say, which is suboptimal. **SLSQP** (Sequential Least SQuares Programming) finds the per-model weights that maximise an objective (like AUC) subject to constraints (weights sum to 1, each weight between 0 and 1).

**Technical detail.** The optimisation problem:

```
maximise   AUC(sum_m w_m * probs[m], labels)
subject to sum(w) == 1
           0.01 <= w_m <= 0.99    for all m
```

SLSQP needs gradients of the objective, but AUC is *non-smooth* (it changes in steps as the threshold sweeps through probability values). The workaround: do many SLSQP runs from different starting points and take the best. We start from:
- uniform `[1/M, 1/M, ..., 1/M]`
- one-hot-ish `[0.7, 0.06, 0.06, ...]` for each model in turn

Without multi-restart, SLSQP from uniform-init usually gets stuck on the AUC plateau and returns the uniform weights it started with — that's a real failure mode we hit and fixed.

**In your project.** [`src/ensemble.py`](../v2/src/ensemble.py) (global ensemble weights) and [`src/post_hoc.py`](../v2/src/post_hoc.py) (per-class). For the canonical 4-model HAM10000 ensemble:
```
weights:
  convnextv2_large    : 0.775
  efficientnetv2_m    : 0.075
  dinov2_registers_ft : 0.075
  pandem_real_probe   : 0.075
test AUC : 0.9731
```
The strongest model dominates (~78% weight); the other three contribute diversity at the floor weight (~7.5% each). That's the SLSQP-optimal partition.

**Why per-class SLSQP is better than global.** A global weight vector assigns the same weight to convnextv2 across all 7 classes. A per-class SLSQP gives convnextv2 weight 0.85 for `mel/nv/bkl/df` and only 0.20 for `bcc/akiec` — because pandem_real_probe is actually better on the rarer pigmented-lesion classes. Per-class search lifts the post-hoc result from 0.9731 to 0.9733.

**References.**
- [`scipy.optimize.minimize(method="SLSQP")`](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)
- [Kraft, 1988 — A Software Package for Sequential Quadratic Programming](https://www.researchgate.net/publication/238757005) (the original SLSQP algorithm)

---

# PART 27: Cross-Dataset SLSQP — The Accuracy-Fairness Frontier

**Plain English.** A single global ensemble weight vector is optimal *for the dataset it was fit on*. The HAM10000-optimal ensemble (which weights convnextv2 heavily) is *not* the ensemble you want when deploying on DDI (where pandem_real_probe carries more signal) or Fitz17k (where efficientnetv2_m wins on dark skin). v2 fits **per-target ensembles** to make this explicit.

**Technical detail.** Same SLSQP machinery as PART 26, but:
- Use the *target dataset's* probabilities and labels (DDI, Fitz17k, or ISIC), not HAM10000.
- Two objectives, fit separately:
  - **Accuracy-optimal**: maximise target dataset's macro OvR AUC.
  - **Fairness-optimal**: maximise dark-skin Malignant AUC on the target.

The two objectives often pick *different* weight vectors. That difference is the **accuracy-fairness Pareto frontier** made empirical.

**In your project.** [`src/cross_dataset_ensemble.py`](../v2/src/cross_dataset_ensemble.py). Concretely on Fitz17k:

| Target | Recipe | Macro OvR | Dark-Mal AUC | Light-Dark Gap |
|---|---|---:|---:|---:|
| Accuracy-optimal | 0.775 `pandem_real_probe` | 0.5789 | 0.5750 | +0.108 |
| Fairness-optimal | 0.775 `efficientnetv2_m` | 0.5689 | 0.6192 | +0.007 |

Reading the table: sacrificing 0.010 macro AUC buys +0.044 dark-skin Mal AUC and closes the disparity gap from +0.108 to +0.007 (16× reduction). That tradeoff is a *real* decision: a clinician deploying for a diverse patient population should pay the macro-AUC cost.

**On DDI it gets even better.** With TENT-adapted probabilities, SLSQP's accuracy-optimal weights *also* close the disparity gap to ~0 (light-dark gap -0.0004) at dark-skin Mal AUC 0.6571. The two objectives converge — meaning on biopsy-proven dermoscopy-like data, you don't have to trade accuracy for fairness; the TENT-adapted ensemble Pareto-dominates the un-adapted single models.

**Why this is a publishable methodology contribution.** It's small but real:
1. SLSQP for ensemble weights is standard (Phase 21 of v1 used it).
2. Per-class SLSQP is a known trick.
3. Per-*target* SLSQP — fitting different weights for different deployment contexts — is less common in clinical-AI literature and is a clean way to expose the fairness-accuracy frontier.

**References.**
- `experiments/cross_dataset_ensemble_tent_lr1e-3/report.txt` — the actual numbers.
- [Daneshjou et al., 2022 — DDI](https://www.science.org/doi/10.1126/sciadv.abq6147) and [Groh et al., 2021 — Fitzpatrick17k](https://openaccess.thecvf.com/content/CVPR2021W/ISIC/papers/Groh_Evaluating_Deep_Neural_Networks_Trained_on_Clinical_Images_in_Dermatology_CVPRW_2021_paper.pdf) for the fairness framing.

---

# PART 28: Binary Focal Loss for Mel Specialist — The Alpha Parameter Trap

**Plain English.** When training a binary classifier (melanoma vs not), positives (mel) are rare — about 11% of HAM10000. Standard cross-entropy is dominated by the easy negatives. **Focal loss** (Lin et al. 2017) fixes that by down-weighting easy examples (using the `gamma` parameter) and re-weighting positive vs negative classes (the `alpha` parameter).

**Technical detail.** Binary focal loss:
```
FL = -α_t · (1 - p_t)^γ · log(p_t)
where p_t = p if y==1 else 1-p
      α_t = α if y==1 else 1-α
```
- `γ` (focusing) — typically 2.0. Higher γ = more focus on hard examples.
- `α` (class balance) — and *this is where you can shoot yourself in the foot.*

**The OA1 trap.** The Lin et al. paper uses α=0.25 *in object detection*, where the rare class is the foreground object and there are ~10000 background anchors per image — so α=0.25 *down-weights* positives because they're hard to even sample without overwhelming everything. But our setting uses a *balanced batch sampler* (50/50 mel/non-mel via `WeightedRandomSampler`). With balanced batches, α=0.25 down-weights what's already the rare class — exactly backwards.

The first v2 binary specialist trained with α=0.25 produced `convnextv2_large` test mel AUC 0.7671 with **0 recall at every precision threshold** — the model collapsed. After switching to α=0.5 (neutral weighting, since the sampler already balances) and halving the learning rates, test mel AUC recovered to 0.8942 with recall 0.220 at precision=0.80.

**The deeper lesson.** Default hyperparameters from a paper are tied to that paper's setup. Focal-loss α=0.25 assumes you're *not* batch-balancing. With balanced sampling, α=0.5 is the right starting point.

**Did it work?** Sort of. The v2 binary specialist's best run (`mel_binary_convnextv2_large_v2`, test mel AUC 0.8942) does NOT beat the 7-way multi-class model with an F2-tuned mel threshold (mel recall 0.566 at the operating point). On honest lesion-level splits, the v1 Phase 19 result (0.83 mel recall) doesn't reproduce. A clean negative result.

**In your project.** `src/losses.py:BinaryFocalLoss`, `scripts/train_mel_binary.py`, `experiments/mel_binary/mel_binary_*_v2/results.txt`.

**References.**
- [Lin et al., 2017 — Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [`torch.nn.WeightedRandomSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler) — what we used for balanced batches.

---

# PART 29: MLflow as the Project's Memory

**Plain English.** Over 22+ phases × multiple runs per phase × multiple hyperparam tweaks, you lose track of which configuration produced which number. **MLflow** is a per-project SQLite database (`mlflow.db`) that records every training run's parameters, metrics-per-epoch, and final checkpoints. You can query it later: "show me every run with `model_type=pandem` sorted by best val AUC."

**Technical detail.** Three calls per training run:

```python
import mlflow
mlflow.set_experiment("skin-lesion-v2")
with mlflow.start_run(run_name=cfg.name):
    mlflow.log_params({"model_key": model_key, "lr_head": cfg.lr_head, ...})
    for epoch in range(epochs):
        ...
        mlflow.log_metrics({"train_loss": loss, "val_auc": val_auc, ...}, step=epoch)
    mlflow.log_metrics({"test_auc": test_auc})
```

To browse: `mlflow ui` in the v2 directory opens `http://localhost:5000` with sortable run tables, side-by-side metric plots, and hyperparam diffs.

**In your project.** Every entry point (train, sample_efficiency, train_mel_binary) wraps its loop in `mlflow.start_run()`. Look at `D:\ML\skin-lesion-vit\v2\mlflow.db` for the v2 history; `D:\ML\skin-lesion-vit\mlflow.db` for v1.

**Why this saved us.** When the iteration-2 ConvNeXt-V2 binary specialist's recall collapsed to 0, we could pull up the previous (broken) run's `train_loss` curve in mlflow ui side-by-side with the new run's curve. The bad run showed loss plateauing at a non-trivial value with no progress on val mel AUC — diagnostic of the α-trap above. Without mlflow, this is detective work; with mlflow, it's a five-second comparison.

**References.**
- [MLflow Tracking docs](https://mlflow.org/docs/latest/tracking.html)
- [`mlflow.pytorch` integration](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html)

---

# PART 30: The v2 Knowledge Map

If someone asks "where in the project is X?", these are the file pointers:

| Concept | File | Purpose |
|---|---|---|
| Lesion-level splits | `src/data/splits.py` | `StratifiedGroupKFold` + leakage guard |
| Pytest leakage check | `tests/test_splits.py` | `test_no_lesion_leak` |
| HAM10000 loader | `src/data/ham10000.py` | df + Dataset + sampler |
| Normalisation per model_type | `src/data/transforms.py` | `get_norm_stats()` ImageNet / SigLIP / CLIP |
| Model registry | `src/models/registry.py` | `ModelConfig` + `build_model()` |
| Real PanDerm load fix | `src/models/registry.py:RealPanDermClassifier` | timm BEiT + open_clip weights |
| Engine | `src/engine.py` | `train_one_epoch`, `evaluate`, `run_phase` |
| 8-view TTA | `src/tta.py` | softmax-averaged augmented views |
| 4-model SLSQP ensemble | `src/ensemble.py` | multi-restart SLSQP, global weights |
| 3-stage post-hoc | `src/post_hoc.py` | per-class SLSQP + Platt + F2 |
| Cross-dataset eval | `src/cross_dataset.py` | DDI / Fitz17k / ISIC inference |
| Per-target SLSQP | `src/cross_dataset_ensemble.py` | accuracy-fairness frontier |
| TENT adapter | `src/tent.py` | norm-affine entropy minimisation |
| TENT runner | `scripts/run_tent.py` | apply TENT to a (model, dataset) pair |
| Sample efficiency | `scripts/sample_efficiency.py` | subsample train, retrain, log |
| Binary mel head | `scripts/train_mel_binary.py` + `src/losses.py` | focal-loss specialist |
| External-dataset taxonomies | `src/data/external/taxonomy.py` | HAM-7 ↔ universal-3 ↔ ISIC mappings |
| DDI loader | `src/data/external/ddi.py` | 656-image Fitzpatrick-stratified |
| Fitz17k loader | `src/data/external/fitzpatrick17k.py` | URL-based, dead-link tolerant |
| Fitz17k downloader | `data/external/Fitzpatrick17k/download_images.py` | parallel scrape with tqdm |

For runs and artifacts:

| Artifact | Path | Contains |
|---|---|---|
| Per-model HAM10000 results | `experiments/<model>/results.txt` | Best val/test AUC + per-class report |
| Per-model TTA results | `experiments/<model>/tta_results.txt` | `[VAL]` and `[TEST]` blocks |
| 4-model SLSQP ensemble | `experiments/ensemble/results.txt` | canonical 0.9731 |
| Side-by-side ensemble variants | `experiments/ensemble/results_*.txt` | 5-model, pandem_real_ft, etc. |
| Post-hoc pipeline | `experiments/post_hoc/results.txt` | 0.9733 final |
| Cross-dataset baseline | `experiments/cross_dataset/<ds>/<model>/report.txt` | no TENT |
| Cross-dataset + TENT | `experiments/cross_dataset_tent/<ds>/<model>[_suffix]/report.txt` | per-config |
| Per-target SLSQP ensemble | `experiments/cross_dataset_ensemble_tent_lr1e-3/report.txt` | DDI gap closure |
| Sample efficiency | `experiments/sample_efficiency/<model>_fracNNN/results.txt` | E12 |
| Binary mel | `experiments/mel_binary/mel_binary_<bb>_v2/results.txt` | OA1 v2 |

For top-level docs:

| File | Contents |
|---|---|
| `D:\ML\skin-lesion-vit\info_bank\AI_context.md` | Single source of truth for AI agents picking up work |
| `D:\ML\skin-lesion-vit\info_bank\FINDINGS.md` | Consolidated findings + publishability + resume bullets |
| `D:\ML\skin-lesion-vit\info_bank\OVERNIGHT_RESULTS.md` | Chronological log of every overnight run |
| `D:\ML\skin-lesion-vit\info_bank\NEXT_STEPS.md` | Archive index — what's done, what's deliberately deferred |
| `D:\ML\skin-lesion-vit\info_bank\Skin_Lesion_Project_Complete_Guide.md` | This file — pedagogical reference |
| `D:\ML\skin-lesion-vit\README.md` | v1 leaderboard + phase descriptions (frozen) |
| `D:\ML\skin-lesion-vit\v2\README.md` | v2 public-facing project page |
| `D:\ML\skin-lesion-vit\v2\PHASES.md` | Live phase ledger (E1-E15, B1-B6, OA1-OA4) |

---

# PART 31: Glossary of v2-Specific Terms

- **Lesion-level split** — train/val/test disjoint by `lesion_id`. Multiple images of the same lesion can't span splits.
- **Universal-3 taxonomy** — `[malignant, benign, non_neoplastic]`. Used for DDI / Fitz17k where mapping to HAM10K-7 isn't clean.
- **Fitzpatrick bucketing** — collapse the 6-point Fitzpatrick scale into 3 buckets (light: 1–2, medium: 3–4, dark: 5–6) for adequately-sized stratified statistics.
- **Light-to-dark AUC gap** — Mal AUC on light skin minus Mal AUC on dark skin. Positive = worse on dark skin (typical). Near zero = parity. Negative = better on dark skin (rare).
- **Macro OvR AUC** — average per-class one-vs-rest AUC. Treats every class equally regardless of frequency — useful for the 7-class HAM10000 problem where `nv` dominates.
- **F2 score** — F-beta with β=2. Weights recall 4× more than precision. Right metric when missing a positive is much worse than a false alarm — like for melanoma.
- **Platt scaling** — 1-parameter logistic-regression calibration. Strictly monotonic, preserves AUC.
- **SLSQP** — Sequential Least Squares Programming. Constrained optimisation. Used for ensemble weights under a sum-to-1 simplex constraint.
- **TTA (Test-Time Augmentation)** — apply 8 deterministic transforms to each test image, average the softmax probabilities. Small free AUC boost (~0.01-0.02).
- **TENT (Test-Time Entropy Minimisation)** — adapt only normalisation-layer affines per test batch by minimising prediction entropy. No labels needed.
- **Universal projection** — `map_ham10k_probs_to_universal(probs_7) → probs_3`. Sums HAM10K classes into universal-3 buckets. Used when applying a HAM10K-trained model to DDI / Fitz17k.
- **Probe** (synonym: linear probe) — train only a linear head on top of frozen features. Tests how good the features are out-of-the-box.
- **Full FT** (fine-tune) — unfreeze the entire backbone and train all parameters. More expressive, but can over-write strong pretrained features (we saw this with `pandem_real_ft`).
- **TENT-aware ensemble** — fit SLSQP weights on the TENT-adapted probabilities (not the un-adapted ones). Closes the DDI Fitzpatrick gap to ~0.

---

# PART 32: Recommended Reading Order

If a new collaborator joins:

1. **PARTS 1–9** of this guide — ML basics + the 7 v1 backbones.
2. **PART 17** — why v2 exists (the leakage fix).
3. **PARTS 20–22** — foundation models in dermatology + the PanDerm loader story.
4. **PART 23** — the post-hoc pipeline that produces the 0.9733 number.
5. **PARTS 18, 19, 24** — cross-dataset evaluation + the fairness audit + TENT.
6. **PART 25** — sample efficiency.
7. **PARTS 26, 27** — the SLSQP machinery.
8. **PART 30 (knowledge map)** — for the rest of their tenure on the project.

That order matches the conceptual dependency graph: ML basics → honest evaluation → modern backbones → calibration → fairness → optimisation. Skip around within those groups freely.

---

*This guide was extended in 2026-05-15. Original Parts 1–16 cover v1 (with PART 16 a Quick Reference Index); Parts 17–32 cover the v2 extensions. Numbers in v2 sections traced to `experiments/` artifacts as of that date.*
