# Clothes Segmentation Report (CVCY002)

**Task:** segment worn clothes from images of people (virtual fitting use case).
**Result in one line:** U-Net / ResNet34 reaches **0.7685 mIoU** (0.863 mDice, 98.2% pixel accuracy) on the 1,000-image validation split and 0.6748 mIoU over the classes present in the small 200-image test split, at 9.2 ms per 512x512 image.

All numbers below come from one training run on Kaggle (2x Tesla T4, `notebooks/segmentation-clothes-kaggle.ipynb`). The scripts in `src/` implement the same pipeline.

---

## 1. Dataset choice and reason

**Dataset:** ATR (Active Template Regression) human-parsing dataset, 18 pixel-level labels. Splits used: **16,706 train / 1,000 val / 200 test** images.

**Why ATR**
- Person-centric images with per-pixel labels for individual garments (upper-clothes, pants, skirt, dress, belt, shoes, bag, scarf, hat), matching the "segment worn clothes from people" task directly.
- Body-part labels (face, hair, arms, legs) let the model learn the boundary between skin and clothing, which matters for virtual fitting.
- Large enough to train a deep model, with ready-made train/val/test splits.

**Label handling.** The 18 raw labels are merged into **11 classes**: hair, face, arms, legs and sunglasses became `body`, and left/right shoes became `shoes`. This focuses the task on clothes and removes left/right ambiguity. (`--full18` keeps all 18.)

**Class balance (train split, first 2,000 images):**

| Class | Pixel share |
|-------|-------------|
| background | 76.33% |
| body | 7.32% |
| upper-clothes | 6.08% |
| pants | 2.86% |
| dress | 2.83% |
| skirt | 1.92% |
| bag | 1.18% |
| shoes | 1.00% |
| hat | 0.18% |
| scarf | 0.17% |
| belt | 0.10% |

The data is strongly imbalanced: background dominates and hat, belt and scarf are under 0.2% of pixels. This drives the loss design (Section 3) and the choice of mIoU over pixel accuracy (Section 4).

**Preprocessing**
- Aspect-preserving resize (longest side to 512 px) and centered zero-padding to a square, so garments are not distorted.
- ImageNet mean/std normalization (required by the pretrained encoder).

**Augmentation (training only)**
- Horizontal flip (p=0.5) with left/right label swapping (shoes 9<->10, legs 12<->13, arms 14<->15) applied on the raw mask before class merging. Albumentations' own flip would silently corrupt these labels.
- Affine: scale 0.75-1.25, rotation +-15 deg, translation +-10% (p=0.7; masks use nearest-neighbor interpolation).
- Color jitter (brightness, contrast, saturation +-0.3, hue 0.03; p=0.6) and, with p=0.2, Gaussian blur or noise to mimic phone-camera conditions.
- Validation and test use no augmentation. Samples: `results/figures/augmentation_samples.png`.

## 2. Model architecture

**U-Net** with a **ResNet34** encoder pretrained on ImageNet (`segmentation_models_pytorch`), 24.4M parameters.

- **Encoder:** ResNet34 extracts features at five scales (1/2 to 1/32 resolution). Pretrained weights speed up convergence and help generalization.
- **Decoder:** upsamples step by step and concatenates the matching encoder feature map at each level (skip connections) to recover fine boundaries, which matters for garment edges.
- **Head:** 1x1 convolution to 11 class logits at input resolution.
- **Why this choice:** skip connections suit precise boundaries, and ResNet34 is light enough to train in under two hours on two T4 GPUs and to run in real time.

**Training setup**
- AdamW (lr 3e-4, weight decay 1e-4), cosine learning-rate schedule to 1e-6, 25 epochs.
- Input 512x512, batch 16 per GPU on 2 GPUs (effective batch 32), mixed precision (AMP), DistributedDataParallel.
- **EMA of the weights** (decay 0.999). After every epoch both the raw and the EMA weights are validated and the better one is kept; the final checkpoint (epoch 22) uses the EMA weights.
- Best checkpoint selected by **validation mIoU**; the test split was evaluated once, after training.
- About 235 s per epoch, roughly 1.6 hours in total.

## 3. Loss function selection and reason

**Loss = class-weighted cross-entropy + Dice loss.**

- **Cross-entropy** gives stable per-pixel gradients and drives overall classification.
- **Dice loss** optimizes region overlap directly and is much less affected by class imbalance, so small classes (hat, belt, scarf) still receive meaningful gradient.
- **Class weights** are the inverse square root of pixel frequency, scaled so the most common class has weight 1 and capped at 10: background 1.0, body 3.2, upper-clothes 3.5, pants 5.2, dress 5.2, skirt 6.3, bag 8.0, shoes 8.7, hat/belt/scarf 10.

Plain cross-entropy alone would reach high pixel accuracy by predicting background almost everywhere, which is why the combination is used.

## 4. Performance analysis and evaluation metrics

**Metrics**
- **mIoU** (headline): intersection over union averaged over classes, insensitive to the dominant background.
- **mDice:** overlap measure that is more forgiving on small regions.
- **Per-class IoU/Dice:** exposes weak classes hidden by the average.
- **Pixel accuracy:** reported for completeness only, since background is about 76% of pixels.
- **mIoU (present classes):** mIoU over classes that occur in the ground truth. This matters on the test split, where scarf has no ground-truth pixels at all and would otherwise count as IoU 0.

All metrics are computed from a confusion matrix accumulated over the whole split.

**Training behavior** (`results/figures/curves.png`). Validation mIoU rose steadily and then flattened at about 0.765-0.768 over the last six epochs (0.7685 at epochs 22 and 25), while the training loss kept falling (0.289 to 0.278) and the validation loss stayed near 0.32. The model has converged for this setup; more epochs of the same recipe would not help much. EMA weights were equal or slightly better than raw weights in most epochs.

**Headline results**

| Split | Images | mIoU | mIoU (present classes) | mDice | Pixel acc. |
|-------|--------|------|------------------------|-------|-----------|
| Validation | 1,000 | **0.7685** | 0.7685 | 0.8631 | 0.9816 |
| Test | 200 | 0.6135 | 0.6748 | 0.7068 | 0.9745 |
| Test, flip TTA | 200 | 0.6150 | 0.6765 | 0.7078 | 0.9743 |

Flip test-time augmentation adds only +0.0015 mIoU for twice the inference cost, so the scripts leave it off by default (`--tta` enables it).

**Per-class results** (test Dice derived from test IoU, no TTA)

| Class | Val IoU | Test IoU | Test Dice | Test GT pixel share |
|-------|---------|----------|-----------|---------------------|
| background | 0.990 | 0.985 | 0.993 | 72.9% |
| body | 0.861 | 0.940 | 0.969 | 7.2% |
| hat | 0.755 | 0.520 | 0.684 | 0.023% |
| upper-clothes | 0.827 | 0.893 | 0.944 | 13.1% |
| skirt | 0.808 | 0.604 | 0.753 | 0.73% |
| pants | 0.862 | 0.870 | 0.930 | 5.6% |
| dress | 0.751 | 0.270 | 0.425 | 0.35% |
| belt | 0.545 | 0.491 | 0.659 | 0.013% |
| shoes | 0.745 | 0.832 | 0.909 | 0.13% |
| bag | 0.767 | 0.342 | 0.509 | 0.020% |
| scarf | 0.541 | 0.000 | n/a (no GT) | 0% |

**How to read the validation/test gap.** Validation (1,000 images) is the more reliable estimate. The test split is small and different in two ways:
1. **Several classes are nearly absent.** Hat, belt and bag each cover about 0.02% of test pixels or less, and scarf never appears, so their per-class IoU rests on a handful of pixels and is very noisy. Scarf's 0.0 is an artifact of having no ground truth.
2. **Different image distribution.** Upper-clothes covers 13.1% of test pixels versus 6.1% in train. The worst test images are studio e-commerce photos with plain backgrounds, cropped at the waist or thigh.

On the garments that are well represented in the test split, results are strong: upper-clothes 0.893, pants 0.870, shoes 0.832, body 0.940.

**Per-image view (test).** Mean per-image mIoU is 0.831 and the median is 0.935, but the 10th percentile is 0.586 and the minimum 0.272: most images are segmented very well, and the errors are concentrated in a few failure cases (Section 5). Figures: `results/figures/worst_cases.png`, `results/figures/best_cases.png`.

**Confusion matrix** (`results/figures/confusion_matrix.png`, row-normalized): recall is high for every class that occurs in the test split (background 0.99, body 0.98, shoes 0.98, upper-clothes 0.94, hat 0.92, pants 0.91, bag 0.90, skirt 0.88, dress 0.87), with belt lowest at 0.70. The main confusions are belt to pants (0.27), dress to upper-clothes (0.13), skirt to pants (0.11) and hat to body (0.08).

**Speed.** 9.2 ms per 512x512 image on a Tesla T4 (batch 1, mixed precision), about 109 FPS, 24.4M parameters.

**Test on a personal photo** (`results/figures/personal_photo.png`): a full-body photo of a person in a suit is segmented cleanly into upper-clothes, pants and shoes, with face, hair and hands labeled as body, giving a usable clothes-only cutout.

## 5. Limitations of the system

**Strengths**
- High accuracy on the large, common garments: upper-clothes, pants, shoes (test IoU 0.83-0.89) and clean body/background separation (0.94 / 0.99).
- Sharp boundaries and a usable clothes-only cutout on a real photo with a cluttered background.
- Real-time speed (about 109 FPS on a T4) and a small model (24.4M parameters).

**Drawbacks and observed failure modes**
- **Single-color top and trousers are often labeled as one "dress".** All six worst test images show this: a white top with white trousers, a black top with black trousers, a navy top with navy trousers, a grey t-shirt with dark trousers, and so on, where the model paints most of the outfit as dress. This is why dress has recall 0.87 but test IoU only 0.27: the model over-predicts dress, producing many false positives. It is most likely caused by the missing color boundary between top and bottom in waist-up catalog photos; this explanation has not been tested with a controlled experiment.
- **Rare and thin items are weakest.** Belt (val IoU 0.545) and scarf (0.541) are the lowest classes on validation, and belt is frequently absorbed into pants (27% of belt pixels). Hat, belt, scarf and bag have too few test pixels for a reliable estimate.
- **Neighboring garments blend into each other:** skirt to pants (11%), dress to upper-clothes (13%).

**Limitations / recommended capture conditions**
- Best suited to a single person, mostly upright, with the garments visible. Solid-color outfits photographed only from the waist up are the weak spot described above.
- The model outputs a class mask, not instance masks, so garments of several people in one photo cannot be separated.
- Only 11 classes: no finer garment types (jacket vs. coat vs. shirt) and no separation of layered clothing.
- Input is processed at 512 px and upsampled to the original size, so very thin details (narrow belts, straps) can be lost.
- Dataset bias: ATR's image style and demographics limit generalization to very different domains. Only one personal photo was tested, so behavior under low light, heavy occlusion or unusual poses has not been measured.

**Possible improvements (not done)**
- Reduce the dress class weight or drop class weighting, which may reduce dress false positives.
- Add more waist-up studio-style training images.
- Use a stronger encoder (for example ResNet50) and evaluate on a larger test set.
