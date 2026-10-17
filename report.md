# Clothes Segmentation Report (CVCY002)

**Task:** segment worn clothes from images of people (virtual fitting use case).
**Result in one line:** U-Net / ResNet34 reaches **0.7652 mIoU** (0.861 mDice, 98.1% pixel accuracy) on the 1,000-image validation split and 0.6352 mIoU over the classes present in the small 200-image test split, at 9.2 ms per 512x512 image.

All headline numbers below come from the submitted checkpoint (`results/exp1/best.pt`), trained on Kaggle (2x Tesla T4) and evaluated locally with `src/evaluate.py`. An earlier run of a similar recipe (not included in the repo) is mentioned only for comparison in Section 4. The scripts in `src/` implement the same pipeline.

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
- **EMA of the weights** (decay 0.999). After every epoch both the raw and the EMA weights are validated and the better one is kept (EMA won in 23 of 25 epochs); the final checkpoint (epoch 24) uses the EMA weights.
- Best checkpoint selected by **validation mIoU**; the test split was evaluated only after training.
- About 236 s per epoch, roughly 1.6 hours in total.

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

**Training behavior** (`results/figures/curves.png`). Validation mIoU rose from 0.552 after epoch 1 to 0.712 at epoch 6 and 0.742 at epoch 10, then improved slowly to 0.7585 at epoch 16 and 0.7621 at epoch 18. It dipped to 0.755 at epoch 20 and then climbed again as the learning rate decayed, ending at 0.7652 (epoch 24) and 0.7651 (epoch 25). The last five epochs stay within 0.763-0.765, so the model has essentially converged for this recipe. The training loss kept falling (1.14 to 0.279) while the validation loss bottomed out at 0.323 around epoch 18 and then stayed near 0.327-0.332, a mild sign of overfitting that did not hurt mIoU.

**Headline results**

| Split | Images | mIoU | mIoU (present classes) | mDice | Pixel acc. |
|-------|--------|------|------------------------|-------|-----------|
| Validation | 1,000 | **0.7652** | 0.7652 | 0.8612 | 0.9811 |
| Test | 200 | 0.5775 | 0.6352 | 0.6673 | 0.9734 |

**Per-class results**

| Class | Val IoU | Test IoU | Test Dice | Test GT pixel share |
|-------|---------|----------|-----------|---------------------|
| background | 0.991 | 0.986 | 0.993 | 72.9% |
| body | 0.860 | 0.938 | 0.968 | 7.2% |
| hat | 0.754 | 0.423 | 0.595 | 0.023% |
| upper-clothes | 0.819 | 0.890 | 0.942 | 13.1% |
| skirt | 0.784 | 0.554 | 0.713 | 0.73% |
| pants | 0.863 | 0.870 | 0.931 | 5.6% |
| dress | 0.724 | 0.129 | 0.228 | 0.35% |
| belt | 0.552 | 0.304 | 0.466 | 0.013% |
| shoes | 0.748 | 0.835 | 0.910 | 0.13% |
| bag | 0.777 | 0.424 | 0.595 | 0.020% |
| scarf | 0.544 | 0.000 | n/a (no GT) | 0% |

**How to read the validation/test gap.** Validation (1,000 images) is the more reliable estimate. The test split is small and different in three ways:
1. **Several classes are nearly absent.** Hat, belt and bag each cover about 0.02% of test pixels or less, and scarf never appears, so their per-class IoU rests on a handful of pixels and is very noisy. Scarf's 0.0 is an artifact of having no ground truth.
2. **Different class mix.** Upper-clothes covers 13.1% of test pixels versus 6.1% in train, which suggests a different image distribution (the largest test errors are on a small number of images, see below).
3. **Run-to-run noise on test vs. stable validation.** The submitted checkpoint and an earlier run of a similar recipe (not included in the repo) reached virtually identical validation scores (**0.7652 and 0.7685 mIoU**), but their test mIoU differed noticeably (**0.5775 vs. 0.6135**), almost entirely driven by dress IoU (0.129 vs. 0.270) on the tiny fraction of dress pixels (0.35%) in test. Meanwhile, the well-represented garments remained virtually identical across both runs (upper-clothes 0.890 vs. 0.893, pants 0.870 vs. 0.870, shoes 0.835 vs. 0.832, body 0.938 vs. 0.940). This confirms that validation is the primary benchmark to trust.

On the garments that are well represented in the test split, results are strong: upper-clothes 0.890, pants 0.870, shoes 0.835, body 0.938.

**Per-image view (test).** Mean per-image mIoU is 0.818 and the median 0.881, but the 10th percentile is 0.588 and the minimum 0.282: most images are segmented well, and the errors are concentrated in a few failure cases (Section 5). Figures: `results/figures/worst_cases.png`, `results/figures/best_cases.png`, `results/figures/confusion_matrix.png`.

**Confusion matrix** (`results/figures/confusion_matrix.png`, row-normalized): recall is high for well-represented classes (background 0.99, body 0.98, shoes 0.98, upper-clothes 0.94, pants 0.91, skirt 0.91, bag 0.91, hat 0.85). The lowest recalls are belt (0.54) and dress (0.38). The primary confusions are dress to upper-clothes (0.39) and skirt (0.23), belt to pants (0.34), hat to body (0.14), skirt to pants (0.08), and pants to skirt (0.06).

**Speed.** 9.2 ms per 512x512 image on a Tesla T4 (batch 1, mixed precision), about 109 FPS, 24.4M parameters (the architecture is identical for every checkpoint).

**Test on a personal photo** (`results/figures/personal_photo.png` / `samples/output/`): a full-body photo is segmented into body, upper-clothes, pants and shoes, matching the visible garments cleanly.

## 5. Limitations of the system

**Strengths**
- High accuracy on the large, common garments: upper-clothes, pants and shoes (test IoU 0.84-0.89) and clean body/background separation (0.94 / 0.99).
- Consistent validation performance across the garment classes that have enough pixels (val IoU 0.72-0.86 for every garment except belt and scarf).
- Real-time speed (about 109 FPS on a T4) and a small model (24.4M parameters).

**Drawbacks and observed failure modes**
- **Dress is the weakest garment on test** (IoU 0.129 vs 0.724 on validation, recall 0.38). Real dresses are frequently split into upper-clothes (39%) and skirt (23%), and solid-color tops with trousers are sometimes merged into a single dress or confused with skirt. Dress covers only 0.35% of test pixels, so a handful of images drive this number. In `results/figures/worst_cases.png`, solid-color monochromatic top and trousers (e.g. image #95, black top + trousers) are merged into a single dress, and white trousers are labeled as skirt (images #47 and #131). Conversely, true dresses (images #123 and #192) are split into upper-clothes and skirt. Where no distinct visual seam or color boundary separates garments, the boundary between full-body and two-piece outfits becomes ambiguous.
- **Rare and thin items are weakest.** Belt (val IoU 0.552, test 0.304) and scarf (val 0.544) are the lowest classes on validation, and belt is frequently absorbed into pants (34% of belt pixels). Hat, belt, scarf and bag have too few test pixels for a reliable estimate.
- **Skirt is mid-range** (val IoU 0.784, test 0.554), occasionally confused with pants (8%) or vice versa (6%).

**Limitations / recommended capture conditions**
- Best suited to a single person, mostly upright, with the garments visible. Solid-color outfits photographed only from the waist up are the weak spot described above.
- The model outputs a class mask, not instance masks, so garments of several people in one photo cannot be separated.
- Only 11 classes: no finer garment types (jacket vs. coat vs. shirt) and no separation of layered clothing.
- Input is processed at 512 px and upsampled to the original size, so very thin details (narrow belts, straps) can be lost.
- Dataset bias: ATR's image style and demographics limit generalization to very different domains. Behavior under low light, heavy occlusion or unusual poses has not been measured systematically.

**Possible improvements (not done)**
- Reduce the dress class weight or drop class weighting, which may reduce dress false positives.
- Add more waist-up studio-style training images.
- Use a stronger encoder (for example ResNet50) and evaluate on a larger test set.
