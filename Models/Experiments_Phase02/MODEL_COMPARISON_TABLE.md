# Model Performance Comparison on Mendeley Plant Disease Dataset

**Dataset:** Mendeley Plant Disease Dataset — 7 classes (Bacteria, Fungi, Healthy, Nematode, Pest, Phytopthora, Virus)
**Dataset Split:** 2,460 train / 616 validation (80:20 stratified)
**Evaluation Metrics:** Macro-averaged Precision, Sensitivity (Recall), Specificity, F1-Score; Overall Accuracy; Balanced Accuracy; Matthews Correlation Coefficient (MCC)

---

## Comparison Table

| Model | Framework | Pretrained | Trainable Params | Epochs | Overall Acc. (%) | Avg. Prec. (%) | Avg. Sensitivity (%) | Avg. Specificity (%) | Avg. F1 (%) | Avg. MCC (%) | Bal. Acc. (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **DenseNet169** | TensorFlow | ImageNet | 12.50 M | 25 | **91.56** | **91.98** | **91.56** | 98.51 | **91.62** | **89.76** | 90.47 |
| **DenseNet201** | TensorFlow | ImageNet | 18.11 M | 26 | 89.12 | 89.35 | 89.12 | 98.18 | 89.11 | 86.82 | **91.46** |
| **ResNet50** | TensorFlow | ImageNet | 23.55 M | 28 | 89.12 | 89.69 | 89.12 | 98.12 | 89.19 | 86.81 | 88.01 |
| **MobileNetV2** | PyTorch | ImageNet | 2.23 M | 45 | 87.01 | 87.16 | 87.01 | 97.55 | 86.93 | 84.19 | 81.83 |
| **VGG16** | PyTorch | ImageNet | 134.29 M | 44 | 86.69 | 86.97 | 86.69 | 97.78 | 86.58 | 83.76 | 81.87 |
| **VGG16 + MHA** | PyTorch | ImageNet | 134.49 M | 39 | 85.39 | 86.40 | 85.39 | 97.82 | 85.40 | 82.32 | 81.46 |
| **InceptionV3** | PyTorch | ImageNet | 21.80 M | 36 | 84.74 | 85.44 | 84.74 | 97.50 | 84.63 | 81.48 | 79.64 |

**Best overall value in bold.** Sorted by Overall Accuracy descending.

---

## Architecture Descriptions

| Model | Description |
|---|---|
| **DenseNet169** | Densely Connected Convolutional Network (169 layers). Each layer receives feature maps from all preceding layers via concatenation, enabling strong feature reuse, mitigating vanishing gradients, and reducing parameter count. Achieves SOTA parameter efficiency. |
| **DenseNet201** | Deeper variant of DenseNet (201 layers). More growth steps increase representational capacity at the cost of additional parameters and compute, with marginal accuracy gain over DenseNet169 on this dataset. |
| **ResNet50** | Residual Network (50 layers). Introduces skip (identity) connections that bypass convolutional blocks, enabling training of very deep networks by alleviating the degradation problem. Widely used vision backbone. |
| **MobileNetV2** | Lightweight architecture built on inverted residuals and linear bottlenecks. Uses depthwise separable convolutions to drastically reduce FLOPs and model size (2.2M params), optimized for mobile/edge deployment. |
| **VGG16** | Classic deep CNN (16 weight layers: 13 conv + 3 FC). Simple uniform architecture of 3×3 convolutions. High parameter count (134M) due to large fully-connected layers. Uses ImageNet pretrained weights. |
| **VGG16 + MHA** | VGG16 backbone with a Multi-Head Attention module inserted after the convolutional feature extractor. Dimension reduction to 128 with 8 attention heads allows the model to capture long-range spatial dependencies absent in vanilla VGG16. |
| **InceptionV3** | Inception network with factorized convolutions (e.g., 7×7 → 1×7 + 7×1), auxiliary classifiers for regularization, and label smoothing. Balances depth and width efficiently at 21.8M parameters. |

---

## Key Observations

1. **DenseNet169 achieves the best overall accuracy (91.56%)** with the fewest trained parameters among top performers, validating its efficient feature reuse mechanism for plant disease classification.

2. **DenseNet201, despite being deeper, does not outperform DenseNet169** — likely due to overfitting on this moderate-sized dataset (3,076 images), suggesting diminishing returns beyond 169 layers.

3. **TensorFlow models consistently outperform PyTorch models**, likely due to different data preprocessing pipelines:
   - TensorFlow: `rescale=1./255` (global normalization)
   - PyTorch: `CustomMinMaxNormalize` (per-image min-max normalization, which can amplify noise in low-contrast images)

4. **MobileNetV2 achieves 87.01% accuracy with only 2.23M parameters** — 1/6th the size of the next smallest architecture. Highly suitable for resource-constrained deployment.

5. **VGG16 + MHA marginally outperforms vanilla VGG16** (+0.49% accuracy, +0.005 MCC) when VGG16 was trained from scratch, but underperforms after VGG16 switched to ImageNet pretrained weights. The attention mechanism may provide less benefit when strong pretrained features are already available.

6. **ImageNet pretrained VGG16 (86.69%) outperforms from-scratch VGG16 (84.90% before fix)** by +1.79% accuracy, confirming transfer learning benefit on this dataset.

7. **All models achieve very high specificity (97.50–98.51%)**, indicating excellent true negative detection across all classes — the models are highly reliable at ruling out incorrect diseases.

8. **The Healthy and Nematode classes** remain the most challenging across all models due to class imbalance (161 and 54 training samples vs. 455–598 for other classes).

---

## Notes

- Specificity for TensorFlow models (DenseNet169, DenseNet201, ResNet50) is estimated from per-class precision/recall/support in classification reports. PyTorch model specificities are extracted from per-epoch validation outputs at the best epoch.
- MCC is reported as percentage (×100) for consistency with other metrics. Standard decimal range is −1 to +1.
