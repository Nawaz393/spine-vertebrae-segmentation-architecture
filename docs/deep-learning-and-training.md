# Deep Learning Architecture, Loss Formulation & Experimental Results

[← Main README](../README.md) · [← Architecture Guide](architecture.md) · [← Medical Preprocessing](medical-image-preprocessing.md) · [← 3D Reconstruction](3d-volumetric-reconstruction.md)

This document provides a comprehensive analysis of the neural network topology, mathematical evaluation formulas, training hyperparameters, and empirical evaluation metrics for the **Patch-Based U-Net Vertebrae Segmentation** platform.

---

## 1. Network Architecture Specifications

The model is a customized **2D Patch-Based U-Net** designed specifically for high-precision boundary localization on $128 \times 128$ overlapping CT patches.

```mermaid
flowchart TD
    IN["Input Patch (1x128x128)"] --> E1["Conv 3x3 (pad 1, 64 ch) + ReLU"]
    E1 --> P1["MaxPool 2x2"]
    P1 --> E2["Conv 3x3 (128 ch) + ReLU"]
    E2 --> P2["MaxPool 2x2"]
    P2 --> E3["Conv 3x3 (256 ch) + ReLU"]
    E3 --> P3["MaxPool 2x2"]
    P3 --> B["Bottleneck Convolutions"]

    B --> UP3["Up-Sampling Transposed Conv"]
    E3 -.->|"Skip Connection 3"| CAT3["Concatenation (Feature Fusion)"]
    UP3 --> CAT3
    CAT3 --> D3["Decoder Conv Layers"]

    D3 --> UP2["Up-Sampling Transposed Conv"]
    E2 -.->|"Skip Connection 2"| CAT2["Concatenation (Feature Fusion)"]
    UP2 --> CAT2
    CAT2 --> D2["Decoder Conv Layers"]

    D2 --> UP1["Up-Sampling Transposed Conv"]
    E1 -.->|"Skip Connection 1"| CAT1["Concatenation (Feature Fusion)"]
    UP1 --> CAT1
    CAT1 --> D1["Decoder Conv Layers"]

    D1 --> OUT["Conv 1x1 (2 Classes: Bone vs. Background)"]
```

### Architectural Details
- **Encoder Path**:
  - The initial convolutional block receives $128 \times 128$ patches, applying $3 \times 3$ kernels with padding=1 to preserve spatial dimensions, generating 64 feature channels followed by Rectified Linear Unit (ReLU) activation.
  - Successive convolutional blocks double the channel depth ($128 \rightarrow 256$ channels), each followed by $2 \times 2$ max-pooling downsampling layers to build translation-invariant receptive fields.
- **Decoder Path**:
  - Mirrored up-sampling layers expand feature map resolution via transposed convolutions.
  - Skip connections concatenate high-resolution feature maps from the contracting encoder directly into the expansive decoder, restoring fine localized spatial geometry lost during pooling.
- **Output Head**:
  - A final $1 \times 1$ convolution maps the feature representations to a binary output mask (vertebrae vs. background).

---

## 2. Training Setup & Hyperparameters

- **Dataset Scale**: **482,833 total patches** generated via a 32-pixel sliding stride.
- **Split Ratio**:
  - **Training Set (80%)**: **386,266 patches**
  - **Validation Set (20%)**: **96,567 patches**
- **Independent Test Evaluation**: **5 unseen clinical test cases containing 3,418 slices**.
- **Mini-Batch Size**: 128
- **Initial Learning Rate**: $0.0003$ ($3 \times 10^{-4}$) with adaptive learning rate scheduling.
- **Optimizer**: **AdamW** with zero bias initialization.
- **Loss Function**: **Binary Cross-Entropy with Logits (`BCELogitLoss`)**.
- **Epochs & Convergence**:
  - Trained over **20 epochs** (~48 hours on Google Cloud v3-8 TPU).
  - The optimal model checkpoint was selected at **Epoch 15**, which achieved minimal validation loss. Subsequent epochs showed minor loss plateauing/divergence.

---

## 3. Mathematical Evaluation Measures

### 3.1 Classification Metrics
To evaluate the patch-based classification capacity across the 482,833 patch dataset:

1. **Specificity (SPE)**: Measures accuracy in rejecting background/non-bone patches.
   $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$
2. **Precision (PRE)**: Measures the proportion of predicted bone patches that are true bone.
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
3. **Accuracy (ACC)**: Overall classification correctness.
   $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
4. **F-Score / F1**: Harmonic balance between precision and recall.
   $$\text{F-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 3.2 Volumetric Segmentation Metrics
To evaluate full slice-by-slice vertebrae boundary overlap across the 3,418 test slices:

1. **Sensitivity (SEN / Recall)**: Detects true vertebral bone pattern coverage.
   $$\text{Sensitivity} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
2. **Jaccard Index (JAC / IoU)**: Spatial intersection over union.
   $$\text{JAC} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$
3. **Dice Similarity Coefficient (DSC)**: Volume overlap metric.
   $$\text{DSC} = \frac{2 \times \text{TP}}{2 \times \text{TP} + \text{FP} + \text{FN}}$$
4. **Boundary F-Score (BF-score)**: Quantifies the contour alignment between predicted and ground-truth anatomical boundaries.
   $$\text{BF-score} = \frac{2 \times \text{Precision}_{\text{contour}} \times \text{Recall}_{\text{contour}}}{\text{Precision}_{\text{contour}} + \text{Recall}_{\text{contour}}}$$

---

## 4. Concrete Empirical Findings

### 4.1 Patch-Level Classification Results
- **Accuracy (ACC)**: **98.46%**
- **Specificity (SPE)**: **98.51%**
- **Precision (PRE)**: **88.37%**
- **F-Score**: **88.46%**

*High accuracy and specificity confirm that non-vertebral tissues (paraspinal muscle, CSF, lungs, air) are reliably rejected without generating false positives.*

### 4.2 Full Segmentation Performance (3,418 Slices across 5 Patients)
- **Dice Similarity Coefficient (DSC)**: **88.46%**
- **Jaccard Index (JAC / IoU)**: **83.26%**
- **Boundary F-Score (BF-score)**: **91.01%**
- **Sensitivity (SEN)**: **93.81%**

---

## 5. Comparative SOTA Benchmark (Table 2 from Research)

Comparison against published segmentation architectures evaluated under identical clinical protocols:

| Category | Model | SEN (%) | BF-Score (%) | JAC (%) | DSC (%) |
|---|---|:---:|:---:|:---:|:---:|
| **Segmentation Networks** | Standard U-Net (Ronneberger et al.) | 79.40 | 81.40 | 71.90 | 83.70 |
| | D-TVNet (Lyu et al.) | 84.90 | 78.20 | 76.50 | 86.70 |
| | SpineParseNet (Cheng et al.) | 88.00 | 87.60 | 77.50 | 87.30 |
| | MultiResUNet | 87.50 | 89.10 | 73.80 | 84.90 |
| **Classification-Based Networks** | PaDBN (Qadri et al.) | 89.70 | 84.90 | 75.60 | 86.10 |
| | VGG Net (Simonyan & Zisserman) | 90.10 | 78.80 | 70.00 | 81.90 |
| | ResNet (He et al.) | 85.40 | 83.70 | 73.20 | 84.50 |
| | DenseNet (Huang et al.) | 77.90 | 80.00 | 77.60 | 87.40 |
| **Proposed Method** | **UNet (Proposed)** | **93.81%** | **91.01%** | **83.26%** | **88.46%** |

### Benchmark Analysis:
1. **Vs. Standard Whole-Image U-Net**: Standard U-Net takes full $512 \times 512$ slices in a single pass, losing local high-frequency boundary textures. Our patch-based approach improves DSC from **83.70% to 88.46% (+4.76%)** and Jaccard from **71.90% to 83.26% (+11.36%)**.
2. **Vs. Multi-Stage Models (SpineParseNet & D-TVNet)**: D-TVNet suffers from sensitivity to noisy edge pixels (78.20% BF-score), while SpineParseNet achieves 87.30% DSC. Our approach surpasses SpineParseNet by **+1.16% in DSC** and **+5.76% in JAC**.
3. **Vs. Classification Networks (PaDBN, VGG, ResNet)**: Classification-based segmentation models (like PaDBN at 86.10% DSC) lack the dense pixel-to-pixel skip connections inherent in U-Net, resulting in degraded boundary sensitivity.

Related: [Architecture](architecture.md) · [Preprocessing Deep Dive](medical-image-preprocessing.md) · [3D Reconstruction](3d-volumetric-reconstruction.md)
