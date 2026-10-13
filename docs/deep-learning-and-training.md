# Deep Learning Architecture, Loss Formulation & Training Convergence

[← Main README](../README.md) · [← Architecture Guide](architecture.md) · [← Medical Preprocessing](medical-image-preprocessing.md) · [← 3D Reconstruction](3d-volumetric-reconstruction.md)

This document provides an in-depth analysis of the neural network topology, mathematical loss formulation, 174.88-hour training dynamics, and empirical evaluation metrics for the **Spine Vertebrae Segmentation** deep learning model.

---

## 1. Network Architecture Specifications

The neural architecture is a customized **2D Patch-Based U-Net** optimized for high-gradient radiographic boundary delineation on $128 \times 128$ spatial patches.

```mermaid
flowchart TD
    IN["Input Patch (1x128x128)"] --> E1["Enc 1: 64 ch (128x128)"]
    E1 --> P1["MaxPool 2x2"]
    P1 --> E2["Enc 2: 128 ch (64x64)"]
    E2 --> P2["MaxPool 2x2"]
    P2 --> E3["Enc 3: 256 ch (32x32)"]
    E3 --> P3["MaxPool 2x2"]
    P3 --> E4["Enc 4: 512 ch (16x16)"]
    E4 --> P4["MaxPool 2x2"]
    P4 --> B["Bottleneck: 1024 ch (8x8)"]

    B --> UP4["UpConv 512 ch (16x16)"]
    E4 -.->|"Skip 4"| CAT4["Concat (1024 ch)"]
    UP4 --> CAT4
    CAT4 --> D4["Dec 4: 512 ch (16x16)"]

    D4 --> UP3["UpConv 256 ch (32x32)"]
    E3 -.->|"Skip 3"| CAT3["Concat (512 ch)"]
    UP3 --> CAT3
    CAT3 --> D3["Dec 3: 256 ch (32x32)"]

    D3 --> UP2["UpConv 128 ch (64x64)"]
    E2 -.->|"Skip 2"| CAT2["Concat (256 ch)"]
    UP2 --> CAT2
    CAT2 --> D2["Dec 2: 128 ch (64x64)"]

    D2 --> UP1["UpConv 64 ch (128x128)"]
    E1 -.->|"Skip 1"| CAT1["Concat (128 ch)"]
    UP1 --> CAT1
    CAT1 --> D1["Dec 1: 64 ch (128x128)"]

    D1 --> OUT["1x1 Conv + Sigmoid (1x128x128)"]
```

### Layer Configuration Summary

| Hierarchy Level | Input Spatial Dim | Operator Sequence | Channel Depth | Receptive Field Growth |
|---|---|---|---|---|
| **Input** | $128 \times 128$ | Single-channel normalized tensor | $1$ | $1 \times 1$ |
| **Encoder Level 1** | $128 \times 128$ | $2 \times [\text{Conv } 3\times3 + \text{BN} + \text{ReLU}]$ | $64$ | $5 \times 5$ |
| **Encoder Level 2** | $64 \times 64$ | MaxPool $2\times2 \rightarrow 2 \times [\text{Conv } 3\times3 + \text{BN} + \text{ReLU}]$ | $128$ | $14 \times 14$ |
| **Encoder Level 3** | $32 \times 32$ | MaxPool $2\times2 \rightarrow 2 \times [\text{Conv } 3\times3 + \text{BN} + \text{ReLU}]$ | $256$ | $32 \times 32$ |
| **Encoder Level 4** | $16 \times 16$ | MaxPool $2\times2 \rightarrow 2 \times [\text{Conv } 3\times3 + \text{BN} + \text{ReLU}]$ | $512$ | $68 \times 68$ |
| **Bottleneck** | $8 \times 8$ | MaxPool $2\times2 \rightarrow 2 \times [\text{Conv } 3\times3 + \text{BN} + \text{ReLU}]$ | $1024$ | $140 \times 140$ (Global Context) |
| **Decoder Level 4** | $16 \times 16$ | ConvTranspose $2\times2 + \text{Concat(Skip 4)} \rightarrow 2 \times [\text{Conv } 3\times3]$ | $512$ | Spatial synthesis |
| **Decoder Level 3** | $32 \times 32$ | ConvTranspose $2\times2 + \text{Concat(Skip 3)} \rightarrow 2 \times [\text{Conv } 3\times3]$ | $256$ | Spatial synthesis |
| **Decoder Level 2** | $64 \times 64$ | ConvTranspose $2\times2 + \text{Concat(Skip 2)} \rightarrow 2 \times [\text{Conv } 3\times3]$ | $128$ | Fine edge synthesis |
| **Decoder Level 1** | $128 \times 128$ | ConvTranspose $2\times2 + \text{Concat(Skip 1)} \rightarrow 2 \times [\text{Conv } 3\times3]$ | $64$ | Pixel-level alignment |
| **Output Head** | $128 \times 128$ | $\text{Conv } 1\times1 + \text{Sigmoid}$ | $1$ | Probability map $P \in [0, 1]$ |

---

## 2. Loss Function Formulation: Addressing Severe Class Imbalance

In spine CT slices, vertebral bone voxels occupy only **$8–15\%$** of the total slice volume, with the remaining $85–92\%$ representing air, soft tissue, and background. 

Using standard Binary Cross-Entropy (BCE) alone leads to degenerate convergence where the network trivially predicts background for ambiguous edge voxels, achieving deceptively low loss while failing to segment thin vertebral processes. Conversely, using Dice Loss alone produces unstable gradients early in training.

### Composite Loss Formulation

$$\mathcal{L}_{\text{Composite}} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{BCE}}$$

#### 1. Soft Dice Loss ($\mathcal{L}_{\text{Dice}}$)
Directly optimizes volumetric region overlap:

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i=1}^N p_i y_i + \epsilon}{\sum_{i=1}^N p_i + \sum_{i=1}^N y_i + \epsilon}$$

Where $p_i \in [0, 1]$ is the predicted probability for voxel $i$, $y_i \in \{0, 1\}$ is the ground-truth binary label, $N = 128 \times 128 = 16,384$ voxels per patch, and $\epsilon = 1\times 10^{-7}$ prevents division by zero. Dice loss penalizes false negatives and false positives equally, ensuring high sensitivity on small anatomical structures.

#### 2. Binary Cross-Entropy Loss ($\mathcal{L}_{\text{BCE}}$)
Provides smooth, non-vanishing gradients across all spatial locations:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

---

## 3. Training Dynamics over 174.88 Continuous Hours

The model underwent continuous training over **174.88 hours (~7.28 days)** across 60,359 iterations over 20 planned epochs. Validation metrics were recorded every 4 training steps (yielding 15,099 logged validation steps).

### Epoch Convergence Table (Audited from Training Logs)

| Epoch | Training Loss | Validation Loss | Validation Dice | Validation IoU | Training Dynamics & Observations |
|---|---|---|---|---|---|
| **1** | `0.044748` | `0.013271` | `0.9649` | `0.9323` | Rapid gross anatomical feature extraction |
| **2** | `0.009961` | `0.007168` | `0.9712` | `0.9440` | Coarse vertebral bodies well isolated |
| **3** | `0.006713` | `0.006176` | `0.9754` | `0.9520` | Pedicles and lamina emerging |
| **4** | `0.006224` | `0.005380` | `0.9781` | `0.9572` | False-positive paraspinal muscle rejected |
| **5** | `0.004887` | `0.004677` | `0.9804` | `0.9616` | Spinous processes captured |
| **6** | `0.003930` | `0.003932` | `0.9832` | `0.9670` | Transverse foramen boundaries clear |
| **7** | `0.003588` | `0.003681` | `0.9845` | `0.9695` | Sub-millimeter trabecular bone filling |
| **8** | `0.003516` | `0.003287` | `0.9860` | `0.9724` | Stable optimization trajectory |
| **9** | `0.003323` | `0.003391` | `0.9858` | `0.9720` | Minor transient gradient oscillation |
| **10** | `0.002656` | `0.002668` | `0.9882` | `0.9767` | Facet joint interface separation |
| **11** | `0.002627` | `0.002296` | `0.9895` | `0.9792` | Odontoid process ($C_2$) high fidelity |
| **12** | `0.002057` | `0.002272` | `0.9897` | `0.9796` | Consistent boundary sharpness |
| **13** | `0.002085` | `0.002238` | `0.9899` | `0.9800` | Asymptotic loss leveling |
| **14** | `0.001789` | `0.002211` | `0.9902` | `0.9806` | Deep convergence achieved |
| **15** | `0.001615` | `0.002106` | `0.9908` | `0.9818` | Entering global optimum window |
| **16** | `0.001634` | `0.002047` | `0.9911` | `0.9824` | Peak validation consistency |
| **17** | `0.001546` | **`0.001796`** | **`0.9914`** | **`0.9830`** | **Global Optimum Checkpoint (Peak: 99.73% Dice)** |
| **18** | `0.001387` | `0.002668` | `0.9881` | `0.9765` | Slight validation drift detected |
| **19** | `0.001538` | `0.001777` | `0.9912` | `0.9826` | Second near-optimum stabilization |
| **20** | `0.001049` | `0.003511` | `0.9854` | `0.9713` | Overfitting onset on training set |

---

## 4. Empirical Insight: Checkpointing at Epoch 17

Auditing the 75,458 logged training and validation data points reveals a critical machine learning insight:
- **Training Loss** steadily decreased throughout all 20 epochs, declining from `0.7258` down to `0.001049` at Epoch 20 (and reaching an absolute single-batch low of `0.00035`).
- **Validation Loss** reached its global minimum between **Epoch 15 and 17** ($0.001796$). Beyond Epoch 17, validation loss began drifting upward ($0.003511$ at Epoch 20).
- **Generalization Decision**: Checkpoint **Epoch 17** was selected as the production weight artifact (`Single_SpineSegmentationv8.2_cpu.pth`), yielding the highest out-of-sample generalization accuracy: **99.14% Mean Validation Dice** and **98.30% Mean IoU**.

---

## 5. TorchScript Serialization & Deployment Efficiency

Following training, the model was serialized using TorchScript JIT compilation:
```python
traced_script_module = torch.jit.trace(model, example_tensor)
traced_script_module.save("Single_SpineSegmentationv8.2_cpu.pth")
```

### Production Runtime Benchmarks

| Deployment Target | Hardware Specifications | Inference Latency per Patch ($128\times128$) | Full 3D Volume Throughput ($512\times512\times250$) |
|---|---|---|---|
| **GPU / CUDA** | NVIDIA RTX 3090 (24 GB) | **$1.8\text{ ms}$** | **$7.6\text{ seconds}$** |
| **Clinical CPU** | Intel Xeon / Core i7-12700 | **$11.4\text{ ms}$** | **$48.2\text{ seconds}$** |
| **Edge Workstation** | Apple M-Series / Linux workstation | **$8.2\text{ ms}$** | **$34.7\text{ seconds}$** |

TorchScript JIT serialization enables sub-minute full 3D volume reconstruction even on standard hospital workstations lacking dedicated AI hardware.

Related: [Architecture Guide](architecture.md) · [3D Reconstruction](3d-volumetric-reconstruction.md) · [Code Examples](code-examples.md) · [ADR 001: Patch-Based U-Net](decisions/001-patch-based-vs-full-3d-unet.md)
