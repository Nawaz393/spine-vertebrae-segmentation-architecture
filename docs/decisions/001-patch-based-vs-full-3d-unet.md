# ADR 001: 2D Patch-Based Tiling with 3D Overlap Reassembly vs. Full 3D U-Net

[← ADR Index](README.md) · [← Main README](../../README.md)

- **Status:** Accepted
- **Context:** High-resolution 3D full-spine CT scans frequently exceed $512 \times 512 \times 400$ voxels. Processing full 3D isotropic volumes directly through 3D volumetric convolutional networks (3D U-Net / V-Net) requires massive GPU VRAM ($>48\text{ GB}$) and drastically limits batch sizes, leading to unstable gradient estimates, excessive downsampling artifacts, and blurred intervertebral boundary segmentations.
- **Decision:** Adopt a 2D patch-based tiling strategy extracting $128 \times 128$ axial sub-volumes with an overlapping stride of $32 \times 32$, train a high-capacity 2D U-Net on 482,833 patches using AdamW and BCELogitLoss, and reassemble volumetric masks via sliding-window coordinate reverse mapping.
- **Benefits:**
  - High spatial resolution preserved without downsampling voxel density.
  - Scalable to standard workstation hardware and Cloud TPUs during training and clinical inference.
  - Generates 482,833 diverse anatomical training samples (386,266 train / 96,567 val).
  - Achieves empirical performance of **88.46% Dice**, **83.26% Jaccard (IoU)**, **93.81% Sensitivity**, and **91.01% BF-score**, outperforming 8 state-of-the-art segmentation networks.
- **Trade-offs:** Introduces inference overhead during reverse volumetric stitching and requires post-processing to eliminate patch boundary discontinuities.
- **Alternatives:**
  - Full 3D U-Net: Rejected due to GPU memory exhaustion and severe spatial downsampling requirements.
  - Slice-by-slice 2D segmentation without patch tiling: Rejected due to loss of fine local trabecular bone features.
