# ADR 001: 2D Patch-Based Tiling with 3D Overlap Reassembly vs. Full 3D U-Net

[← ADR Index](README.md) · [← Main README](../../README.md)

- **Status:** Accepted
- **Context:** High-resolution 3D full-spine CT scans frequently exceed $512 \times 512 \times 400$ voxels. Processing full 3D isotropic volumes directly through 3D volumetric convolutional networks (3D U-Net / V-Net) requires massive GPU VRAM ($>48\text{ GB}$) and drastically limits batch sizes, leading to unstable gradient estimates, excessive downsampling artifacts, and blurred intervertebral boundary segmentations.
- **Decision:** Adopt a 2D patch-based tiling strategy extracting $128 \times 128$ axial sub-volumes with an overlapping stride of $32 \times 32$, train a high-capacity 2D U-Net on over 850,000 augmented patches, and reassemble volumetric masks via sliding-window Gaussian overlap averaging.
- **Benefits:**
  - High spatial resolution preserved without downsampling voxel density.
  - Scalable to consumer/workstation NVIDIA GPUs during training and clinical inference.
  - Generates >850k diverse anatomical training samples, eliminating overfitting.
  - Achieves empirical performance of **99.14% Dice** and **98.30% IoU**.
- **Trade-offs:** Introduces inference overhead during reverse volumetric stitching and requires post-processing to eliminate patch boundary discontinuities.
- **Alternatives:**
  - Full 3D U-Net: Rejected due to GPU memory exhaustion and severe spatial downsampling requirements.
  - Slice-by-slice 2D segmentation without patch tiling: Rejected due to loss of fine local trabecular bone features.
