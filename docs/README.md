# Documentation Index

[← Main README](../README.md)

This documentation provides an exhaustive architectural, mathematical, and algorithmic breakdown of the **Cervical Vertebrae Segmentation & 3D Volumetric Reconstruction** platform.

---

## Architectural Guides & Scientific Deep Dives

- [System Architecture](architecture.md) — Medical imaging lifecycle, decoupled inference server, and spatial coordinate preservation.
- [Domain Map & System Decomposition](domain-map.md) — Volumetric ingestion, patch generation, PyTorch training, 3D reconstruction, and clinical WebGL review boundaries.
- [Representative Code Examples](code-examples.md) — Production Python and React implementations (NiBabel volume loader, Hounsfield windowing, multi-threaded tiler, PyTorch U-Net, 3D reconstructor).
- [Medical Image Preprocessing & Radiographic Physics](medical-image-preprocessing.md) — Computed Tomography physics, linear X-ray attenuation, Hounsfield Unit calibrations, and 3D median filtering.
- [Deep Learning Architecture & Training Pipeline](deep-learning-and-training.md) — 2D U-Net skip connection topology, combined Dice + BCE loss formulations, 174.88-hour convergence metrics, and validation curves.
- [3D Volumetric Reconstruction & Clinical Validation](3d-volumetric-reconstruction.md) — Sliding-window reverse patch stitching, Gaussian weighted overlap blending, morphological hole filling, and affine-preserving NIfTI export.

---

## Architectural Decision Records (ADRs)

- [ADR Index](decisions/README.md) — Log of design choices and trade-offs.
- [ADR 001: 2D Patch Tiling vs. Full 3D U-Net](decisions/001-patch-based-vs-full-3d-unet.md) — GPU memory constraints vs. sub-millimeter boundary precision.
- [ADR 002: Hounsfield Windowing Strategy](decisions/002-hounsfield-windowing-strategy.md) — Setting the $[100, 1100]\text{ HU}$ radiodensity window for bone isolation.
- [ADR 003: Hybrid Clinical Review Stack](decisions/003-hybrid-clinical-ui-stack.md) — Decoupled Python Flask inference + Node/Express session gateway + React WebGL UI.

---

## Visual Diagrams

- [Diagram Sources](../diagrams/README.md) — Reusable Mermaid source files for pipeline context, CT preprocessing, patch extraction, U-Net architecture, and 3D volumetric reassembly.
