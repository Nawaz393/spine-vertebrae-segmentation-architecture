# Documentation Index

[← Main README](../README.md)

This documentation provides an in-depth breakdown of the architecture, data pipeline, deep learning models, and clinical visualization tooling built for the **3D Spine CT Vertebrae Segmentation & Volumetric Reconstruction** platform.

---

## Architecture & Deep Dives

- [System Architecture](architecture.md) — Volumetric image processing lifecycle, memory management, and decoupled inference server design.
- [Domain Map & System Decomposition](domain-map.md) — How the system is structured across volume ingestion, patch generation, model inference, 3D reconstruction, and clinical review.
- [Representative Code Examples](code-examples.md) — Clean, production-ready Python and React snippets extracted directly from the codebase (NiBabel I/O, HU windowing, multi-threaded tiler, PyTorch U-Net, 3D stitcher).
- [Medical Image Preprocessing & CT Physics](medical-image-preprocessing.md) — CT physics, X-ray attenuation, the Hounsfield Unit scale, and isolating vertebral bone from soft tissue.
- [Deep Learning Architecture & Training Pipeline](deep-learning-and-training.md) — 2D U-Net skip connection topology, BCELogitLoss optimization via AdamW, training dynamics across 482,833 patches on Cloud TPU v3-8, and quantitative evaluation metrics.
- [3D Volumetric Reconstruction & Clinical Validation](3d-volumetric-reconstruction.md) — Sliding-window reverse patch stitching, Gaussian weighted overlap blending, morphological post-processing, and affine coordinate matrix restoration into clinical `.nii` volumes.

---

## Architecture Decision Records (ADRs)

- [ADR Index](decisions/README.md) — Summary log of key architectural decisions.
- [ADR 001: 2D Patch Tiling vs. Full 3D U-Net](decisions/001-patch-based-vs-full-3d-unet.md) — Balancing GPU memory limits with sub-millimeter boundary precision.
- [ADR 002: Hounsfield Windowing Strategy](decisions/002-hounsfield-windowing-strategy.md) — Why we chose the $[+100, +1100]\text{ HU}$ window for vertebral bone contrast.
- [ADR 003: Hybrid Clinical Review Stack](decisions/003-hybrid-clinical-ui-stack.md) — Decoupled Python/Flask inference + Node/Express session gateway + React/WebGL UI for zero-install hospital browser access.

---

## Diagrams

- [Diagram Sources](../diagrams/README.md) — Version-controlled Mermaid diagram definitions for system context, preprocessing, patch extraction, U-Net architecture, and 3D volumetric reassembly.
