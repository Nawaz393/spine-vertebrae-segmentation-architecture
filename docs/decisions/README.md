# Architecture Decision Records (ADRs)

[← Main README](../../README.md) · [← Documentation Index](../README.md)

This directory documents the core architectural, algorithmic, and biomedical engineering decisions governing the **Cervical Vertebrae Segmentation & 3D Volumetric Reconstruction** platform.

---

## Decision Log

| ID | Title | Status | Decision Summary |
|---|---|---|---|
| [ADR 001](001-patch-based-vs-full-3d-unet.md) | **2D Patch-Based Tiling with 3D Overlap Reassembly vs. Full 3D U-Net** | Accepted | Adopt $128 \times 128$ patch tiling (stride $32 \times 32$) with Gaussian overlap averaging to prevent GPU memory exhaustion while maintaining sub-millimeter spatial resolution. |
| [ADR 002](002-hounsfield-windowing-strategy.md) | **Hounsfield Unit Windowing Strategy [100, 1100 HU]** | Accepted | Constrain radiodensity to $[100, 1100]\text{ HU}$ to suppress non-osseous soft tissues and focus gradient optimization exclusively on trabecular and cortical bone. |
| [ADR 003](003-hybrid-clinical-ui-stack.md) | **Hybrid Clinical Review Stack (Decoupled Python/Flask + Node/Express + React UI)** | Accepted | Decouple heavy Python/PyTorch inference from the interactive React/WebGL clinical frontend via an Express session gateway for zero-install hospital browser access. |
