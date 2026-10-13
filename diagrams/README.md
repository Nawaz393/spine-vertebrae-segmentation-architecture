# Diagram Sources

[← Main README](../README.md) · [← Documentation Index](../docs/README.md)

This directory contains standalone, reusable Mermaid diagram source files for the **3D Spine CT Vertebrae Segmentation & Volumetric Reconstruction** architecture:

- [System Context](system-context.mmd) — High-level pipeline from raw CT volumes to clinical WebGL rendering.
- [Preprocessing Pipeline](preprocessing-pipeline.mmd) — HU windowing, spatial median filtering, and dynamic normalization.
- [Patch Extraction Flow](patch-extraction-flow.mmd) — Multi-threaded 2D axial tiling generating 482,833 training patches.
- [U-Net Architecture](unet-architecture.mmd) — Contracting encoder, expansive decoder, and identity skip connection topology.
- [3D Reconstruction & Evaluation](reconstruction-and-evaluation.mmd) — Sliding-window reverse patch stitching, Gaussian overlap blending, and Dice/IoU evaluation.

GitHub renders Mermaid natively in Markdown. Keeping pure `.mmd` files ensures diagrams can be easily embedded, updated, or compiled in CI/CD pipelines.
