# Architecture Diagrams

[← Main README](../README.md) · [← Documentation Index](../docs/README.md)

This directory maintains the canonical Mermaid specification files for the Cervical Vertebrae Segmentation and 3D Volumetric Reconstruction architecture. By keeping the pure `.mmd` diagrams decoupled from narrative markdown, these architectural assets remain modular, version-controlled, and directly renderable in CI/CD pipelines and documentation portals.

## Diagram Directory

| Diagram | Source File | Description |
|---|---|---|
| **System Context** | [`system-context.mmd`](system-context.mmd) | End-to-end data lifecycle from raw DICOM/NIfTI ingestion through preprocessing, inference, 3D reconstruction, and clinical WebGL visualization. |
| **Preprocessing Pipeline** | [`preprocessing-pipeline.mmd`](preprocessing-pipeline.mmd) | Radiographic signal processing, Hounsfield Unit bone isolation window `[100, 1100]`, 3D median filtering, and axial slice extraction. |
| **Patch Extraction Flow** | [`patch-extraction-flow.mmd`](patch-extraction-flow.mmd) | Multi-threaded `ThreadPoolExecutor` tiling algorithm generating >850,000 spatial patches with boundary clamping and stride orchestration. |
| **U-Net Architecture** | [`unet-architecture.mmd`](unet-architecture.mmd) | Hierarchical encoder-decoder neural network topology with skip connections, bottleneck operations, and TorchScript JIT serialization. |
| **Reconstruction & Evaluation** | [`reconstruction-and-evaluation.mmd`](reconstruction-and-evaluation.mmd) | Reverse coordinate projection, patch reassembly, 3D affine matrix preservation, morphological cleaning, and validation Dice/IoU scoring. |

## Rendering Guidelines

All `.mmd` files in this directory adhere strictly to Mermaid syntax conventions:
- Node labels containing brackets, parentheses, or arithmetic operators are explicitly double-quoted.
- Directional graphs utilize standard node syntax (`flowchart TB`, `flowchart LR`, `sequenceDiagram`).
- Diagrams can be previewed locally using Mermaid CLI (`mmdc -i <file>.mmd -o <file>.svg`) or directly rendered by GitHub markdown processors.
