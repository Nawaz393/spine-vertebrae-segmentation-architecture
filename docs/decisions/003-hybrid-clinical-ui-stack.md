# ADR 003: Hybrid Clinical Review Stack (Decoupled Python/Flask + Node/Express + React UI)

[← ADR Index](README.md) · [← Main README](../../README.md)

- **Status:** Accepted
- **Context:** Radiologists and spine surgeons require interactive 2D axial/sagittal/coronal slice navigation and real-time 3D volumetric mesh inspection. Python libraries (`NiBabel`, `PyVista`, `VTK`) excel at numeric tensor computations and NIfTI file processing, but provide subpar web UX. Conversely, modern web browsers with React and WebGL/Three.js offer smooth 60fps rendering, but cannot directly run native C-accelerated medical imaging pipelines.
- **Decision:** Architect a decoupled three-tier system:
  1. **Python / Flask Inference Microservice**: Executes NiBabel ingestion, patch inference, and NIfTI reconstruction.
  2. **Node.js / Express Gateway**: Coordinates clinical sessions, patient metadata caching, and authentication.
  3. **React + Three.js / Cornerstone.js UI**: Renders synchronous multi-planar reconstruction (MPR) and 3D surface meshes in the browser.
- **Benefits:**
  - Isolates heavy deep learning inference workloads from interactive web client sessions.
  - Enables zero-footprint clinical access: clinicians inspect 3D segmentations in web browsers without installing native workstation desktop software.
  - Modular scalability: GPU inference workers can scale independently of the UI gateway.
- **Trade-offs:** Introduces serialization overhead when streaming 3D volumetric meshes across HTTP boundaries (mitigated by binary VTK/GLTF streaming).
- **Alternatives:**
  - Desktop-only C++ / Qt Application (e.g. ITK-SNAP plugin): High barrier to clinical adoption; requires local GPU hardware on every hospital workstation.
  - Monolithic Python Dash / Streamlit App: Lacks performance for responsive 60fps multi-planar volumetric slice manipulation.
