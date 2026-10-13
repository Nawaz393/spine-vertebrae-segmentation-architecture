# ADR 002: Hounsfield Unit Windowing Strategy [100, 1100 HU]

[← ADR Index](README.md) · [← Main README](../../README.md)

- **Status:** Accepted
- **Context:** Medical CT data represents X-ray attenuation in Hounsfield Units (HU), ranging from $-1000\text{ HU}$ (air) to $>+3000\text{ HU}$ (dense surgical implants/cortical bone). Training neural networks on raw wide-dynamic-range HU arrays forces models to allocate capacity to irrelevant soft tissue structures (fat $-100\text{ HU}$, water $0\text{ HU}$, muscle $+40\text{ HU}$), obscuring subtle trabecular-cortical transitions.
- **Decision:** Implement a dedicated bone windowing filter that clamps raw HU values strictly to the interval $[+100\text{ HU}, +1100\text{ HU}]$, followed by dynamic linear min-max normalization into $[0.0, 1.0]$.
- **Benefits:**
  - Completely suppresses air artifacts, paraspinal muscles, adipose tissue, and CSF.
  - Maximizes gradient sensitivity across cortical bone boundaries ($+700\text{ to }+1100\text{ HU}$) and porous trabecular centers ($+150\text{ to }+400\text{ HU}$).
  - Consistent across multi-manufacturer CT scanners (Siemens, GE, Philips).
- **Trade-offs:** Any vertebrae structure presenting radiodensity lower than $+100\text{ HU}$ (severe osteolytic lesions or extreme osteoporosis) is suppressed.
- **Alternatives:**
  - Standard Medical Soft Tissue Windowing ($[-160, 240]\text{ HU}$): Bleaches bone structure into homogeneous white blocks.
  - Unbounded Min-Max Normalization: Suppresses bone contrast due to metallic dental/surgical artifact outliers ($>3000\text{ HU}$).
