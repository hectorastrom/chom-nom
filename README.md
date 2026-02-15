# One-Click-Compress

Shrink the size of any PyTorch model in a single click!

## Project Goal

This project aims to build a **one-click universal model compression tool** designed to shrink PyTorch models for real-time edge deployment on hardware like the Raspberry Pi 4. While the Pi 4 can hold 100–500MB models in its 2–8GB RAM, we target a post-compression footprint of **50MB or less** to guarantee low-latency inference. 

We are prioritizing a pipeline that balances aggressive size reduction with performance stability, starting with **INT8 Quantization** as our primary lever.

---

## The Four Levels of Compression

1.  **Quantization (INT8):** Our starting point. We utilize **Quantization-Aware Training (QAT)** on a provided dataset to secure massive size reductions while preserving accuracy by simulating quantization errors during fine-tuning.
2.  **Structural Pruning:** Unlike unstructured methods, this physically removes network blocks (entire filters or channels). This is the only pruning strategy that genuinely reduces RAM usage and compute operations on ARM architectures.
3.  **Low-Rank Factorization:** This stage uses the best **Rank-R Approximation** in the Frobenius norm to decompose large weight matrices into smaller, more efficient products.
4.  **Logit Distillation:** As a final safety measure, we use **KL Divergence** to align the compressed student model's logits with the original teacher model, recovering accuracy lost during the previous three stages.

---

## Technical Constraints & Hardware Notes

* **Target Hardware:** Raspberry Pi 4 / ARM Cortex-A72.
* **Inference Footprint:** < 50MB for real-time performance.
* **Agnostic Design:** The goal is to build a process compatible with any model
  architecture, including bespoke ones, without requiring external libraries.

---

Built by:
- Christina Lee
- Danny Lin
- Albert Astrom
- Hector Astrom

*for treehacks 2026*
