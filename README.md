# Time-series Random Process Complexity Ranking Using a Bound on Conditional Differential Entropy

**Accompanying Repository for IZS 2026 Submission**

This repository contains the source code, experiment configurations, and extended derivations for the paper:  
**“Time-series Random Process Complexity Ranking Using a Bound on Conditional Differential Entropy”**  
by *Jacob Ayers et al., Institute of Neuroinformatics, ETH Zürich & University of Zurich*.

---

## Repository Structure

- **`experiment1/`** – Contains the code for *Experiment 1*, which tests the Fang bound on synthetic autoregressive data with **known conditional entropy**.  
- **`experiment2/`** – Contains the code for *Experiment 2*, which applies the Fang bound to synthetic bio-inspired time series with **unknown entropy but known complexity ordering**.  
- **`IZS_Fang_Extended_Proof.pdf`** – Extended derivations and theoretical details expanding on Section II of the paper.  
- Each experiment folder includes its own `README.md` with setup, training, and figure reproduction instructions.


---

## Hardware Specifications

All experiments were run on a dual-socket server with:
- Two **AMD EPYC 7H12 (64 cores each)**
- **2 TB DDR4-3200 ECC RAM**
- **Two Nvidia Quadro RTX 8000 GPUs (48 GB VRAM each)**

Results should still be reproducible on smaller machines for reduced dataset sizes.

---

