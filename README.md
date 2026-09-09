# Generalizable and Per-Scene NeRF Architectures under Sparse-View Conditions: A Systematic Benchmarking Methodology

[![Paper](https://img.shields.io/badge/SIBGRAPI-2026-blue.svg)](docs/sibgrapi-2026/main.pdf)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Nerfstudio](https://img.shields.io/badge/Nerfstudio-Integrated-FF6B6B.svg)](https://docs.nerf.studio/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)

This repository hosts the official benchmarking infrastructure, experiment pipelines, and integration wrappers presented in the study:

> **"Generalizable and Per-Scene NeRF Architectures under Sparse-View Conditions: A Systematic Benchmarking Methodology"**  
> *Felipe G. Silva, Leandro A. Neves, Guilherme F. Roberto, Thiago Pozati, Domingos Oliveira, Adriano B. Silva, Thaína A. A. Tosta, Marcelo Z. do Nascimento.*  
> **SIBGRAPI 2026** (Conference on Graphics, Patterns and Images).

---

## Overview

Neural Radiance Fields (NeRFs) have redefined novel view synthesis by learning continuous 5D volumetric representations from multi-view imagery. Despite impressive visual quality under densely sampled inputs, real-world deployment in resource-constrained settings (e.g., casual mobile scanning, medical imaging, and virtual try-on) fundamentally confronts **sparse-view acquisition constraints** ($N \le 10$ views).

Under such constraints, existing neural rendering paradigms diverge:
1. **Per-Scene Optimization Models** (e.g., *Instant-NGP*, *Nerfacto*): Optimize continuous scenes from scratch using multi-scale hash tables and lightweight MLPs. While computationally agile, they lack structural priors and are prone to geometric degradation when ray coverage is sparse.
2. **Generalizable Priors** (e.g., *PixelNeRF*, *Generalizable NeRF Transformer / GNT*): Infer volumetric representations across unseen scenes by conditioning rendering on pre-trained 2D CNN or Vision Transformer (ViT) feature backbones, optionally adapting weights via test-time fine-tuning (FT).

Although numerous architectures exist, prior evaluations have remained fragmented—often isolated to disparate training splits, unstandardized ray budgets, or hardware configurations. This repository establishes a **standardized, fully reproducible benchmarking methodology** to systematically investigate the Pareto trade-offs governing visual fidelity, runtime convergence, rendering latency, and memory consumption across these paradigms.

---

## Key Contributions

* **Unified Benchmarking Protocol:** Standardized data partitions on the forward-facing Local Light Field Fusion (LLFF) dataset, controlled ray budgets (512 training rays, 128 evaluation rays), and reproducible camera sampling regimes across $N \in \{3, 6, 10\}$ views.
* **Modular Nerfstudio Integration Wrappers:** Custom, open-source integration wrappers extending the Nerfstudio framework with PixelNeRF and GNT, distributed directly via PyPI (`nerfstudio-pixelnerf`, `nerfstudio-gnt`) and GitHub.
* **Pareto Frontier Characterization:** Joint quantitative profiling of perceptual quality (PSNR, SSIM, LPIPS via VGG features) and computational metrics (training/FT convergence time, inference FPS, and dynamic peak VRAM footprint) on consumer-grade hardware.

---

## Quantitative Findings

Experiments were conducted on the 8 real-world scenes of the LLFF dataset at downscale factor 8 ($504 \times 378$ resolution) using an isolated NVIDIA GeForce GTX 1660 Super (6 GB VRAM) running CUDA 12.6 inside containerized Docker environments.

### Consolidated Global Results

The table below summarizes the global averages across all evaluated LLFF scenes under the standardized $N=10$ sparse-view regime (and zero-shot inference for generalizable models):

| Model | Paradigm / Regime | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ | LPIPS (VGG) $\downarrow$ | Train / FT Time $\downarrow$ | Inference (FPS) $\uparrow$ | Peak VRAM (GB) $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PixelNeRF** | Generalizable (Zero-Shot) | 6.62 | 0.093 | 0.814 | $\sim$22 s | 0.004 | 1.93 |
| **PixelNeRF** | Generalizable (Fine-Tuning) | 6.38 | 0.071 | 0.801 | 4 min 23 s | 0.006 | 1.97 |
| **GNT** | Generalizable (Zero-Shot) | 10.73 | 0.300 | 0.760 | **$\sim$9 s** | 0.014 | 1.94 |
| **GNT** | Generalizable (Fine-Tuning) | 10.42 | 0.336 | 0.724 | 7 min 19 s | 0.015 | 1.93 |
| **Instant-NGP** | Per-Scene ($N=10$) | 14.06 | 0.442 | 0.498 | 31 min 51 s | 0.199 | **1.09** |
| **Nerfacto** | Per-Scene ($N=10$) | **27.11** | **0.875** | **0.171** | 8 min 57 s | **0.388** | 1.43 |

### Core Empirical Insights

1. **Reconstruction Ceiling:** Per-scene models (specifically Nerfacto) establish the visual quality upper bound, achieving **27.11 dB PSNR** and **0.875 SSIM**, reliably recovering high-frequency geometric structures even when trained on only 10 sparse views.
2. **Inference Rendering Bottlenecks:** While generalizable architectures bypass per-scene optimization (rendering novel scenes in seconds under Zero-Shot), their volumetric feature aggregation across multi-view projections creates severe inference latency bottlenecks ($\le 0.015$ FPS). In contrast, hash-encoded per-scene representations render between **$13\times$ (Instant-NGP vs. GNT)** and **$97\times$ (Nerfacto vs. PixelNeRF)** faster.
3. **Memory Footprint:** Explicit hash table representations constrain dynamic VRAM peak allocation during inference down to **1.09 GB** for Instant-NGP and **1.43 GB** for Nerfacto, well within low-tier GPU envelopes.
4. **Generalization Breakdown under Extreme Sparsity:**
   * *PixelNeRF* exhibits severe isotropic blurring, as 2D CNN backbones fail to accurately triangulate epipolar features when view baseline disparity is wide and $N \le 10$.
   * *GNT* achieves higher structural coherence through cross-attention, but keeping feature extraction backbones frozen during test-time adaptation bounds generalization against out-of-distribution geometry, occasionally inducing local performance plateaus.
5. **View Sampling Dynamics:** Uniform view selection along the camera trajectory out-performs unconstrained random sampling by **+0.54 dB PSNR** on the *fern* scene, underscoring the critical role of baseline distribution in sparse reconstruction.

---

## Repository Structure

```text
generalizable-nerf-benchmark/
├── assets/                       # Data and experiment logs (generated at runtime)
│   ├── data/baseline/            # LLFF dataset scenes
│   └── logs/                     # Consolidated metrics.csv and checkpoint logs
├── docs/
│   ├── entrega-mestrado/         # Thesis documentation
│   └── sibgrapi-2026/            # Complete LaTeX paper, figures, and bib source
├── scripts/                      # Post-processing, aggregation, and plotting utilities
│   ├── aggregate_metrics.py      # Aggregates metrics across all scene logs
│   ├── generate_article_tables.py# Formats Markdown and LaTeX result tables
│   ├── generate_selected_plots.py# Generates publication radar charts and boxplots
│   ├── generate_lr_ablation.py   # Learning rate ablation analysis
│   └── generate_30k_ablation.py  # 30,000-iteration adaptation ablation analysis
├── src/                          # Modular benchmarking pipeline
│   ├── dataset/
│   │   ├── ImageDataset.py       # LLFF loader, downscaling, and view partitioner
│   │   └── SubsetWrapper.py      # Uniform and random sparse subset sampler
│   ├── nerf/
│   │   └── NerfModel.py          # Unified Nerfstudio wrapper for PixelNeRF, GNT, NGP, Nerfacto
│   ├── utils/
│   │   ├── config.py             # Global execution configurations
│   │   ├── metrics.py            # MetricsLogger (PSNR, SSIM, LPIPS, VRAM, FPS)
│   │   └── types.py              # Typings, Enums, and metric identifiers
│   └── validation/
│       └── eval_images.py        # Per-image evaluation routines
├── Dockerfile                    # Isolated container recipe (CUDA 12.6, COLMAP, tiny-cuda-nn)
├── docker-compose.yml            # Container service and volume bindings
├── export_delivery.sh            # Script to package artifact zip without cache
├── install.sh                    # Automated dependency and dataset setup script
├── main.py                       # Primary experiment runner and entrypoint
├── pyproject.toml                # UV package specification and dependencies
├── run_docker.sh                 # Docker launcher with auto-detected GPU architecture
└── uv.lock                       # Deterministic dependency lockfile
```

---

## Installation & Reproduction

The benchmark is designed for fully reproducible execution using Docker with NVIDIA Container Toolkit, isolating system libraries, CUDA kernels, and tiny-cuda-nn bindings.

### Prerequisites

* Linux (Ubuntu 22.04 / 24.04 recommended)
* NVIDIA GPU (Compute Capability $\ge 7.5$, $\ge 6$ GB VRAM)
* NVIDIA Driver $\ge 525$ and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
* Docker & Docker Compose

### 1. Automated Setup & Dataset Download

Run the automated installer script:

```bash
chmod +x install.sh run_docker.sh export_delivery.sh
./install.sh
```

Alternatively, download the LLFF dataset manually via the Kaggle CLI:

```bash
mkdir -p assets/data/baseline/
kaggle datasets download -d arenagrenade/llff-dataset-full -p assets/data/baseline/ --unzip
```

### 2. Execution via Docker (Recommended)

Launch the containerized experiment suite with automatic GPU architecture detection:

```bash
bash run_docker.sh
```

Or invoke Docker Compose directly:

```bash
GPU_ARCH=86 docker compose up --build
```

The container mounts `./assets/data`, `./assets/logs`, `./src`, and `./main.py`, logging metrics directly to disk.

### 3. Local Execution via `uv`

If running natively outside Docker with pre-installed CUDA 12.x and COLMAP 3.10:

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Synchronize environment dependencies
uv sync --prerelease=allow

# Install tiny-cuda-nn bindings
uv pip install ninja "setuptools<70.0.0" wheel
uv pip install -v git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation

# Run the benchmark pipeline
python main.py
```

---

## Post-Processing & Result Generation

All analysis scripts are located in `scripts/` and operate directly on the experiment logs produced under `assets/logs/`:

```bash
# Consolidate and inspect global averages across all scenes and models
python scripts/aggregate_metrics.py

# Generate publication Markdown and LaTeX comparison tables
python scripts/generate_article_tables.py

# Generate visual trade-off radar charts and PSNR boxplots
python scripts/generate_selected_plots.py

# Inspect learning rate ablation tables
python scripts/generate_lr_ablation.py

# Inspect 30,000-step training ablation tables
python scripts/generate_30k_ablation.py
```

---

## External Wrappers & Packages

As part of this research, custom integration wrappers for Nerfstudio were implemented and published to enable reproducible benchmarking:

* **Wrapper Source Code:** [https://github.com/Felipe-gsilva/Few-Shot-NeRFs/](https://github.com/Felipe-gsilva/Few-Shot-NeRFs/)
* **`nerfstudio-gnt`:** [PyPI](https://pypi.org/project/nerfstudio-gnt/) | [piwheels](https://www.piwheels.org/project/nerfstudio-gnt/)
* **`nerfstudio-pixelnerf`:** [PyPI](https://pypi.org/project/nerfstudio-pixelnerf/) | [piwheels](https://www.piwheels.org/project/nerfstudio-pixelnerf/)

---

## Citation

If you utilize this benchmark, codebase, or the integration wrappers in your research, please cite our paper:

```bibtex
@inproceedings{silva2026generalizable,
  title     = {Generalizable and Per-Scene NeRF Architectures under Sparse-View Conditions: A Systematic Benchmarking Methodology},
  author    = {Silva, Felipe G. and Neves, Leandro A. and Roberto, Guilherme F. and Pozati, Thiago and Oliveira, Domingos and Silva, Adriano B. and Tosta, Tha{\'i}na A. A. and do Nascimento, Marcelo Z.},
  booktitle = {Proceedings of the Conference on Graphics, Patterns and Images (SIBGRAPI)},
  year      = {2026},
  publisher = {SBC},
  address   = {S{\~a}o Paulo, Brazil}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
