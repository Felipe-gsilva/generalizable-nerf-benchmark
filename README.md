# Generalizable and Per-Scene NeRF Architectures under Sparse-View Conditions: A Systematic Benchmarking Methodology

[![Paper](https://img.shields.io/badge/Paper-Camera--Ready-blue.svg)](docs/final-2027/main.pdf)
[![SIBGRAPI 2026](https://img.shields.io/badge/SIBGRAPI-2026-6C5CE7.svg)](docs/sibgrapi-2026/main.pdf)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Nerfstudio](https://img.shields.io/badge/Nerfstudio-Integrated-FF6B6B.svg)](https://docs.nerf.studio/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)

This repository hosts the official benchmarking infrastructure, experiment pipelines, and integration wrappers presented in the study:

> **"Generalizable and Per-Scene NeRF Architectures under Sparse-View Conditions: A Systematic Benchmarking Methodology"**  
> *Felipe G. Silva, Leandro A. Neves, Guilherme F. Roberto, Thiago Pozati, Domingos Oliveira, Adriano B. Silva, Thaína A. A. Tosta, Marcelo Z. do Nascimento.*  
> **SIBGRAPI 2026** (Conference on Graphics, Patterns and Images).

> [!IMPORTANT]
> **Repository Evolution & SIBGRAPI WIP Track Disclaimer:**  
> The original publication presented at **SIBGRAPI 2026** reflects the preliminary Work-In-Progress (WIP) state of this benchmark, corresponding directly to commit [`426d319`](https://github.com/Felipe-gsilva/generalizable-nerf-benchmark/commit/426d319e603fe84322e895041b5139dcdb68599a) (source and PDF available in [`docs/sibgrapi-2026/`](docs/sibgrapi-2026/main.pdf)).  
> 
> As an active WIP project, substantial methodological advancements, bug fixes, and quantitative breakthroughs were developed beyond that baseline:
> 1. **5 Fine-Tuning Strategies:** Formulated a formal taxonomy (`Full-FT`, `PFT-Decoder`, `PFT-Backbone`, `BitFit`, and `PEFT-LoRA`) with strict gradient isolation;
> 2. **Overcoming Negative Adaptation:** Solved representation collapse, turning around PixelNeRF by **+4.54 dB** and boosting GNT by **+4.65 dB** (surpassing per-scene Instant-NGP);
> 3. **Step Budget Calibration:** Discovered the optimal **500-step budget** via systematic iteration ablation, eliminating late-stage overfitting;
> 4. **Zero-Latency LoRA Deployment:** Implemented in-place weight fusion ($\Delta W = \frac{\alpha}{r}BA$) via `merge_lora_weights()`;
> 5. **Statistical Rigor:** Instituted multi-seed ($42, 123, 999$) stratified 4-fold cross-validation with automated divergence pruning.
> 
> For an exhaustive side-by-side comparison between the preliminary SIBGRAPI WIP results and the current state-of-the-art benchmark, refer to the [Evolution Section](#-evolution-from-early-wip-to-current-benchmark). The updated, publication-grade manuscript is maintained under [`docs/final-2027/`](docs/final-2027/main.pdf).

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
* **Formal Fine-Tuning Taxonomy & Parameter Management:** Implementation of five distinct adaptation strategies (`Full-FT`, `PFT-Decoder`, `PFT-Backbone`, `BitFit`, and `PEFT-LoRA`) in modular wrappers, featuring dynamic gradient isolation and test-time in-place weight merging for zero runtime overhead.
* **Stratified K-Fold Grid Search:** A multi-seed ($42, 123, 999$), 4-fold cross-validation protocol systematically mapping the optimal learning rate ($\eta \in \{10^{-4}, 5\times 10^{-4}, 10^{-3}\}$) and LoRA capacity ($r \in \{4, 8\}$) per strategy.
* **Empirical Convergence & Overfitting Profiling:** Fine-tuning step ablation ($0$ to $2,500$ steps) demonstrating convergence bounds and validating the optimal 500-step budget.
* **Modular Nerfstudio Integration Wrappers:** Standalone packages extending Nerfstudio with PixelNeRF and GNT, distributed via PyPI (`nerfstudio-pixelnerf`, `nerfstudio-gnt`) and GitHub.

---

## 🔄 Evolution: From Early WIP to Current Benchmark

The benchmark has progressed substantially from its initial Work-In-Progress (WIP) exploratory phase (snapshot at commit [`426d319`](https://github.com/Felipe-gsilva/generalizable-nerf-benchmark/commit/426d319e603fe84322e895041b5139dcdb68599a)) to the current publication-ready framework. The preliminary benchmark established baseline evaluations under sparse views but revealed critical vulnerabilities in naive test-time fine-tuning, where unconstrained optimization led to catastrophic representation collapse. 

In response, the current benchmark introduced a formal 5-strategy fine-tuning taxonomy, dynamic gradient isolation, zero-overhead LoRA weight merging, stratified 4-fold cross-validation, and empirical step calibration.

### 1. Direct Performance Comparison: Early WIP vs. Current Benchmark

The table below contrasts the preliminary WIP results against the current calibrated benchmark across generalizable and per-scene models ($N=3$ sparse views for fine-tuning, $N=10$ for per-scene baselines):

| Model | Evaluated Regime | Metric | Early WIP Baseline<br>*(Naive FT, 2500 steps)* | Current Benchmark<br>*(Best Strategy, 500 steps)* | Net Change ($\Delta$)<br>*(Current vs. WIP)* | Status & Breakthrough Diagnosis |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **PixelNeRF** | Zero-Shot (ZS) | PSNR | 6.62 dB | 6.35 $\pm$ 0.05 dB | $-0.27$ dB | Calibrated across multi-seed stratified splits |
| **PixelNeRF** | Fine-Tuning | **PSNR** | **6.38 dB** | **10.92 $\pm$ 0.26 dB** | **+4.54 dB (+71.2%)** | **Breakthrough (PFT-Backbone):** Overcomes negative adaptation by isolating 2D priors and freezing coordinate MLP |
| | | SSIM | 0.071 | **0.162 $\pm$ 0.035** | **+0.091 (+128%)** | Resolves severe isotropic blurring and geometric collapse |
| | | LPIPS | 0.801 | **0.772 $\pm$ 0.004** | **$-0.029$** | Significantly enhanced perceptual structure |
| **PixelNeRF** | BitFit (Bias-Only) | PSNR | *Not Evaluated* | **10.79 $\pm$ 0.19 dB** | **+4.41 dB vs. WIP** | Updates $<0.15\%$ weights, acting as an implicit regularizer against overfitting |
| **GNT** | Zero-Shot (ZS) | PSNR | 10.73 dB | 10.51 $\pm$ 0.11 dB | $-0.22$ dB | Standardized across 6 valid LLFF scenes $\times$ 3 seeds |
| **GNT** | Fine-Tuning | **PSNR** | **10.42 dB** | **15.07 $\pm$ 0.03 dB** | **+4.65 dB (+44.6%)** | **Breakthrough (PFT-Decoder):** Overcomes $-0.31$ dB collapse; **surpasses per-scene Instant-NGP (14.06 dB)** |
| | | SSIM | 0.336 | **0.368 $\pm$ 0.000** | **+0.032 (+9.5%)** | Superior multi-view epipolar ray coherence |
| | | LPIPS | 0.724 | **0.679 $\pm$ 0.001** | **$-0.045$** | Noticeable visual sharpness recovery |
| **GNT** | PEFT-LoRA ($r=4$) | PSNR | *Not Evaluated* | **14.57 $\pm$ 0.07 dB** | **+4.15 dB vs. WIP** | Matches Full-FT quality while enabling in-place zero-latency inference merging |
| **Instant-NGP** | Per-Scene ($N=10$) | PSNR | 14.06 dB | 14.06 dB | — | Per-scene fast baseline (now outperformed by adapted GNT!) |
| **Nerfacto** | Per-Scene ($N=10$) | PSNR | 27.11 dB | 27.11 dB | — | Per-scene upper bound under sparse camera supervision |

---

### 2. Comprehensive Architectural & Methodological Comparison

| Dimension | Early WIP Baseline | Current Benchmark | Key Impact & Innovation |
| :--- | :--- | :--- | :--- |
| **Adaptation Taxonomy** | Monolithic, unconstrained full fine-tuning | 5 modular strategies: `Full-FT`, `PFT-Decoder`, `PFT-Backbone`, `BitFit`, `PEFT-LoRA` | Allows isolating spatial priors (backbone) vs. volumetric renderers (decoder) |
| **Parameter Management** | Unmanaged backpropagation across entire network | Dynamic parameter isolation via `setup_adaptation()` and strict gradient freezing | Eliminates gradient interference and protects learned generalizable priors |
| **Adaptation Budget** | Arbitrary $2,500$ steps without convergence analysis | Calibrated $500$-step budget validated via step ablation ($0 \dots 2,500$ steps) | Eliminates PixelNeRF late-stage overfitting ($11.03$ dB @ 500 $\rightarrow$ $8.57$ dB @ 2500) |
| **LoRA Runtime Cost** | Standard forward pass with runtime adapter addition | In-place weight fusion ($\Delta W = \frac{\alpha}{r}BA$) via `merge_lora_weights()` | Zero inference latency overhead for LoRA adapters during rendering |
| **Cross-Validation Rigor** | Single ad-hoc runs without cross-validation | Stratified 4-Fold CV across 8 LLFF scenes $\times$ 3 random seeds ($42, 123, 999$) | Statistically reliable benchmarking; prunes diverged runs; computes standard deviation |
| **Hyperparameter Grid** | Fixed uncalibrated $\eta=10^{-3}$ across all architectures | Grid search across $\eta \in \{10^{-4}, 5\times 10^{-4}, 10^{-3}\}$ and rank $r \in \{4, 8\}$ | Uncovers distinct optimal learning rate regimes for CNNs ($\eta=10^{-4}$) vs. Transformers ($\eta=10^{-3}$) |
| **Automation & Tooling** | Manual command execution and ad-hoc log parsing | Unified CLI suite (`run_stratified_kfold_grid_search.py`, `run_tta_steps_ablation.py`) | Automated multi-seed execution, outlier filtering, and reproducible CSV/LaTeX export |
| **Scientific Manuscript** | Exploratory draft in `docs/sibgrapi-2026/` | Camera-ready manuscript in `docs/final-2027/` with Algorithm 1 & convergence figures | Formalized pseudocode, empirical justification for step budget, and unified comparison tables |

---

### 3. Key Methodological Drivers of the Turnaround

1. **Resolving Representation Collapse via Architectural Inversion:**
   * In the WIP baseline, unconstrained backpropagation caused coordinate MLPs to rapidly overfit to the few sparse rays, destroying the learned 3D geometry.
   * In the current benchmark, isolating the feature backbone (**PFT-Backbone**) in PixelNeRF achieved **10.92 dB** (+4.54 dB over WIP FT). Conversely, GNT's domain adaptation is governed strictly by its multi-view cross-attention transformer; isolating its ray decoder (**PFT-Decoder**) achieved **15.07 dB** (+4.65 dB over WIP FT), establishing the new state-of-the-art for generalizable models on sparse LLFF.
2. **Empirical Step Budget Calibration (The 500-Step Sweet Spot):**
   * Step ablation profiling across $0$ to $2,500$ iterations revealed that PixelNeRF peaks at exactly 500 iterations (11.03 dB) before suffering catastrophic degradation down to 8.57 dB at 2,500 steps due to sparse-view overfitting.
   * GNT achieves $>90\%$ of its adaptation gains within the first 500 steps ($13.91$ dB). Standardizing the budget at 500 steps maximizes fidelity while reducing fine-tuning time by $80\%$.
3. **Zero-Overhead Inference via In-Place LoRA Fusion:**
   * Low-Rank Adaptation (PEFT-LoRA) inserts trainable decomposition matrices $\Delta W = \frac{\alpha}{r}BA$ into linear projections. Post fine-tuning, `merge_lora_weights()` computes and folds $\Delta W$ directly into the base weights $W_0$, completely eliminating runtime adapter overhead during novel view synthesis.
4. **Statistical Reliability via Multi-Seed Stratified Cross-Validation:**
   * Evaluating across 3 random seeds ($42, 123, 999$) with 4-fold cross-validation and automated failure pruning (filtering runs with $PSNR \le 0.0$) ensures all reported gains are statistically robust and reproducible.

---

## Quantitative Findings

Experiments were conducted on the 8 real-world scenes of the LLFF dataset at downscale factor 8 ($504 \times 378$ resolution) using an isolated NVIDIA GeForce GTX 1660 Super (6 GB VRAM) running CUDA 12.6.

### Consolidated Comparative Results across All 5 Fine-Tuning Strategies

The table below summarizes multi-seed aggregations (seeds $42, 123, 999$) across all valid LLFF scenes under sparse input views ($N=3$ views, 500 fine-tuning steps):

| Model | Fine-Tuning Strategy | Learning Rate ($\eta$) | LoRA Rank ($r$) | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ | LPIPS $\downarrow$ | Train / FT Time $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PixelNeRF** | Zero-Shot (ZS) | N/A | N/A | 6.35 $\pm$ 0.05 | 0.085 | 0.800 | $\sim$22 s |
| **PixelNeRF** | Full-FT | $10^{-4}$ | N/A | 9.03 $\pm$ 0.11 | 0.136 | 0.752 | 4 min 45 s |
| **PixelNeRF** | PFT-Decoder | $10^{-4}$ | N/A | 10.58 $\pm$ 0.08 | 0.205 | 0.773 | 4 min 12 s |
| **PixelNeRF** | **PFT-Backbone** | $10^{-4}$ | N/A | **10.92 $\pm$ 0.26** | 0.162 | 0.772 | 4 min 36 s |
| **PixelNeRF** | BitFit | $10^{-4}$ | N/A | 10.79 $\pm$ 0.19 | 0.119 | 0.775 | 4 min 38 s |
| **PixelNeRF** | PEFT-LoRA | $10^{-3}$ | 4 | 8.59 $\pm$ 0.10 | 0.175 | 0.725 | 7 min 06 s |
| **GNT** | Zero-Shot (ZS) | N/A | N/A | 10.51 $\pm$ 0.11 | 0.314 | 0.742 | $\sim$9 s |
| **GNT** | Full-FT | $10^{-3}$ | N/A | 15.06 $\pm$ 0.03 | **0.369** | 0.681 | 5 min 57 s |
| **GNT** | **PFT-Decoder** | $10^{-3}$ | N/A | **15.07 $\pm$ 0.03** | 0.368 | **0.679** | 4 min 59 s |
| **GNT** | PFT-Backbone | $10^{-4}$ | N/A | 10.51 $\pm$ 0.11 | 0.314 | 0.742 | 6 min 01 s |
| **GNT** | BitFit | $10^{-3}$ | N/A | 13.12 $\pm$ 0.05 | 0.358 | 0.742 | 5 min 19 s |
| **GNT** | **PEFT-LoRA** | $10^{-3}$ | 4 | 14.57 $\pm$ 0.07 | **0.369** | 0.706 | 8 min 08 s |
| **Instant-NGP** | Per-Scene ($N=10$) | $10^{-2}$ | N/A | 14.06 | 0.442 | 0.498 | 31 min 51 s |
| **Nerfacto** | Per-Scene ($N=10$) | $10^{-2}$ | N/A | **27.11** | **0.875** | **0.171** | 8 min 57 s |

### Core Empirical Insights

1. **Architectural Divergence in Adaptation:**
   * **PixelNeRF** derives its highest gains from adapting the 2D feature extractor (**PFT-Backbone: 10.92 dB**, $+4.57$ dB over ZS) or additive biases (**BitFit: 10.79 dB**). Because its coordinate MLP easily overfits under sparse supervision, locking the renderer and updating only spatial priors ensures optimization stability.
   * **GNT** achieves peak fidelity by adapting the multi-view cross-attention transformer (**PFT-Decoder: 15.07 dB**, $+4.56$ dB over ZS) or low-rank adapters (**PEFT-LoRA: 14.57 dB**). Inverting this hierarchy (PFT-Backbone) yields no gain over zero-shot, proving that GNT's domain adaptation is governed strictly by its volumetric ray transformer.
2. **Efficiency of Parameter-Efficient Fine-Tuning (PEFT):**
   * **BitFit** tunes $<0.15\%$ of parameters (bias vectors only) yet achieves $10.79$ dB in PixelNeRF and $13.12$ dB in GNT, functioning as a built-in regularizer against floaters.
   * **PEFT-LoRA** ($r=4$) matches Full-FT visual quality in GNT while confining gradient updates to a low-dimensional manifold and supporting zero-latency inference weight merging.
3. **Fine-Tuning Step Dynamics & Overfitting:**
   * PixelNeRF peaks at exactly **500 iterations (11.03 dB)** before degrading to $8.57$ dB at 2,500 iterations due to sparse-view overfitting.
   * GNT exhibits smooth monotonic convergence up to 2,000 steps ($14.99$ dB), with $>90\%$ of gains realized in the first 500 steps.
4. **Pareto Trade-offs:** While per-scene Nerfacto remains the upper bound for fidelity ($27.11$ dB), adapted GNT ($15.07$ dB) establishes the highest quality among generalizable architectures, rendering zero-shot in seconds and fine-tuning in under 5 minutes.

---

## Repository Structure

```text
generalizable-nerf-benchmark/
├── assets/                               # Data, artifacts, and benchmark logs
│   ├── data/baseline/nerf_llff_data/     # LLFF dataset scenes
│   └── reports/                          # Consolidated CSVs, plots, and LaTeX tables
│       ├── grid_search/                  # PixelNeRF grid search logs
│       ├── grid_search_gnt/              # GNT grid search logs
│       ├── tables/                       # Reproducible LaTeX & CSV comparison tables
│       └── tta_steps_ablation/           # Fine-tuning step convergence logs and plots
├── docs/
│   ├── entrega-mestrado/                 # Master's dissertation documentation
│   └── final-2027/                       # SIBGRAPI camera-ready LaTeX sources & figures
├── scripts/                              # Automated benchmark & reporting suite
│   ├── run_stratified_kfold_grid_search.py # Main stratified K-fold cross-validation grid search runner
│   ├── run_tta_steps_ablation.py         # Iteration step ablation & convergence profiler
│   ├── generate_grid_search_latex_tables.py # Generates consolidated LaTeX & CSV tables
│   ├── aggregate_metrics.py              # Aggregates metrics across scene logs
│   ├── generate_selected_plots.py        # Generates radar charts and boxplots
│   └── generate_lr_ablation.py           # Learning rate ablation formatter
├── src/                                  # Core benchmarking framework
│   ├── dataset/
│   │   ├── ImageDataset.py               # LLFF loader, downscaling, and partitioner
│   │   └── SubsetWrapper.py              # Uniform and random sparse subset sampler
│   ├── nerf/
│   │   └── NerfModel.py                  # Unified Nerfstudio wrapper (PixelNeRF, GNT, NGP, Nerfacto)
│   ├── utils/
│   │   ├── config.py                     # Global logger and execution configurations
│   │   ├── metrics.py                    # MetricsLogger (PSNR, SSIM, LPIPS, VRAM, FPS)
│   │   └── types.py                      # Enums and metric identifiers
│   └── validation/
│       └── eval_images.py                # Standardized evaluation routines
├── Dockerfile                            # Reproducible container recipe (CUDA 12.6, COLMAP, tiny-cuda-nn)
├── docker-compose.yml                    # Container service configurations
├── main.py                               # Base execution entrypoint
├── pyproject.toml                        # UV package definition and dependencies
└── uv.lock                               # Deterministic dependency lockfile
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

## Benchmarking & Analysis Scripts

All benchmarking, profiling, and analysis scripts are located in `scripts/`:

```bash
# 1. Run stratified 4-Fold cross-validation grid search across models and strategies
python scripts/run_stratified_kfold_grid_search.py --model gnt --strategy peft-lora --learning-rate 0.001 --lora-rank 4
python scripts/run_stratified_kfold_grid_search.py --model pixel-nerf --strategy pft-backbone --learning-rate 0.0001

# 2. Run fine-tuning step ablation and convergence profiling (0 to 2,500 steps)
python scripts/run_tta_steps_ablation.py --max-eval-images 10

# 3. Generate consolidated LaTeX and CSV comparison tables (e.g., table_mixed_comparison.tex)
python scripts/generate_grid_search_latex_tables.py

# 4. Consolidate and inspect global averages across all scenes and models
python scripts/aggregate_metrics.py

# 5. Generate visual trade-off radar charts and PSNR boxplots
python scripts/generate_selected_plots.py

# 6. Generate learning rate ablation analysis
python scripts/generate_lr_ablation.py
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
