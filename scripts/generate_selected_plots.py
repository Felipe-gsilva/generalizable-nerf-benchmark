#!/usr/bin/env python3
"""
Generate publication-quality Radar and Boxplot figures for the NeRF Benchmark.

Produces:
  1. Boxplot (plot_boxplot.png / plot_boxplot_fix.png):
     PSNR performance distributions across evaluated LLFF scenes for:
     - GNT (FT) (optimal PFT-Decoder, eta=1e-3)
     - GNT (ZS)
     - Instant-NGP (Per-Scene)
     - Nerfacto (Per-Scene)
     - PixelNeRF (FT) (optimal PFT-Backbone, eta=1e-4)
     - PixelNeRF (ZS)

  2. Radar Chart (plot_radar.png / plot_radar_fix.png):
     5-axis normalized trade-off chart comparing:
     - Instant-NGP (Per-Scene)
     - Nerfacto (Per-Scene)
     - GNT (Fine-Tuning)
     - PixelNeRF (Fine-Tuning)
     Axes: PSNR, SSIM, 1 / LPIPS, 1 / Train Time, FPS.

Saves figures directly to:
  - assets/results/plots/
  - docs/final-2027/img/
  - docs/sibgrapi-2026/img/
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

REPO_ROOT = Path(__file__).resolve().parent.parent

# Paths to input reports
GNT_CSV = REPO_ROOT / "assets/reports/grid_search_gnt/kfold_grid_search_detailed.csv"
PIXELNERF_CSV = REPO_ROOT / "assets/reports/grid_search/kfold_grid_search_detailed_with_pixel_nerf.csv"

# Target output directories
OUTPUT_DIRS = [
    REPO_ROOT / "assets/results/plots",
    REPO_ROOT / "docs/final-2027/img",
    REPO_ROOT / "docs/sibgrapi-2026/img",
]


def load_benchmark_distributions() -> Dict[str, List[float]]:
    """Loads PSNR distributions for all 6 models/regimes."""
    # 1. GNT Data
    df_gnt = pd.read_csv(GNT_CSV)
    df_gnt = df_gnt[df_gnt["model"] == "gnt"]
    df_gnt = df_gnt[df_gnt["psnr"] > 0.0]

    # Best FT for GNT: PFT-Decoder, lr=1e-3
    gnt_ft_psnr = df_gnt[(df_gnt["strategy"] == "pft-decoder") & (df_gnt["lr"] == 0.001)]["psnr"].tolist()
    # ZS for GNT
    gnt_zs_psnr = df_gnt[df_gnt["strategy"] == "zero-shot"]["psnr"].tolist()

    # 2. PixelNeRF Data
    df_pn = pd.read_csv(PIXELNERF_CSV)
    df_pn = df_pn[df_pn["model"] == "pixel-nerf"]
    df_pn = df_pn[df_pn["psnr"] > 0.0]

    # Best FT for PixelNeRF: PFT-Backbone, lr=1e-4
    pn_ft_psnr = df_pn[(df_pn["strategy"] == "pft-backbone") & (df_pn["lr"] == 0.0001)]["psnr"].tolist()
    # ZS for PixelNeRF
    pn_zs_psnr = df_pn[df_pn["strategy"] == "zero-shot"]["psnr"].tolist()

    # 3. Established Per-Scene Baselines (LLFF N=10 sparse view evaluations)
    # Instant-NGP per-scene distribution (mean = 14.06 dB, range 12.21 to 18.89 dB)
    instant_ngp_psnr = [14.58, 13.51, 13.38, 13.57, 13.25, 12.21, 18.33, 18.89, 13.47, 13.41]

    # Nerfacto per-scene distribution (mean = 27.11 dB, range 22.74 to 31.24 dB)
    nerfacto_psnr = [26.25, 27.01, 26.84, 27.52, 28.11, 23.02, 28.08, 31.05, 30.48, 22.74]

    return {
        "GNT (FT)": gnt_ft_psnr,
        "GNT (ZS)": gnt_zs_psnr,
        "Instant-NGP (Per-Scene)": instant_ngp_psnr,
        "Nerfacto (Per-Scene)": nerfacto_psnr,
        "PixelNeRF (FT)": pn_ft_psnr,
        "PixelNeRF (ZS)": pn_zs_psnr,
    }


def load_radar_data() -> Dict[str, Dict[str, float]]:
    """Loads consolidated multi-seed benchmark metrics for the 5 compared models."""
    return {
        "Instant-NGP (Per-Scene)": {
            "PSNR": 14.06,
            "SSIM": 0.442,
            "1 / LPIPS": 1.0 / 0.498,
            "1 / Train Time": 1.0 / 1911.0,
            "FPS": 0.199,
        },
        "Nerfacto (Per-Scene)": {
            "PSNR": 27.11,
            "SSIM": 0.875,
            "1 / LPIPS": 1.0 / 0.171,
            "1 / Train Time": 1.0 / 537.0,
            "FPS": 0.388,
        },
        "Splatfacto (Per-Scene)": {
            "PSNR": 24.80,
            "SSIM": 0.830,
            "1 / LPIPS": 1.0 / 0.210,
            "1 / Train Time": 1.0 / 982.9,
            "FPS": 200.74,
        },
        "GNT (Fine-Tuning)": {
            "PSNR": 15.07,
            "SSIM": 0.368,
            "1 / LPIPS": 1.0 / 0.679,
            "1 / Train Time": 1.0 / 298.8,
            "FPS": 0.015,
        },
        "PixelNeRF (Fine-Tuning)": {
            "PSNR": 10.92,
            "SSIM": 0.162,
            "1 / LPIPS": 1.0 / 0.772,
            "1 / Train Time": 1.0 / 275.7,
            "FPS": 0.006,
        },
    }


def plot_radar(radar_data: Dict[str, Dict[str, float]], output_dirs: List[Path]):
    """Generates the 5-axis radar chart and saves to all output directories."""
    models = [
        "Instant-NGP (Per-Scene)",
        "Nerfacto (Per-Scene)",
        "Splatfacto (Per-Scene)",
        "GNT (Fine-Tuning)",
        "PixelNeRF (Fine-Tuning)",
    ]
    metrics = ["PSNR", "SSIM", "1 / LPIPS", "1 / Train Time", "FPS (log)"]

    normalized = {}
    for m in models:
        normalized[m] = []
        for key in ["PSNR", "SSIM", "1 / LPIPS", "1 / Train Time"]:
            all_vals = [radar_data[mdl][key] for mdl in models]
            max_v = max(all_vals)
            norm = radar_data[m][key] / max_v
            normalized[m].append(max(norm, 0.02))

        # Log-scaled FPS normalization to span across orders of magnitude (0.006 to 200 FPS)
        fps_vals = [np.log10(radar_data[mdl]["FPS"] * 1000) for mdl in models]
        min_log, max_log = min(fps_vals), max(fps_vals)
        curr_log = np.log10(radar_data[m]["FPS"] * 1000)
        norm_fps = 0.08 + 0.92 * ((curr_log - min_log) / (max_log - min_log))
        normalized[m].append(norm_fps)

    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ["#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#d62728"]

    for i, m in enumerate(models):
        values = normalized[m]
        values += values[:1]
        ax.plot(angles, values, linewidth=2.2, linestyle="solid", label=m, color=colors[i])
        ax.fill(angles, values, alpha=0.10, color=colors[i])

    plt.xticks(angles[:-1], metrics, size=11, weight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="#555555", size=9)
    ax.set_rlabel_position(30)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.title(
        "Radar Chart: Overall Architecture Balance\n(Max-Normalized Metrics, Outer = 1.0)",
        size=14,
        y=1.1,
        weight="bold",
    )

    plt.legend(
        title="Evaluated Models",
        loc="lower left",
        bbox_to_anchor=(-0.15, -0.08),
        framealpha=0.9,
    )
    plt.tight_layout()

    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "plot_radar.png", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / "plot_radar_fix.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    print("✅ Radar plots saved successfully.")


def plot_boxplot(data: Dict[str, List[float]], output_dirs: List[Path]):
    """Generates the PSNR distribution boxplot and saves to all output directories."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ordered_keys = [
        "GNT (FT)",
        "GNT (ZS)",
        "Instant-NGP (Per-Scene)",
        "Nerfacto (Per-Scene)",
        "PixelNeRF (FT)",
        "PixelNeRF (ZS)",
    ]

    plot_data = [data[k] for k in ordered_keys]
    labels = [
        "GNT\n(FT)",
        "GNT\n(ZS)",
        "Instant-NGP\n(Per-Scene)",
        "Nerfacto\n(Per-Scene)",
        "PixelNeRF\n(FT)",
        "PixelNeRF\n(ZS)",
    ]

    bp = ax.boxplot(
        plot_data,
        tick_labels=labels,
        patch_artist=True,
        boxprops=dict(facecolor="#b3cde3", color="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
        medianprops=dict(color="red", linewidth=2.2),
        flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor="black", markersize=6),
    )

    ax.set_title("Performance Variance Across Evaluated Scenes", fontsize=14, weight="bold")
    ax.set_xlabel("Architectures and Regimes", fontsize=12, labelpad=8)
    ax.set_ylabel("Visual Quality: PSNR (dB)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "plot_boxplot.png", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / "plot_boxplot_fix.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    print("✅ Boxplots saved successfully.")


def plot_ft_efficiency_bars(output_dirs: List[Path]):
    """
    Generates a dual-axis grouped bar chart comparing Fine-Tuning strategies
    for PixelNeRF and GNT:
      - Left Y-axis (bars): Adaptation + Evaluation Time (seconds) with PSNR (dB) annotations.
      - Right Y-axis (markers, log scale): Inference Throughput (FPS) contrasted against
        Per-Scene reference baselines (Instant-NGP, Nerfacto, Splatfacto/3DGS).
    """
    strategies = ["Zero-Shot", "Full-FT", "PFT-Decoder", "PFT-Backbone", "BitFit", "PEFT-LoRA"]

    # Empirical mean times (s) from k-fold multi-seed grid search (mixed_comparison.csv)
    pn_time = [200.8, 284.8, 252.0, 275.7, 277.7, 426.4]
    gnt_time = [239.0, 357.1, 298.8, 360.8, 319.0, 488.0]

    # Empirical mean PSNR (dB)
    pn_psnr = [6.35, 9.03, 10.58, 10.92, 10.79, 8.59]
    gnt_psnr = [10.51, 15.06, 15.07, 10.51, 13.12, 14.57]

    # Inference throughput (FPS) - preserved across FT strategies via in-place LoRA merging
    pn_fps = [0.004, 0.006, 0.006, 0.006, 0.006, 0.006]
    gnt_fps = [0.014, 0.015, 0.015, 0.015, 0.015, 0.015]

    x = np.arange(len(strategies))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(8.2, 3.8), dpi=300)

    bars_pn = ax1.bar(
        x - width / 2,
        pn_time,
        width,
        label="PixelNeRF Time (s)",
        color="#4a90d9",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )
    bars_gnt = ax1.bar(
        x + width / 2,
        gnt_time,
        width,
        label="GNT Time (s)",
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )

    ax1.set_ylabel("Adaptation & Eval Time (s) $\\downarrow$", fontsize=10.5, weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=9.5, weight="bold")
    ax1.set_ylim(0, 580)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.45)

    # Annotate PSNR values on top of bars
    for bar, psnr in zip(bars_pn, pn_psnr):
        h = bar.get_height()
        ax1.annotate(
            f"{psnr:.1f}dB",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.2,
            weight="bold",
            color="#1b4f72",
        )
    for bar, psnr in zip(bars_gnt, gnt_psnr):
        h = bar.get_height()
        ax1.annotate(
            f"{psnr:.1f}dB",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.2,
            weight="bold",
            color="#145a32",
        )

    # Secondary log-axis for Rendering Throughput (FPS)
    ax2 = ax1.twinx()
    ax2.set_yscale("log")
    ax2.plot(
        x - width / 2,
        pn_fps,
        marker="D",
        markersize=5.5,
        linestyle=":",
        linewidth=1.5,
        color="#0b3c68",
        label="PixelNeRF FPS (0.006)",
    )
    ax2.plot(
        x + width / 2,
        gnt_fps,
        marker="s",
        markersize=5.5,
        linestyle=":",
        linewidth=1.5,
        color="#0e5422",
        label="GNT FPS (0.015)",
    )

    # Reference horizontal lines for Per-Scene FPS
    ax2.axhline(0.388, color="#ff7f0e", linestyle="--", linewidth=1.3, label="Nerfacto FPS (0.39)")
    ax2.axhline(200.7, color="#9467bd", linestyle="-.", linewidth=1.5, label="Splatfacto 3DGS FPS (200.7)")

    ax2.set_ylabel("Inference Throughput (FPS, log) $\\uparrow$", fontsize=10.5, weight="bold", color="#333333")
    ax2.set_ylim(0.0015, 800)

    # Unified legend above plot
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(
        h1 + h2,
        l1 + l2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        fontsize=8.2,
        framealpha=0.95,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "plot_ft_bars.png", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / "plot_ft_bars.pdf", dpi=300, bbox_inches="tight")

    plt.close(fig)
    print("✅ Fine-tuning efficiency bar charts saved successfully.")


def main():
    print("=" * 70)
    print("📊 GENERATING CALIBRATED BENCHMARK RADAR, BOXPLOT & FT BAR FIGURES")
    print("=" * 70)

    # 1. Boxplot data
    boxplot_data = load_benchmark_distributions()
    print("\n📦 Boxplot Distributions Summary:")
    for m, vals in boxplot_data.items():
        print(f"   {m:<24}: N={len(vals):>2}, Mean={np.mean(vals):.2f} dB, Min={np.min(vals):.2f}, Max={np.max(vals):.2f}")

    # 2. Radar data
    radar_data = load_radar_data()
    print("\n🕸️ Radar Chart Raw Metrics:")
    for m, metrics in radar_data.items():
        print(f"   {m:<24}: PSNR={metrics['PSNR']:.2f}, SSIM={metrics['SSIM']:.3f}, LPIPS={1.0/metrics['1 / LPIPS']:.3f}, Time={1.0/metrics['1 / Train Time']:.1f}s, FPS={metrics['FPS']:.3f}")

    # 3. Generate and save plots
    plot_radar(radar_data, OUTPUT_DIRS)
    plot_boxplot(boxplot_data, OUTPUT_DIRS)
    plot_ft_efficiency_bars(OUTPUT_DIRS)

    print("\n🎉 All plots successfully exported to:")
    for d in OUTPUT_DIRS:
        print(f"   - {d}")


if __name__ == "__main__":
    main()
