#!/usr/bin/env python3
"""
Generate Reproducible LaTeX Tables for Stratified K-Fold Grid Search Experiments.

Filters out incomplete / failed runs (where PSNR == 0.0), aggregates performance
across 3 random seeds (42, 123, 999) and scenes, and produces publication-ready
LaTeX tables for:
  1. GNT-exclusive adaptation strategies summary.
  2. Full GNT hyperparameter grid search ablation.
  3. Mixed comparison between PixelNeRF and GNT (strictly using GNT from the GNT
     experiment and PixelNeRF from the PixelNeRF experiment, never mixing).
  4. Strict shared-scene comparison (fair 1:1 evaluation on common scenes).
  5. Per-scene breakdown table.

Usage:
    python scripts/generate_grid_search_latex_tables.py
    python scripts/generate_grid_search_latex_tables.py \
        --gnt-csv assets/reports/grid_search_gnt/kfold_grid_search_detailed.csv \
        --pixelnerf-csv assets/reports/grid_search/kfold_grid_search_detailed_with_pixel_nerf.csv \
        --output-dir assets/reports/tables
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Default paths
DEFAULT_GNT_CSV = Path("assets/reports/grid_search_gnt/kfold_grid_search_detailed.csv")
DEFAULT_PIXELNERF_CSV = Path("assets/reports/grid_search/kfold_grid_search_detailed_with_pixel_nerf.csv")
DEFAULT_OUTPUT_DIR = Path("assets/reports/tables")
DOCS_TABLES_DIR = Path("docs/final-2027/tables")

# Fallback paths in case assets haven't been copied yet
FALLBACK_GNT_PATHS = [
    Path("assets/reports/grid_search_gnt/kfold_grid_search_detailed.csv"),
    Path("/home/felipe-gsilva/.local/share/Trash/files/kfold_grid_search_detailed(1).csv"),
    Path("/home/felipe-gsilva/Downloads/kfold_grid_search_detailed(1).csv"),
]

FALLBACK_PN_PATHS = [
    Path("assets/reports/grid_search/kfold_grid_search_detailed_with_pixel_nerf.csv"),
    Path("/home/felipe-gsilva/Downloads/kfold_grid_search_detailed_with_pixel_nerf.csv"),
    Path("/home/felipe-gsilva/Downloads/kfold_grid_search_detailed.csv"),
]

STRATEGY_ORDER = [
    "zero-shot",
    "full-ft",
    "pft-decoder",
    "pft-backbone",
    "bitfit",
    "peft-lora",
]

STRATEGY_DISPLAY = {
    "zero-shot": "Zero-Shot (ZS)",
    "full-ft": "Full-FT",
    "pft-decoder": "PFT-Decoder",
    "pft-backbone": "PFT-Backbone",
    "bitfit": "BitFit",
    "peft-lora": "PEFT-LoRA",
}


def resolve_file_path(requested: Path, fallbacks: List[Path]) -> Path:
    """Finds the first existing path among requested and fallback locations."""
    if requested.exists():
        return requested
    for fb in fallbacks:
        if fb.exists():
            return fb
    raise FileNotFoundError(f"Could not locate file at {requested} or any fallback: {fallbacks}")


def load_and_clean_data(
    csv_path: Path,
    model_name: str,
    filter_zeros: bool = True,
    deduplicate: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Loads raw detailed grid search CSV, isolates the target model,
    removes duplicate runs if any, and filters out failed runs (PSNR == 0.0).
    """
    df_raw = pd.read_csv(csv_path)
    total_raw_rows = len(df_raw)

    # Strictly isolate the specified model
    df_model = df_raw[df_raw["model"] == model_name].copy()
    model_rows = len(df_model)

    # Optional deduplication (keeping the last evaluation if duplicate runs were appended)
    if deduplicate:
        df_model = df_model.drop_duplicates(
            subset=["scene", "strategy", "seed", "lr", "lora_rank"],
            keep="last",
        )
    dedup_rows = len(df_model)

    # Filter out failed/zero PSNR runs
    if filter_zeros:
        zero_rows = len(df_model[df_model["psnr"] <= 0.0])
        df_clean = df_model[df_model["psnr"] > 0.0].copy()
    else:
        zero_rows = 0
        df_clean = df_model.copy()

    stats = {
        "raw_total": total_raw_rows,
        "model_rows": model_rows,
        "duplicates_removed": model_rows - dedup_rows,
        "zeros_removed": zero_rows,
        "clean_rows": len(df_clean),
        "valid_scenes": sorted(df_clean["scene"].unique().tolist()),
    }
    return df_clean, stats


def aggregate_configurations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-configuration metrics by:
      1. Calculating scene-average performance for each seed.
      2. Computing mean and sample standard deviation across the 3 seeds.
    """
    records = []
    for (strat, lr, rank), grp in df.groupby(["strategy", "lr", "lora_rank"]):
        # Group by seed: compute mean across scenes for that seed
        seed_means = grp.groupby("seed")[["psnr", "ssim", "lpips", "time_sec"]].mean()

        mean_psnr = float(seed_means["psnr"].mean())
        std_psnr = float(seed_means["psnr"].std()) if len(seed_means) > 1 else 0.0

        mean_ssim = float(seed_means["ssim"].mean())
        std_ssim = float(seed_means["ssim"].std()) if len(seed_means) > 1 else 0.0

        mean_lpips = float(seed_means["lpips"].mean())
        std_lpips = float(seed_means["lpips"].std()) if len(seed_means) > 1 else 0.0

        mean_time = float(seed_means["time_sec"].mean())
        num_scenes = int(grp["scene"].nunique())
        num_seeds = int(grp["seed"].nunique())

        records.append({
            "strategy": strat,
            "lr": float(lr),
            "lora_rank": int(rank),
            "mean_psnr": mean_psnr,
            "std_psnr": std_psnr,
            "mean_ssim": mean_ssim,
            "std_ssim": std_ssim,
            "mean_lpips": mean_lpips,
            "std_lpips": std_lpips,
            "mean_time": mean_time,
            "scenes": num_scenes,
            "seeds": num_seeds,
        })

    df_agg = pd.DataFrame(records)

    # Sort in canonical order
    strategy_cat = pd.Categorical(df_agg["strategy"], categories=STRATEGY_ORDER, ordered=True)
    df_agg["sort_key"] = strategy_cat
    df_agg = df_agg.sort_values(by=["sort_key", "lr", "lora_rank"]).drop(columns=["sort_key"])
    return df_agg


def select_best_configurations(df_agg: pd.DataFrame) -> pd.DataFrame:
    """Selects the best configuration per strategy based on highest mean PSNR."""
    best_indices = df_agg.groupby("strategy")["mean_psnr"].idxmax()
    df_best = df_agg.loc[best_indices].copy()

    # Maintain canonical strategy order
    strategy_cat = pd.Categorical(df_best["strategy"], categories=STRATEGY_ORDER, ordered=True)
    df_best["sort_key"] = strategy_cat
    df_best = df_best.sort_values(by="sort_key").drop(columns=["sort_key"])
    return df_best


def format_latex_lr(lr: float, math_mode: bool = True) -> str:
    """Formats learning rate for scientific publication in LaTeX."""
    if lr <= 0:
        return "N/A"
    if lr == 1e-4:
        val = r"10^{-4}"
    elif lr == 5e-4:
        val = r"5\times10^{-4}"
    elif lr == 1e-3:
        val = r"10^{-3}"
    else:
        val = f"{lr:.1e}"
    return f"${val}$" if math_mode else val


def format_latex_rank(rank: int) -> str:
    """Formats LoRA rank for LaTeX."""
    return str(rank) if rank > 0 else "N/A"


def generate_gnt_summary_latex(df_gnt_best: pd.DataFrame, num_scenes: int) -> str:
    """Generates Table 1: GNT strategy performance summary in LaTeX."""
    zs_psnr = df_gnt_best[df_gnt_best["strategy"] == "zero-shot"]["mean_psnr"].values[0]

    max_psnr = df_gnt_best["mean_psnr"].max()
    max_ssim = df_gnt_best["mean_ssim"].max()
    min_lpips = df_gnt_best["mean_lpips"].min()

    latex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Desempenho das estratégias de adaptação no modelo GNT (média de {num_scenes} cenas válidas e 3 seeds: 42, 123, 999). As falhas de convergência (PSNR zero em \textit{{leaves}} e \textit{{orchids}}) foram excluídas.}}",
        r"\label{tab:gnt_strategies_summary}",
        r"\scriptsize",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llccccc@{\extracolsep{0pt}}}",
        r"\toprule",
        r"Estratégia & Hiperparâmetro & PSNR (dB) $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ & Tempo (s) $\downarrow$ & $\Delta$PSNR (dB) \\",
        r"\midrule",
    ]

    for _, row in df_gnt_best.iterrows():
        strat_display = STRATEGY_DISPLAY.get(row["strategy"], row["strategy"])

        if row["strategy"] == "zero-shot":
            hp_str = "---"
        elif row["strategy"] == "peft-lora":
            hp_str = rf"$\eta={format_latex_lr(row['lr'], math_mode=False)}, r={row['lora_rank']}$"
        else:
            hp_str = rf"$\eta={format_latex_lr(row['lr'], math_mode=False)}$"

        delta = row["mean_psnr"] - zs_psnr
        delta_str = f"+{delta:.2f}" if delta > 0.001 else f"{delta:.2f}"
        if abs(delta) < 0.001:
            delta_str = "---"

        # Format metrics with bolding on optimal
        psnr_val_str = f"{row['mean_psnr']:.2f} $\\pm$ {row['std_psnr']:.2f}"
        if row["mean_psnr"] == max_psnr:
            psnr_val_str = rf"\textbf{{{row['mean_psnr']:.2f}}} $\pm$ {row['std_psnr']:.2f}"

        ssim_val_str = f"{row['mean_ssim']:.3f}"
        if row["mean_ssim"] == max_ssim:
            ssim_val_str = rf"\textbf{{{row['mean_ssim']:.3f}}}"

        lpips_val_str = f"{row['mean_lpips']:.3f}"
        if row["mean_lpips"] == min_lpips:
            lpips_val_str = rf"\textbf{{{row['mean_lpips']:.3f}}}"

        time_val_str = f"{row['mean_time']:.1f}"

        latex_lines.append(
            f"{strat_display:<18} & {hp_str:<22} & {psnr_val_str:<26} & {ssim_val_str:<14} & {lpips_val_str:<14} & {time_val_str:<8} & {delta_str} \\\\"
        )

    latex_lines += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ]
    return "\n".join(latex_lines)


def generate_gnt_full_grid_latex(df_gnt_all: pd.DataFrame, num_scenes: int) -> str:
    """Generates Table 2: Complete GNT grid search ablation across all hyperparameter levels."""
    latex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Ablação completa de hiperparâmetros para o modelo GNT (média em {num_scenes} cenas e 3 seeds).}}",
        r"\label{tab:gnt_full_grid_search}",
        r"\scriptsize",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccccc@{\extracolsep{0pt}}}",
        r"\toprule",
        r"Estratégia & $\eta$ & Rank $r$ & PSNR (dB) $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ \\",
        r"\midrule",
    ]

    current_strat = None
    for _, row in df_gnt_all.iterrows():
        strat = row["strategy"]
        strat_display = STRATEGY_DISPLAY.get(strat, strat)
        if strat != current_strat:
            if current_strat is not None:
                latex_lines.append(r"\midrule")
            current_strat = strat
            strat_label = strat_display
        else:
            strat_label = ""

        lr_str = format_latex_lr(row["lr"])
        rank_str = format_latex_rank(row["lora_rank"])
        psnr_str = f"{row['mean_psnr']:.2f} $\\pm$ {row['std_psnr']:.2f}"
        ssim_str = f"{row['mean_ssim']:.3f}"
        lpips_str = f"{row['mean_lpips']:.3f}"

        latex_lines.append(
            f"{strat_label:<18} & {lr_str:<10} & {rank_str:<6} & {psnr_str:<24} & {ssim_str:<8} & {lpips_str:<8} \\\\"
        )

    latex_lines += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ]
    return "\n".join(latex_lines)


def generate_mixed_hierarchical_latex(
    df_pn_best: pd.DataFrame,
    df_gnt_best: pd.DataFrame,
    pn_scenes: int,
    gnt_scenes: int,
) -> str:
    """Generates Table 3: Mixed model comparison grouped hierarchically by Model."""
    latex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Comparative analysis of fine-tuning strategies: PixelNeRF ({pn_scenes} valid scenes) vs. GNT ({gnt_scenes} valid scenes). Metrics represent multi-seed aggregation across 3 independent runs (seeds 42, 123, 999). Failed optimization runs (zero PSNR) were filtered out.}}",
        r"\label{tab:mixed_model_comparison}",
        r"\scriptsize",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llccccc@{\extracolsep{0pt}}}",
        r"\toprule",
        r"Model & Strategy & $\eta$ & $r$ & PSNR (dB) $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ \\",
        r"\midrule",
    ]

    # PixelNeRF Section
    latex_lines.append(rf"\multirow{{6}}{{*}}{{PixelNeRF}}")
    max_pn_psnr = df_pn_best["mean_psnr"].max()
    for _, row in df_pn_best.iterrows():
        strat_display = STRATEGY_DISPLAY.get(row["strategy"], row["strategy"])
        lr_str = format_latex_lr(row["lr"])
        rank_str = format_latex_rank(row["lora_rank"])
        psnr_str = f"{row['mean_psnr']:.2f} $\\pm$ {row['std_psnr']:.2f}"
        if row["mean_psnr"] == max_pn_psnr:
            psnr_str = rf"\textbf{{{row['mean_psnr']:.2f}}} $\pm$ {row['std_psnr']:.2f}"
        ssim_str = f"{row['mean_ssim']:.3f}"
        lpips_str = f"{row['mean_lpips']:.3f}"

        latex_lines.append(
            f" & {strat_display:<16} & {lr_str:<10} & {rank_str:<6} & {psnr_str:<26} & {ssim_str:<8} & {lpips_str:<8} \\\\"
        )

    latex_lines.append(r"\midrule")

    # GNT Section
    latex_lines.append(rf"\multirow{{6}}{{*}}{{GNT}}")
    max_gnt_psnr = df_gnt_best["mean_psnr"].max()
    for _, row in df_gnt_best.iterrows():
        strat_display = STRATEGY_DISPLAY.get(row["strategy"], row["strategy"])
        lr_str = format_latex_lr(row["lr"])
        rank_str = format_latex_rank(row["lora_rank"])
        psnr_str = f"{row['mean_psnr']:.2f} $\\pm$ {row['std_psnr']:.2f}"
        if row["mean_psnr"] == max_gnt_psnr:
            psnr_str = rf"\textbf{{{row['mean_psnr']:.2f}}} $\pm$ {row['std_psnr']:.2f}"
        ssim_str = f"{row['mean_ssim']:.3f}"
        lpips_str = f"{row['mean_lpips']:.3f}"

        latex_lines.append(
            f" & {strat_display:<16} & {lr_str:<10} & {rank_str:<6} & {psnr_str:<26} & {ssim_str:<8} & {lpips_str:<8} \\\\"
        )

    latex_lines += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ]
    return "\n".join(latex_lines)


def generate_mixed_side_by_side_latex(
    df_pn_best: pd.DataFrame,
    df_gnt_best: pd.DataFrame,
) -> str:
    """Generates Table 4: Side-by-side strategy comparison between PixelNeRF and GNT."""
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Tabela Comparativa Mista: Desempenho de Adaptação por Estratégia em PixelNeRF e GNT (Média das 3 seeds e cenas válidas).}",
        r"\label{tab:mixed_side_by_side}",
        r"\small",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l|cccc|cccc@{\extracolsep{0pt}}}",
        r"\toprule",
        r" & \multicolumn{4}{c|}{\textbf{PixelNeRF}} & \multicolumn{4}{c}{\textbf{GNT}} \\",
        r"\cmidrule{2-5} \cmidrule{6-9}",
        r"Estratégia & Config. Ótima & PSNR (dB) $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ & Config. Ótima & PSNR (dB) $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ \\",
        r"\midrule",
    ]

    pn_map = {r["strategy"]: r for _, r in df_pn_best.iterrows()}
    gnt_map = {r["strategy"]: r for _, r in df_gnt_best.iterrows()}

    for strat in STRATEGY_ORDER:
        strat_display = STRATEGY_DISPLAY.get(strat, strat)
        pn_row = pn_map.get(strat)
        gnt_row = gnt_map.get(strat)

        # PixelNeRF values
        if pn_row is not None:
            if strat == "zero-shot":
                pn_cfg = "ZS"
            elif strat == "peft-lora":
                pn_cfg = rf"$\eta={format_latex_lr(pn_row['lr'], math_mode=False)}, r={pn_row['lora_rank']}$"
            else:
                pn_cfg = rf"$\eta={format_latex_lr(pn_row['lr'], math_mode=False)}$"
            pn_psnr = f"{pn_row['mean_psnr']:.2f} $\\pm$ {pn_row['std_psnr']:.2f}"
            pn_ssim = f"{pn_row['mean_ssim']:.3f}"
            pn_lpips = f"{pn_row['mean_lpips']:.3f}"
        else:
            pn_cfg, pn_psnr, pn_ssim, pn_lpips = "---", "---", "---", "---"

        # GNT values
        if gnt_row is not None:
            if strat == "zero-shot":
                gnt_cfg = "ZS"
            elif strat == "peft-lora":
                gnt_cfg = rf"$\eta={format_latex_lr(gnt_row['lr'], math_mode=False)}, r={gnt_row['lora_rank']}$"
            else:
                gnt_cfg = rf"$\eta={format_latex_lr(gnt_row['lr'], math_mode=False)}$"
            gnt_psnr = f"{gnt_row['mean_psnr']:.2f} $\\pm$ {gnt_row['std_psnr']:.2f}"
            gnt_ssim = f"{gnt_row['mean_ssim']:.3f}"
            gnt_lpips = f"{gnt_row['mean_lpips']:.3f}"
        else:
            gnt_cfg, gnt_psnr, gnt_ssim, gnt_lpips = "---", "---", "---", "---"

        latex_lines.append(
            f"{strat_display:<16} & {pn_cfg:<18} & {pn_psnr:<22} & {pn_ssim:<6} & {pn_lpips:<6} & "
            f"{gnt_cfg:<18} & {gnt_psnr:<22} & {gnt_ssim:<6} & {gnt_lpips:<6} \\\\"
        )

    latex_lines += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table*}",
    ]
    return "\n".join(latex_lines)


def generate_per_scene_latex(
    df_gnt: pd.DataFrame,
    df_pn: pd.DataFrame,
    df_gnt_best: pd.DataFrame,
    df_pn_best: pd.DataFrame,
) -> str:
    """Generates Table 5: Per-scene PSNR breakdown for best configurations."""
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Detalhamento de PSNR (dB) por cena individual para as configurações ótimas (média das 3 seeds).}",
        r"\label{tab:per_scene_breakdown}",
        r"\small",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llcccccc|c@{\extracolsep{0pt}}}",
        r"\toprule",
        r"Modelo & Estratégia & Fern & Fortress & Flower & Horns & Room & Trex & Média Geral \\",
        r"\midrule",
    ]

    all_scenes = ["fern", "fortress", "flower", "horns", "room", "trex"]

    # PixelNeRF rows
    latex_lines.append(r"\multirow{6}{*}{PixelNeRF}")
    for _, row in df_pn_best.iterrows():
        strat = row["strategy"]
        strat_display = STRATEGY_DISPLAY.get(strat, strat)
        sub = df_pn[(df_pn["strategy"] == strat) & (df_pn["lr"] == row["lr"]) & (df_pn["lora_rank"] == row["lora_rank"])]
        scene_means = sub.groupby("scene")["psnr"].mean().to_dict()

        scene_vals = []
        for sc in all_scenes:
            if sc in scene_means:
                scene_vals.append(f"{scene_means[sc]:.2f}")
            else:
                scene_vals.append("---")
        mean_str = f"{row['mean_psnr']:.2f}"

        latex_lines.append(
            f" & {strat_display:<16} & " + " & ".join(f"{v:>6}" for v in scene_vals) + f" & {mean_str:>6} \\\\"
        )

    latex_lines.append(r"\midrule")

    # GNT rows
    latex_lines.append(r"\multirow{6}{*}{GNT}")
    for _, row in df_gnt_best.iterrows():
        strat = row["strategy"]
        strat_display = STRATEGY_DISPLAY.get(strat, strat)
        sub = df_gnt[(df_gnt["strategy"] == strat) & (df_gnt["lr"] == row["lr"]) & (df_gnt["lora_rank"] == row["lora_rank"])]
        scene_means = sub.groupby("scene")["psnr"].mean().to_dict()

        scene_vals = []
        for sc in all_scenes:
            if sc in scene_means:
                scene_vals.append(f"{scene_means[sc]:.2f}")
            else:
                scene_vals.append("---")
        mean_str = f"{row['mean_psnr']:.2f}"

        latex_lines.append(
            f" & {strat_display:<16} & " + " & ".join(f"{v:>6}" for v in scene_vals) + f" & {mean_str:>6} \\\\"
        )

    latex_lines += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table*}",
    ]
    return "\n".join(latex_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate reproducible LaTeX tables for GNT and PixelNeRF adaptation experiments."
    )
    parser.add_argument(
        "--gnt-csv",
        type=Path,
        default=DEFAULT_GNT_CSV,
        help="Path to GNT grid search detailed CSV",
    )
    parser.add_argument(
        "--pixelnerf-csv",
        type=Path,
        default=DEFAULT_PIXELNERF_CSV,
        help="Path to PixelNeRF grid search detailed CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory to save LaTeX tables and CSV summaries",
    )
    parser.add_argument(
        "--latex-output-dir",
        type=Path,
        default=DOCS_TABLES_DIR,
        help="Optional directory to also save .tex files for paper inclusion",
    )
    args = parser.parse_args()

    # Resolve input paths
    gnt_path = resolve_file_path(args.gnt_csv, FALLBACK_GNT_PATHS)
    pn_path = resolve_file_path(args.pixelnerf-csv if hasattr(args, "pixelnerf-csv") else args.pixelnerf_csv, FALLBACK_PN_PATHS)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("📊 GENERATING REPRODUCIBLE LATEX TABLES FOR GRID SEARCH EXPERIMENTS")
    print(f"   GNT CSV source:        {gnt_path}")
    print(f"   PixelNeRF CSV source:  {pn_path}")
    print(f"   Output directory:      {output_dir}")
    print("=" * 80)

    # 1. Load and clean GNT data
    df_gnt, gnt_stats = load_and_clean_data(gnt_path, model_name="gnt", filter_zeros=True, deduplicate=True)
    print("\n✅ [GNT Dataset Processed]:")
    print(f"   Total rows loaded:       {gnt_stats['raw_total']}")
    print(f"   GNT rows:                {gnt_stats['model_rows']}")
    print(f"   Duplicates removed:      {gnt_stats['duplicates_removed']}")
    print(f"   Failed runs removed (0): {gnt_stats['zeros_removed']}")
    print(f"   Clean valid rows:        {gnt_stats['clean_rows']}")
    print(f"   Valid scenes ({len(gnt_stats['valid_scenes'])}):         {gnt_stats['valid_scenes']}")

    # 2. Load and clean PixelNeRF data (explicitly ensuring model == 'pixel-nerf')
    df_pn, pn_stats = load_and_clean_data(pn_path, model_name="pixel-nerf", filter_zeros=True, deduplicate=True)
    print("\n✅ [PixelNeRF Dataset Processed]:")
    print(f"   Total rows loaded:       {pn_stats['raw_total']}")
    print(f"   PixelNeRF rows:          {pn_stats['model_rows']}")
    print(f"   Duplicates removed:      {pn_stats['duplicates_removed']}")
    print(f"   Failed runs removed (0): {pn_stats['zeros_removed']}")
    print(f"   Clean valid rows:        {pn_stats['clean_rows']}")
    print(f"   Valid scenes ({len(pn_stats['valid_scenes'])}):         {pn_stats['valid_scenes']}")

    # 3. Aggregate configurations
    df_gnt_all = aggregate_configurations(df_gnt)
    df_gnt_best = select_best_configurations(df_gnt_all)

    df_pn_all = aggregate_configurations(df_pn)
    df_pn_best = select_best_configurations(df_pn_all)

    # 4. Shared scenes evaluation (exact 1:1 fairness)
    shared_scenes = sorted(list(set(gnt_stats["valid_scenes"]).intersection(pn_stats["valid_scenes"])))
    print(f"\n🔍 Shared Valid Scenes ({len(shared_scenes)}): {shared_scenes}")

    df_gnt_shared = df_gnt[df_gnt["scene"].isin(shared_scenes)]
    df_gnt_shared_all = aggregate_configurations(df_gnt_shared)
    df_gnt_shared_best = select_best_configurations(df_gnt_shared_all)

    df_pn_shared = df_pn[df_pn["scene"].isin(shared_scenes)]
    df_pn_shared_all = aggregate_configurations(df_pn_shared)
    df_pn_shared_best = select_best_configurations(df_pn_shared_all)

    # 5. Generate LaTeX Tables
    table_gnt_summary = generate_gnt_summary_latex(df_gnt_best, len(gnt_stats["valid_scenes"]))
    table_gnt_full = generate_gnt_full_grid_latex(df_gnt_all, len(gnt_stats["valid_scenes"]))
    table_mixed_hierarchical = generate_mixed_hierarchical_latex(
        df_pn_best, df_gnt_best, len(pn_stats["valid_scenes"]), len(gnt_stats["valid_scenes"])
    )
    table_mixed_side_by_side = generate_mixed_side_by_side_latex(df_pn_best, df_gnt_best)
    table_mixed_shared = generate_mixed_side_by_side_latex(df_pn_shared_best, df_gnt_shared_best)
    table_per_scene = generate_per_scene_latex(df_gnt, df_pn, df_gnt_best, df_pn_best)

    # 6. Save LaTeX files
    save_targets = [output_dir]
    if args.latex_output_dir is not None:
        args.latex_output_dir.mkdir(parents=True, exist_ok=True)
        save_targets.append(args.latex_output_dir)

    latex_files = {
        "table_gnt_summary.tex": table_gnt_summary,
        "table_gnt_full_grid.tex": table_gnt_full,
        "table_mixed_comparison.tex": table_mixed_hierarchical,
        "table_mixed_side_by_side.tex": table_mixed_side_by_side,
        "table_mixed_shared_scenes.tex": table_mixed_shared,
        "table_per_scene_breakdown.tex": table_per_scene,
    }

    for target in save_targets:
        for filename, content in latex_files.items():
            out_file = target / filename
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(content + "\n")

    # 7. Save Cleaned and Aggregated CSV summaries
    df_gnt_best.to_csv(output_dir / "gnt_best_strategies.csv", index=False)
    df_gnt_all.to_csv(output_dir / "gnt_all_configurations.csv", index=False)
    df_pn_best.to_csv(output_dir / "pixelnerf_best_strategies.csv", index=False)
    df_pn_all.to_csv(output_dir / "pixelnerf_all_configurations.csv", index=False)

    # Mixed summary CSV
    df_pn_export = df_pn_best.copy()
    df_pn_export["model"] = "pixel-nerf"
    df_gnt_export = df_gnt_best.copy()
    df_gnt_export["model"] = "gnt"
    df_mixed_export = pd.concat([df_pn_export, df_gnt_export], ignore_index=True)
    df_mixed_export.to_csv(output_dir / "mixed_comparison.csv", index=False)

    # 8. Print Results to CLI
    print("\n" + "=" * 90)
    print("🏆 GNT BEST CONFIGURATIONS PER STRATEGY (Averaged across 3 seeds & valid scenes)")
    print("=" * 90)
    print(
        f"{'Strategy':<16} | {'LR':<10} | {'LoRA Rank':<10} | {'PSNR (dB)':<18} | {'SSIM':<8} | {'LPIPS':<8} | {'Time (s)':<8}"
    )
    print("-" * 90)
    for _, r in df_gnt_best.iterrows():
        lr_s = format_latex_lr(r["lr"]).replace("$", "")
        rank_s = str(r["lora_rank"]) if r["lora_rank"] > 0 else "N/A"
        psnr_s = f"{r['mean_psnr']:.2f} ± {r['std_psnr']:.2f}"
        print(
            f"{r['strategy']:<16} | {lr_s:<10} | {rank_s:<10} | {psnr_s:<18} | {r['mean_ssim']:.3f}    | {r['mean_lpips']:.3f}    | {r['mean_time']:.1f}"
        )

    print("\n" + "=" * 90)
    print("🏆 PIXEL-NERF BEST CONFIGURATIONS PER STRATEGY (Averaged across 3 seeds & valid scenes)")
    print("=" * 90)
    print(
        f"{'Strategy':<16} | {'LR':<10} | {'LoRA Rank':<10} | {'PSNR (dB)':<18} | {'SSIM':<8} | {'LPIPS':<8} | {'Time (s)':<8}"
    )
    print("-" * 90)
    for _, r in df_pn_best.iterrows():
        lr_s = format_latex_lr(r["lr"]).replace("$", "")
        rank_s = str(r["lora_rank"]) if r["lora_rank"] > 0 else "N/A"
        psnr_s = f"{r['mean_psnr']:.2f} ± {r['std_psnr']:.2f}"
        print(
            f"{r['strategy']:<16} | {lr_s:<10} | {rank_s:<10} | {psnr_s:<18} | {r['mean_ssim']:.3f}    | {r['mean_lpips']:.3f}    | {r['mean_time']:.1f}"
        )
    print("=" * 90)

    print(f"\n✨ Generated LaTeX tables and summaries successfully written to:\n   - {output_dir}")
    if args.latex_output_dir is not None:
        print(f"   - {args.latex_output_dir}")


if __name__ == "__main__":
    main()
