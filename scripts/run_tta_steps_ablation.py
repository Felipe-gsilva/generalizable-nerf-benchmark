#!/usr/bin/env python3
"""
TTA Steps Ablation Experiment for Generalizable NeRFs (Pixel-NeRF & GNT)
========================================================================

This script evaluates how the number of Test-Time Adaptation (TTA) steps
affects reconstruction fidelity (PSNR, SSIM, LPIPS) for the top-performing
adaptation strategies of Pixel-NeRF and GNT.

Empirical top configurations:
  - Pixel-NeRF:
      1. PFT-Backbone (lr=1e-4) -> Top #1 overall
      3. PEFT-LoRA (lr=1e-3, r=4)
  - GNT:
      1. PFT-Backbone (lr=1e-4)
      3. PEFT-LoRA (lr=1e-3, r=4)

Evaluates steps: [0 (Zero-Shot), 100, 250, 500, 1000, 1500, 2000, 2500].
Limits test views to 10 images (as specified by user) for fast evaluation.

Generates:
  - assets/reports/tta_steps_ablation/tta_steps_results.csv
  - assets/reports/tta_steps_ablation/tta_steps_psnr.png
  - assets/reports/tta_steps_ablation/tta_steps_ssim.png
  - assets/reports/tta_steps_ablation/tta_steps_lpips.png
  - assets/reports/tta_steps_ablation/tta_steps_combined.png
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from adaptation import FineTuningStrategy
from src.dataset.ImageDataset import ImageDataset
from src.nerf.NerfModel import (
    AvailableMetrics,
    NerfModel,
    RenderMode,
)
from src.utils.config import config


# Dicionário com as configurações empiricamente comprovadas como as melhores
TOP_CONFIGS = [
    # --- Pixel-NeRF ---
    {
        "model": "pixel-nerf",
        "strategy": FineTuningStrategy.PFT_BACKBONE_ONLY.value,
        "lr": 1e-4,
        "lora_rank": 0,
        "lora_alpha": 0.0,
        "label": "PixelNeRF (PFT-Backbone)",
        "color": "#1f77b4",  # Blue
        "marker": "o",
        "linestyle": "-",
    },
    {
        "model": "pixel-nerf",
        "strategy": FineTuningStrategy.PEFT_LORA.value,
        "lr": 1e-3,
        "lora_rank": 4,
        "lora_alpha": 8.0,
        "label": "PixelNeRF (LoRA r=4)",
        "color": "#3b528b",  # Dark Blue
        "marker": "^",
        "linestyle": ":",
    },
    # --- GNT ---
    {
        "model": "gnt",
        "strategy": FineTuningStrategy.PFT_BACKBONE_ONLY.value,
        "lr": 1e-4,
        "lora_rank": 0,
        "lora_alpha": 0.0,
        "label": "GNT (PFT-Backbone)",
        "color": "#d62728",  # Red
        "marker": "o",
        "linestyle": "-",
    },
    {
        "model": "gnt",
        "strategy": FineTuningStrategy.PEFT_LORA.value,
        "lr": 1e-3,
        "lora_rank": 4,
        "lora_alpha": 8.0,
        "label": "GNT (LoRA r=4)",
        "color": "#9467bd",  # Purple
        "marker": "^",
        "linestyle": ":",
    },
]


def run_single_step_eval(
    model_name: str,
    strategy: str,
    steps: int,
    lr: float,
    lora_rank: int,
    lora_alpha: float,
    data_path: Path,
    dataset_name: str,
    num_views: int = 3,
    downscale: int = 8,
    max_eval_images: int = 10,
    seed: int = 42,
) -> Dict[str, float]:
    """Executa um ponto de teste (treino com `steps` passos + avaliação com até `max_eval_images` visões)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_obj = ImageDataset(
        name=dataset_name,
        step="train",
        dataset_path=data_path,
    )

    regime = "zero-shot" if steps == 0 or strategy == "zero-shot" else "tta"
    actual_strategy = "zero-shot" if steps == 0 else strategy

    config.logger.info(
        f"🚀 Running {model_name} | Strategy: {actual_strategy} | Steps: {steps} | LR: {lr}"
    )

    nerf_model = NerfModel(
        model_name=model_name,
        images=dataset_obj,
        split_strategy="uniform",
        num_views=num_views,
        llffhold=8,
        regime=regime,
        adaptation_strategy=actual_strategy,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        tta_steps=steps,
    )

    extra_args = []
    if regime != "zero-shot":
        if model_name == "gnt":
            extra_args += [
                "--optimizers.network.optimizer.lr", str(lr),
                "--optimizers.feature_net.optimizer.lr", str(lr),
            ]
        elif model_name == "pixel-nerf":
            extra_args += [
                "--optimizers.encoder.optimizer.lr", str(lr),
                "--optimizers.nerf.optimizer.lr", str(lr),
            ]

    t0 = time.time()
    success = nerf_model.train(
        data_path=data_path,
        downscale_factor=downscale,
        num_imgs=0,
        extra_cmd_args=extra_args,
    )
    train_time = time.time() - t0

    if not success:
        config.logger.error(
            f"❌ Training failed for {model_name} with {actual_strategy} at step {steps}"
        )
        return {"psnr": 0.0, "ssim": 0.0, "lpips": 1.0, "train_time": train_time}

    # Avaliação em perspectiva limitando o número de imagens de teste conforme requisitado
    metrics = nerf_model.evaluate_all_metrics(
        mode=RenderMode.PERSPECTIVE,
        metrics=[AvailableMetrics.PSNR, AvailableMetrics.SSIM, AvailableMetrics.LPIPS],
        downscale_factor=downscale,
        max_eval_images=max_eval_images,
    )

    return {
        "psnr": float(metrics.get("psnr", 0.0)),
        "ssim": float(metrics.get("ssim", 0.0)),
        "lpips": float(metrics.get("lpips", 1.0)),
        "train_time": float(train_time),
    }


def plot_results(
    csv_path: Path,
    output_dir: Path,
    top_configs: List[Dict[str, Any]],
):
    """Gera gráficos de linhas profissionais e publicáveis para PSNR, SSIM e LPIPS vs TTA Steps."""
    if not csv_path.exists():
        print(f"⚠️ Arquivo de resultados não encontrado: {csv_path}")
        return

    records: List[Dict[str, Any]] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                records.append({
                    "model": row["model"],
                    "strategy": row["strategy"],
                    "tta_steps": int(row["tta_steps"]),
                    "psnr": float(row["psnr"]),
                    "ssim": float(row["ssim"]),
                    "lpips": float(row["lpips"]),
                    "train_time": float(row.get("train_time", 0.0)),
                })
            except Exception:
                pass

    if not records:
        print("⚠️ Nenhum registro válido encontrado no CSV para plotar.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
    })

    # 1. Gráficos Individuais (PSNR, SSIM, LPIPS)
    metric_configs = [
        ("psnr", "PSNR (dB) $\\uparrow$", "Peak Signal-to-Noise Ratio vs TTA Steps", "tta_steps_psnr.png", True),
        ("ssim", "SSIM $\\uparrow$", "Structural Similarity Index vs TTA Steps", "tta_steps_ssim.png", True),
        ("lpips", "LPIPS $\\downarrow$", "Perceptual Loss (LPIPS) vs TTA Steps", "tta_steps_lpips.png", False),
    ]

    for metric_key, y_label, title, filename, higher_better in metric_configs:
        fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
        
        for cfg_item in top_configs:
            m = cfg_item["model"]
            s = cfg_item["strategy"]
            label = cfg_item["label"]
            color = cfg_item["color"]
            marker = cfg_item["marker"]
            linestyle = cfg_item["linestyle"]

            # Filtra registros correspondentes
            series = [
                r for r in records
                if r["model"] == m and (r["strategy"] == s or (r["tta_steps"] == 0 and r["strategy"] == "zero-shot"))
            ]
            if not series:
                # Tenta casar zero-shot para o ponto 0
                zs_rec = [r for r in records if r["model"] == m and r["tta_steps"] == 0]
                other_recs = [r for r in records if r["model"] == m and r["strategy"] == s]
                series = zs_rec + other_recs

            # Ordena por steps
            series = sorted(series, key=lambda x: x["tta_steps"])
            if not series:
                continue

            x_vals = [r["tta_steps"] for r in series]
            y_vals = [r[metric_key] for r in series]

            ax.plot(
                x_vals,
                y_vals,
                label=label,
                color=color,
                marker=marker,
                markersize=6,
                linewidth=2.0,
                linestyle=linestyle,
                alpha=0.9,
            )

            # Destaca o ponto ótimo (máximo ou mínimo)
            if len(y_vals) > 0:
                best_idx = np.argmax(y_vals) if higher_better else np.argmin(y_vals)
                best_x = x_vals[best_idx]
                best_y = y_vals[best_idx]
                ax.scatter([best_x], [best_y], s=100, facecolors="none", edgecolors=color, linewidths=2.0, zorder=5)

        ax.set_xlabel("Number of TTA Steps (Iterations)", fontweight="bold")
        ax.set_ylabel(y_label, fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="best")
        fig.tight_layout()

        out_path = output_dir / filename
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"📊 Gráfico salvo com sucesso: {out_path}")

    # 2. Gráfico Combinado Multi-Painel (1x3: PSNR, SSIM, LPIPS)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    for i, (metric_key, y_label, title, _, higher_better) in enumerate(metric_configs):
        ax = axes[i]
        for cfg_item in top_configs:
            m = cfg_item["model"]
            s = cfg_item["strategy"]
            label = cfg_item["label"]
            color = cfg_item["color"]
            marker = cfg_item["marker"]
            linestyle = cfg_item["linestyle"]

            series = [
                r for r in records
                if r["model"] == m and (r["strategy"] == s or (r["tta_steps"] == 0 and r["strategy"] == "zero-shot"))
            ]
            if not series:
                zs_rec = [r for r in records if r["model"] == m and r["tta_steps"] == 0]
                other_recs = [r for r in records if r["model"] == m and r["strategy"] == s]
                series = zs_rec + other_recs

            series = sorted(series, key=lambda x: x["tta_steps"])
            if not series:
                continue

            x_vals = [r["tta_steps"] for r in series]
            y_vals = [r[metric_key] for r in series]

            ax.plot(
                x_vals,
                y_vals,
                label=label,
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.8,
                linestyle=linestyle,
                alpha=0.9,
            )

            if len(y_vals) > 0:
                best_idx = np.argmax(y_vals) if higher_better else np.argmin(y_vals)
                ax.scatter([x_vals[best_idx]], [y_vals[best_idx]], s=80, facecolors="none", edgecolors=color, linewidths=1.8, zorder=5)

        ax.set_xlabel("TTA Steps", fontweight="bold")
        ax.set_ylabel(y_label, fontweight="bold")
        ax.set_title(f"({chr(97+i)}) {title.split(' vs ')[0]}", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

    # Legenda única compartilhada no topo
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(handles),
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        fontsize=10,
    )
    fig.tight_layout()

    combined_path = output_dir / "tta_steps_combined.png"
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Gráfico multi-painel salvo: {combined_path}")


def main():
    parser = argparse.ArgumentParser(
        description="TTA Steps Ablation Benchmark for Pixel-NeRF and GNT"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["pixel-nerf", "gnt"],
        choices=["pixel-nerf", "gnt"],
        help="Models to evaluate (default: pixel-nerf gnt)",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[0, 100, 250, 500, 1000, 1500, 2000, 2500],
        help="TTA step iterations to test (default: 0 100 250 500 1000 1500 2000 2500)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with reduced step points (0, 250, 500, 1000, 2000)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="fern",
        help="LLFF scene name (default: fern)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="assets/data/baseline/nerf_llff_data/fern",
        help="Path to scene dataset folder",
    )
    parser.add_argument(
        "--views",
        type=int,
        default=3,
        help="Few-shot input views (default: 3)",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=8,
        help="Image downscale factor (default: 8)",
    )
    parser.add_argument(
        "--max-eval-images",
        type=int,
        default=10,
        help="Limit test views evaluated for fast execution (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/reports/tta_steps_ablation",
        help="Output directory for reports, csv, and plots",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip combinations already present in CSV",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip execution and directly generate plots from existing CSV",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tta_steps_results.csv"

    # Filtra configurações pelos modelos solicitados
    active_configs = [c for c in TOP_CONFIGS if c["model"] in args.models]

    if args.quick:
        args.steps = [0, 250, 500, 1000, 2000]

    if args.plot_only:
        print("📈 Modo --plot-only ativado. Gerando gráficos a partir do CSV existente...")
        plot_results(csv_path, output_dir, active_configs)
        return

    # Lê dados existentes para pular se --skip-existing estiver habilitado
    existing_keys = set()
    if csv_path.exists():
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                key = (r["model"], r["strategy"], int(r["tta_steps"]))
                existing_keys.add(key)

    csv_fieldnames = [
        "model",
        "strategy",
        "lr",
        "lora_rank",
        "tta_steps",
        "psnr",
        "ssim",
        "lpips",
        "train_time",
    ]

    if not csv_path.exists():
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()

    print("=" * 80)
    print("🔬 INICIANDO BENCHMARK DE ABLAÇÃO DE PASSOS DE TTA")
    print(f"   Modelos:          {args.models}")
    print(f"   Passos testados:  {args.steps}")
    print(f"   Cena:             {args.scene}")
    print(f"   Visões de teste:  Limite de {args.max_eval_images} imagens")
    print(f"   Downscale:        {args.downscale}")
    print(f"   Arquivo CSV:      {csv_path}")
    print("=" * 80)

    # Executa cada configuração ativa
    for cfg_item in active_configs:
        m = cfg_item["model"]
        s = cfg_item["strategy"]
        lr = cfg_item["lr"]
        lora_rank = cfg_item["lora_rank"]
        lora_alpha = cfg_item["lora_alpha"]
        label = cfg_item["label"]

        print(f"\n▶ Avaliando Configuração: {label} (LR: {lr})")

        for steps in args.steps:
            lookup_strategy = "zero-shot" if steps == 0 else s
            key = (m, lookup_strategy, steps)

            if args.skip_existing and key in existing_keys:
                print(f"  ⏭️ Pulando: {m} | {lookup_strategy} | {steps} steps (já presente no CSV)")
                continue

            print(f"  ⚙️ Executando: {steps} steps...")
            res = run_single_step_eval(
                model_name=m,
                strategy=lookup_strategy,
                steps=steps,
                lr=lr,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                data_path=Path(args.data_path),
                dataset_name=args.scene,
                num_views=args.views,
                downscale=args.downscale,
                max_eval_images=args.max_eval_images,
                seed=args.seed,
            )

            # Salva no CSV incrementalmente
            row = {
                "model": m,
                "strategy": lookup_strategy,
                "lr": lr if steps > 0 else 0.0,
                "lora_rank": lora_rank if s == "peft-lora" else 0,
                "tta_steps": steps,
                "psnr": f"{res['psnr']:.4f}",
                "ssim": f"{res['ssim']:.4f}",
                "lpips": f"{res['lpips']:.4f}",
                "train_time": f"{res['train_time']:.2f}",
            }
            with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                writer.writerow(row)
            existing_keys.add(key)

            print(
                f"  ✅ Resultado {m} @ {steps} steps: "
                f"PSNR={res['psnr']:.2f} dB, SSIM={res['ssim']:.3f}, LPIPS={res['lpips']:.3f} "
                f"({res['train_time']:.1f}s)"
            )

    print("\n" + "=" * 80)
    print("📈 Todos os passos avaliados! Gerando gráficos de linhas...")
    plot_results(csv_path, output_dir, active_configs)
    print(f"🎉 Processo concluído! Gráficos e dados disponíveis em: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
