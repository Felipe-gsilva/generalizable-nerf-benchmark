#!/usr/bin/env python3
"""
Stratified K-Fold Cross-Validation & Hyperparameter Grid Search for Generalizable NeRFs
(GNT and PixelNeRF with the new TTA adaptation strategies).

Evaluates n different random seeds across hyperparameter levels (e.g. low/medium/high learning rates,
LoRA rank and alpha) on stratified scene splits of the LLFF dataset to discover the best seed
and optimal hyperparameter configuration per method.

Usage:
    python scripts/run_stratified_kfold_grid_search.py --dry-run
    python scripts/run_stratified_kfold_grid_search.py --models gnt pixel-nerf --strategies zero-shot pft-decoder peft-lora
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptation import FineTuningStrategy
from src.dataset.ImageDataset import ImageDataset
from src.nerf.NerfModel import NerfModel
from src.utils.config import config
from src.utils.types import AvailableMetrics, RenderMode


# Stratified splits for the 8 LLFF scenes according to visual and geometric complexity:
# Category 1: Intricate natural foliage / thin geometry (leaves, orchids, flower, fern)
# Category 2: Structured outdoor / indoor scenes with deep relief (fortress, room, horns, trex)
LLFF_SCENES = [
    {"name": "fern", "category": "foliage_medium_depth"},
    {"name": "flower", "category": "foliage_macro_intricate"},
    {"name": "fortress", "category": "outdoor_deep_relief"},
    {"name": "horns", "category": "object_complex_occlusion"},
    {"name": "leaves", "category": "foliage_dense_occlusion"},
    {"name": "orchids", "category": "foliage_dense_intricate"},
    {"name": "room", "category": "indoor_wide_baseline"},
    {"name": "trex", "category": "object_table_relief"},
]


def create_stratified_k_folds(k: int = 4) -> List[Dict[str, List[str]]]:
    """
    Partitions the 8 LLFF scenes into k balanced folds, pairing one foliage scene
    with one structured / object scene per fold when k=4.
    """
    foliage = ["fern", "flower", "leaves", "orchids"]
    structured = ["fortress", "horns", "room", "trex"]

    if k == 4:
        folds = []
        for i in range(4):
            val_scenes = [foliage[i], structured[i]]
            train_scenes = [s for s in (foliage + structured) if s not in val_scenes]
            folds.append({"fold_id": i, "val_scenes": val_scenes, "train_scenes": train_scenes})
        return folds
    elif k == 2:
        return [
            {
                "fold_id": 0,
                "val_scenes": ["fern", "flower", "fortress", "horns"],
                "train_scenes": ["leaves", "orchids", "room", "trex"],
            },
            {
                "fold_id": 1,
                "val_scenes": ["leaves", "orchids", "room", "trex"],
                "train_scenes": ["fern", "flower", "fortress", "horns"],
            },
        ]
    else:
        # Fallback uniform chunking
        all_scenes = [s["name"] for s in LLFF_SCENES]
        chunk_size = max(1, len(all_scenes) // k)
        folds = []
        for i in range(k):
            start = i * chunk_size
            end = start + chunk_size if i < k - 1 else len(all_scenes)
            val = all_scenes[start:end]
            train = [s for s in all_scenes if s not in val]
            folds.append({"fold_id": i, "val_scenes": val, "train_scenes": train})
        return folds


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_evaluation(
    model_name: str,
    scene_name: str,
    strategy: str,
    lr: float,
    lora_rank: int,
    lora_alpha: float,
    seed: int,
    num_views: int,
    tta_steps: int,
    downscale: int,
    base_data_path: Path,
) -> Dict[str, float]:
    """Runs adaptation/training and test evaluation on a single scene, returning PSNR, SSIM, LPIPS."""
    set_seed(seed)
    data_path = base_data_path / scene_name

    dataset_obj = ImageDataset(
        name=scene_name,
        step="train",
        dataset_path=data_path,
    )

    regime = "zero-shot" if strategy == "zero-shot" else "tta"

    nerf_model = NerfModel(
        model_name=model_name,
        images=dataset_obj,
        split_strategy="uniform",
        num_views=num_views,
        llffhold=8,
        regime=regime,
        adaptation_strategy=strategy,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        tta_steps=tta_steps,
    )

    extra_args = []
    if strategy != "zero-shot":
        if model_name == "gnt":
            extra_args += [
                "--optimizers.network.optimizer.lr", str(lr),
                "--optimizers.feature-net.optimizer.lr", str(lr),
            ]
        elif model_name == "pixel-nerf":
            extra_args += [
                "--optimizers.encoder.optimizer.lr", str(lr),
                "--optimizers.nerf.optimizer.lr", str(lr),
            ]

    success = nerf_model.train(
        data_path=data_path,
        downscale_factor=downscale,
        extra_cmd_args=extra_args,
    )

    if not success:
        config.logger.error(f"❌ Training failed for {model_name} on {scene_name} with {strategy}")
        return {"psnr": 0.0, "ssim": 0.0, "lpips": 1.0}

    # Render and evaluate test images
    rendered_views = nerf_model.render(
        mode=RenderMode.PERSPECTIVE,
        downscale_factor=downscale,
    )

    metrics = {}
    if rendered_views:
        metrics = nerf_model.evaluate_all_metrics(
            mode=RenderMode.PERSPECTIVE,
            metrics=[AvailableMetrics.PSNR, AvailableMetrics.SSIM, AvailableMetrics.LPIPS],
            already_rendered_images=rendered_views,
            downscale_factor=downscale,
            measure_fps=False,
        ) or {}

    def _get_metric(m: Dict[Any, Any], key: Any, default: float) -> float:
        val = m.get(key)
        if val is None and hasattr(key, "value"):
            val = m.get(key.value)
        if val is None and isinstance(key, str):
            for k, v in m.items():
                if str(k).lower() == key.lower():
                    return float(v)
        return float(val) if val is not None else default

    return {
        "psnr": _get_metric(metrics, AvailableMetrics.PSNR, 0.0),
        "ssim": _get_metric(metrics, AvailableMetrics.SSIM, 0.0),
        "lpips": _get_metric(metrics, AvailableMetrics.LPIPS, 1.0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stratified K-Fold Grid Search for NeRF Adaptation Strategies"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gnt", "pixel-nerf"],
        choices=["gnt", "pixel-nerf", "nerfacto", "splatfacto"],
        help="Models to evaluate in the grid search",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["zero-shot", "full-ft", "pft-decoder", "pft-backbone", "bitfit", "peft-lora"],
        help="Adaptation strategies to test",
    )
    parser.add_argument(
        "--lrs",
        nargs="+",
        type=float,
        default=[1e-4, 5e-4, 1e-3],
        help="Learning rate levels to evaluate (e.g. low, medium, high)",
    )
    parser.add_argument(
        "--lora-ranks",
        nargs="+",
        type=int,
        default=[4, 8],
        help="LoRA rank configurations to search",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 999],
        help="List of random seeds to test for identifying the best seed",
    )
    parser.add_argument(
        "--views",
        type=int,
        default=3,
        help="Few-shot number of views (default: 3)",
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=4,
        choices=[2, 4],
        help="Number of stratified folds across the LLFF dataset (default: 4)",
    )
    parser.add_argument(
        "--tta-steps",
        type=int,
        default=500,
        help="Number of TTA adaptation iterations (default: 500)",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=8,
        choices=[1, 2, 4, 8],
        help="LLFF image downscale factor (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in fast dry-run validation mode (5 steps, 1 scene, 1 seed, downscale=8)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/reports/grid_search",
        help="Directory to save grid search reports and best seed tables",
    )

    args = parser.parse_args()

    base_data_path = Path("assets/data/baseline/nerf_llff_data")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("⚡ Dry-run mode enabled: restricting to 5 steps, 1 fold, 1 seed, 1 scene.")
        args.tta_steps = 5
        args.seeds = args.seeds[:1]
        args.lrs = args.lrs[:1]
        args.lora_ranks = args.lora_ranks[:1]

    folds = create_stratified_k_folds(k=args.k_folds)
    if args.dry_run:
        folds = [folds[0]]
        folds[0]["val_scenes"] = folds[0]["val_scenes"][:1]

    detailed_records: List[Dict[str, Any]] = []
    detailed_csv = output_dir / "kfold_grid_search_detailed.csv"
    csv_fieldnames = [
        "fold_id",
        "scene",
        "model",
        "strategy",
        "seed",
        "lr",
        "lora_rank",
        "lora_alpha",
        "num_views",
        "psnr",
        "ssim",
        "lpips",
        "time_sec",
    ]

    # Initialize CSV header if file is empty/non-existent
    if not detailed_csv.exists() or detailed_csv.stat().st_size == 0:
        with open(detailed_csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=csv_fieldnames)
            writer.writeheader()

    print("=" * 80, flush=True)
    print(f"📊 Starting Stratified K-Fold Grid Search ({args.k_folds} folds)", flush=True)
    print(f"   Models:       {args.models}", flush=True)
    print(f"   Strategies:   {args.strategies}", flush=True)
    print(f"   LR Levels:    {args.lrs}", flush=True)
    print(f"   Seeds (n={len(args.seeds)}):    {args.seeds}", flush=True)
    print(f"   TTA Steps:    {args.tta_steps}", flush=True)
    print(f"   Downscale:    {args.downscale}x", flush=True)
    print("=" * 80, flush=True)

    for fold in folds:
        fold_id = fold["fold_id"]
        val_scenes = fold["val_scenes"]
        print(f"\n📂 [Fold {fold_id + 1}/{len(folds)}] Validation Scenes: {val_scenes}", flush=True)

        for scene in val_scenes:
            for model_name in args.models:
                for strategy in args.strategies:
                    # Strategy hyperparameter exploration
                    lr_list = [0.0] if strategy == "zero-shot" else args.lrs
                    lora_rank_list = args.lora_ranks if strategy == "peft-lora" else [0]

                    for lr in lr_list:
                        for lora_rank in lora_rank_list:
                            lora_alpha = float(lora_rank * 2) if lora_rank > 0 else 0.0

                            for seed in args.seeds:
                                print(
                                    f"  ▶️ Fold {fold_id} | {model_name} | {strategy} | "
                                    f"Scene: {scene} | Seed: {seed} | LR: {lr} | LoRA Rank: {lora_rank}",
                                    flush=True,
                                )

                                t0 = time.time()
                                metrics = run_single_evaluation(
                                    model_name=model_name,
                                    scene_name=scene,
                                    strategy=strategy,
                                    lr=lr,
                                    lora_rank=lora_rank,
                                    lora_alpha=lora_alpha,
                                    seed=seed,
                                    num_views=args.views,
                                    tta_steps=args.tta_steps,
                                    downscale=args.downscale,
                                    base_data_path=base_data_path,
                                )
                                elapsed = time.time() - t0

                                rec = {
                                    "fold_id": fold_id,
                                    "scene": scene,
                                    "model": model_name,
                                    "strategy": strategy,
                                    "seed": seed,
                                    "lr": lr,
                                    "lora_rank": lora_rank,
                                    "lora_alpha": lora_alpha,
                                    "num_views": args.views,
                                    "psnr": metrics["psnr"],
                                    "ssim": metrics["ssim"],
                                    "lpips": metrics["lpips"],
                                    "time_sec": round(elapsed, 2),
                                }
                                detailed_records.append(rec)

                                # Incremental write to CSV
                                with open(detailed_csv, "a", newline="", encoding="utf-8") as fp:
                                    writer = csv.DictWriter(fp, fieldnames=csv_fieldnames)
                                    writer.writerow(rec)
                                    fp.flush()

                                print(
                                    f"    ✅ PSNR: {metrics['psnr']:.2f} | "
                                    f"SSIM: {metrics['ssim']:.4f} | "
                                    f"LPIPS: {metrics['lpips']:.4f} ({elapsed:.1f}s)",
                                    flush=True,
                                )

    print(f"\n📁 Detailed records saved to: {detailed_csv}", flush=True)

    # Discover Best Seed per (Model, Strategy)
    # Aggregation across all evaluated folds and scenes
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in detailed_records:
        # Group by (model, strategy, seed, lr, lora_rank)
        group_key = (r["model"], r["strategy"], r["seed"], r["lr"], r["lora_rank"])
        grouped[group_key].append(r)

    config_summaries = []
    for (model, strat, seed, lr, lora_r), recs in grouped.items():
        psnrs = [x["psnr"] for x in recs]
        ssims = [x["ssim"] for x in recs]
        lpipss = [x["lpips"] for x in recs]

        config_summaries.append({
            "model": model,
            "strategy": strat,
            "seed": seed,
            "lr": lr,
            "lora_rank": lora_r,
            "mean_psnr": np.mean(psnrs),
            "std_psnr": np.std(psnrs),
            "mean_ssim": np.mean(ssims),
            "mean_lpips": np.mean(lpipss),
            "num_evals": len(recs),
        })

    # Find the optimal configuration and best seed per (model, strategy)
    best_per_method: Dict[tuple, Dict[str, Any]] = {}
    for c in config_summaries:
        key = (c["model"], c["strategy"])
        if key not in best_per_method or c["mean_psnr"] > best_per_method[key]["mean_psnr"]:
            best_per_method[key] = c

    # Output Best Seed Summary
    best_csv = output_dir / "kfold_grid_search_best_seeds.csv"
    best_rows = list(best_per_method.values())
    if best_rows:
        with open(best_csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(best_rows[0].keys()))
            writer.writeheader()
            writer.writerows(best_rows)
        print(f"🏆 Best seed rankings saved to: {best_csv}", flush=True)

    # Print Formatted Markdown Table
    print("\n" + "=" * 90, flush=True)
    print("🏆 BEST SEED & OPTIMAL HYPERPARAMETERS PER METHOD (Stratified K-Fold Cross-Validation)", flush=True)
    print("=" * 90, flush=True)
    print(
        f"{'Model':<12} | {'Strategy':<14} | {'Best Seed':<10} | {'Best LR':<10} | {'LoRA Rank':<10} | "
        f"{'PSNR (dB)':<14} | {'SSIM':<8} | {'LPIPS':<8}",
        flush=True,
    )
    print("-" * 90, flush=True)
    for (m, strat), b in sorted(best_per_method.items()):
        lr_str = f"{b['lr']:.0e}" if b['lr'] > 0 else "N/A"
        rank_str = str(b['lora_rank']) if b['lora_rank'] > 0 else "N/A"
        psnr_str = f"{b['mean_psnr']:.2f} ± {b['std_psnr']:.2f}"
        print(
            f"{m:<12} | {strat:<14} | {b['seed']:<10} | {lr_str:<10} | {rank_str:<10} | "
            f"{psnr_str:<14} | {b['mean_ssim']:.4f} | {b['mean_lpips']:.4f}",
            flush=True,
        )
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
