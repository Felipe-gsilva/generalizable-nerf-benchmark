import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# Monkey-patch GNTModel's _load_pretrained to resolve docker workspace paths on host
try:
    import GNTModel
    old_load_pretrained = GNTModel.GNTModel._load_pretrained
    def new_load_pretrained(self, ckpt_path):
        if isinstance(ckpt_path, str) and ckpt_path.startswith("/workspace/"):
            ckpt_path = ckpt_path.replace("/workspace/", "/home/felipe-gsilva/dev/cs/nerf-ann-paper/")
        return old_load_pretrained(self, ckpt_path)
    GNTModel.GNTModel._load_pretrained = new_load_pretrained
    print("🐵 Successfully monkey-patched GNTModel._load_pretrained")
except Exception as e:
    print(f"⚠️ Failed to monkey-patch GNTModel: {e}")

import torch
import time
import sys
import argparse
import json
from pathlib import Path
from src.dataset.ImageDataset import ImageDataset
from src.nerf.NeRFModel import NerfModel
from src.utils.types import AvailableMetrics, RenderMode

# Speed-up optimization: Monkey-patch NerfModel's measure_inference_fps to return 0.0,
# as we already calculate FPS during the actual render step, saving 13 redundant full-frame renders.
NerfModel.measure_inference_fps = lambda self, *args, **kwargs: 0.0

# Explicit mapping of configurations to their exact timestamp directories (since load_from_disk expects dir paths)
CHECKPOINT_MAPPING = {
    # PixelNeRF
    ("pixel-nerf", "zero-shot", "0"): "assets/data/nerf_checkpoints/fern/fern/pixel-nerf/2026-05-31_035234",
    ("pixel-nerf", "tta", "3"): "assets/data/nerf_checkpoints/fern/fern/pixel-nerf/2026-05-31_233339",
    ("pixel-nerf", "tta", "6"): "assets/data/nerf_checkpoints/fern/fern/pixel-nerf/2026-06-01_003300",
    ("pixel-nerf", "tta", "10"): "assets/data/nerf_checkpoints/fern/fern/pixel-nerf/2026-06-01_013616",

    # GNT
    ("gnt", "zero-shot", "0"): "assets/data/nerf_checkpoints/fern/fern/gnt/2026-05-31_055712",
    ("gnt", "tta", "3"): "assets/data/nerf_checkpoints/fern/fern/gnt/2026-05-31_090710",
    ("gnt", "tta", "6"): "assets/data/nerf_checkpoints/fern/fern/gnt/2026-05-31_095044",
    ("gnt", "tta", "10"): "assets/data/nerf_checkpoints/fern/fern/gnt/2026-05-31_103430",

    # Per-scene (evaluated using N=10 uniform split configs matching original paper)
    ("instant-ngp", "per-scene", "10"): "assets/data/nerf_checkpoints/fern/fern/instant-ngp/2026-05-31_221109",
    ("nerfacto", "per-scene", "10"): "assets/data/nerf_checkpoints/fern/fern/nerfacto/2026-05-31_230955"
}

def run_evaluation_for_config(model_name, regime, views, display_name):
    base_data_path = Path("assets/data/baseline/nerf_llff_data")
    ds = "fern"
    
    key = (model_name, regime, views)
    if key not in CHECKPOINT_MAPPING:
        print(f"❌ Configuration not found in mapping: {key}")
        return
        
    ckpt_dir = Path(CHECKPOINT_MAPPING[key])
    print(f"\n🚀 [Subprocess] Evaluating {display_name} ({regime}, views={views})...")
    print(f"  [Subprocess] Exact checkpoint directory: {ckpt_dir}")
    
    if not ckpt_dir.exists():
        print(f"  [Subprocess] ❌ Checkpoint directory does not exist!")
        return

    # Parse views to int
    num_views_val = int(views)
    if num_views_val == 0:
        num_views_val = 10 # Default fallback
        
    dataset_obj = ImageDataset(
        name=ds,
        step="train",
        dataset_path=base_data_path / ds,
    )
    
    try:
        # Instantiate model
        nerf_model = NerfModel(
            model_name=model_name,
            images=dataset_obj,
            split_strategy="uniform",
            num_views=num_views_val,
            llffhold=8,
            regime=regime,
        )
        
        # Load the checkpoint
        loaded = nerf_model.load_from_disk(ckpt_dir, mode="val", downscale_factor=4)
        if not loaded:
            print(f"  [Subprocess] ❌ Failed to load checkpoint for {display_name}")
            return
            
        print(f"  [Subprocess] ✅ Checkpoint loaded successfully!")
        
        # Save Correct Full Renders & Measure Render Time/FPS
        save_dir = Path("results_correct") / f"{model_name}_{regime}_{views}"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [Subprocess] 🎨 Rendering perspective test views to {save_dir}...")
        
        start_time = time.time()
        rendered_imgs = nerf_model.render(
            mode=RenderMode.PERSPECTIVE,
            save_path=save_dir,
            downscale_factor=4,
        )
        elapsed_time = time.time() - start_time
        num_frames = len(rendered_imgs) if rendered_imgs else 0
        fps = num_frames / elapsed_time if elapsed_time > 0 and num_frames > 0 else 0.0
        print(f"  [Subprocess] ⚡ Rendered {num_frames} perspective images in {elapsed_time:.2f}s ({fps:.2f} FPS)!")
        
        # Calculate Visual Metrics (Strictly limit to PSNR, SSIM, LPIPS to avoid PyTorch hub download hangs)
        print("  [Subprocess] 📊 Evaluating metrics (PSNR, SSIM, LPIPS)...")
        metrics = nerf_model.evaluate_all_metrics(
            mode=RenderMode.PERSPECTIVE,
            metrics=[AvailableMetrics.PSNR, AvailableMetrics.SSIM, AvailableMetrics.LPIPS],
            downscale_factor=4
        )
        
        if metrics:
            metrics["fps"] = fps
            print(f"  [Subprocess] ✅ Metrics calculated successfully!")
            print(f"  [Subprocess] PSNR: {metrics.get('psnr'):.2f} | SSIM: {metrics.get('ssim'):.3f} | LPIPS: {metrics.get('lpips'):.3f}")
            
            # Save results to a temp json file
            out_json = Path("results_correct") / f"{model_name}_{regime}_{views}_metrics.json"
            with open(out_json, "w") as f:
                json.dump(metrics, f)
        else:
            print(f"  [Subprocess] ❌ Failed to calculate metrics.")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [Subprocess] ❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-model", type=str, default=None)
    parser.add_argument("--regime", type=str, default=None)
    parser.add_argument("--views", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    
    if args.single_model:
        run_evaluation_for_config(args.single_model, args.regime, args.views, args.name)
        return
        
    # Main orchestrator process to run all 10 configurations of Table 2
    experiments = [
        {"model": "pixel-nerf", "regime": "zero-shot", "views": "0", "name": "PixelNeRF Zero-Shot"},
        {"model": "pixel-nerf", "regime": "tta", "views": "3", "name": "PixelNeRF TTA N=3"},
        {"model": "pixel-nerf", "regime": "tta", "views": "6", "name": "PixelNeRF TTA N=6"},
        {"model": "pixel-nerf", "regime": "tta", "views": "10", "name": "PixelNeRF TTA N=10"},
        
        {"model": "gnt", "regime": "zero-shot", "views": "0", "name": "GNT Zero-Shot"},
        {"model": "gnt", "regime": "tta", "views": "3", "name": "GNT TTA N=3"},
        {"model": "gnt", "regime": "tta", "views": "6", "name": "GNT TTA N=6"},
        {"model": "gnt", "regime": "tta", "views": "10", "name": "GNT TTA N=10"},
        
        {"model": "instant-ngp", "regime": "per-scene", "views": "10", "name": "Instant-NGP Per-Scene"},
        {"model": "nerfacto", "regime": "per-scene", "views": "10", "name": "Nerfacto Per-Scene"},
    ]
    
    print("=" * 95)
    print("🎬 STARTING PARALLELIZABLE ORCHESTRATION OF EXACT CONFIG-SPECIFIC EVALUATIONS (DOWNSCALE 4)")
    print("=" * 95)
    
    Path("results_correct").mkdir(parents=True, exist_ok=True)
    
    for exp in experiments:
        model_name = exp["model"]
        regime = exp["regime"]
        views = exp["views"]
        display_name = exp["name"]
        
        temp_json = Path("results_correct") / f"{model_name}_{regime}_{views}_metrics.json"
        if temp_json.exists():
            temp_json.unlink()
            
        print(f"\n🔄 Spawning subprocess for {display_name}...")
        
        import subprocess
        cmd = [
            sys.executable,
            __file__,
            "--single-model", model_name,
            "--regime", regime,
            "--views", views,
            "--name", display_name
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":" + os.getcwd()
        
        try:
            subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Subprocess for {display_name} failed with exit code {e.returncode}")
            
    # Collect and compile all results
    print("\n" + "=" * 95)
    print("📝 FINAL SUMMARY OF RECALCULATED METRICS & FPS FOR TABLE 2 (DOWNSCALE 4)")
    print("=" * 95)
    print(f"{'Configuration':<30} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'LPIPS':<8} | {'FPS':<8}")
    print("-" * 95)
    
    recalculated_results = []
    for exp in experiments:
        model_name = exp["model"]
        regime = exp["regime"]
        views = exp["views"]
        display_name = exp["name"]
        temp_json = Path("results_correct") / f"{model_name}_{regime}_{views}_metrics.json"
        
        if temp_json.exists():
            try:
                with open(temp_json, "r") as f:
                    metrics = json.load(f)
                print(f"{display_name:<30} | {metrics.get('psnr'):.2f:<10} | {metrics.get('ssim'):.3f:<8} | {metrics.get('lpips'):.3f:<8} | {metrics.get('fps'):.2f:<8}")
                recalculated_results.append((display_name, metrics))
            except Exception as e:
                print(f"⚠️ Failed to read results for {display_name}: {e}")
        else:
            print(f"{display_name:<30} | {'N/A':<10} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8}")
            
    print("=" * 95)
    
    # Save the consolidated results to results_correct/consolidated_results.json
    with open("results_correct/consolidated_results.json", "w") as f:
        json.dump(recalculated_results, f, indent=4)
    print("\n✅ Saved consolidated results to results_correct/consolidated_results.json")

if __name__ == "__main__":
    main()
