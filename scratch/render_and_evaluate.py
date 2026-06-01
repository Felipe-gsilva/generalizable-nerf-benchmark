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

def run_evaluation_for_model(model_name, regime, display_name):
    base_data_path = Path("assets/data/baseline/nerf_llff_data")
    ds = "fern"
    
    print(f"\n🚀 [Subprocess] Evaluating {display_name} ({regime})...")
    
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
            num_views=10,
            llffhold=8,
            regime=regime,
        )
        
        # Identify output path where checkpoints are located
        checkpoint_base_path = nerf_model.get_output_path(dataset_obj.dataset_path)
        print(f"  [Subprocess] Checkpoint search path: {checkpoint_base_path}")
        
        # Load the checkpoint
        # Use mode="val" and downscale_factor=4
        loaded = nerf_model.load_from_disk(checkpoint_base_path, mode="val", downscale_factor=4)
        if not loaded:
            print(f"  [Subprocess] ❌ Failed to load checkpoint for {display_name} at {checkpoint_base_path}")
            return
            
        print(f"  [Subprocess] ✅ Checkpoint loaded successfully!")
        
        # Save Correct Full Renders & Measure Render Time/FPS
        save_dir = Path("results_correct") / model_name
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
        
        # Calculate Visual Metrics
        print("  [Subprocess] 📊 Evaluating metrics (PSNR, SSIM, LPIPS)...")
        metrics = nerf_model.evaluate_all_metrics(
            mode=RenderMode.PERSPECTIVE,
            metrics=list(AvailableMetrics),
            downscale_factor=4
        )
        
        if metrics:
            metrics["fps"] = fps
            print(f"  [Subprocess] ✅ Metrics calculated successfully!")
            # Save results to a temp json file
            out_json = Path("results_correct") / f"{model_name}_metrics.json"
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
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    
    if args.single_model:
        run_evaluation_for_model(args.single_model, args.regime, args.name)
        return
        
    # Main orchestrator process
    experiments = [
        {"model": "pixel-nerf", "regime": "tta", "name": "PixelNeRF"},
        {"model": "gnt", "regime": "tta", "name": "GNT"},
        {"model": "instant-ngp", "regime": "per-scene", "name": "Instant-NGP"},
        {"model": "nerfacto", "regime": "per-scene", "name": "Nerfacto"},
    ]
    
    print("=" * 90)
    print("🎬 STARTING RE-RENDERING, FPS COMPUTATION AND METRIC RE-EVALUATION FROM CHECKPOINTS")
    print("   (Using separate subprocesses, downscale_factor=4 to prevent CUDA Out-Of-Memory errors)")
    print("=" * 90)
    
    # Ensure correct output directory exists
    Path("results_correct").mkdir(parents=True, exist_ok=True)
    
    for exp in experiments:
        model_name = exp["model"]
        regime = exp["regime"]
        display_name = exp["name"]
        
        # Remove any previous temp json file
        temp_json = Path("results_correct") / f"{model_name}_metrics.json"
        if temp_json.exists():
            temp_json.unlink()
            
        print(f"\n🔄 Spawning subprocess for {display_name}...")
        
        # Run subprocess
        import subprocess
        cmd = [
            sys.executable,
            __file__,
            "--single-model", model_name,
            "--regime", regime,
            "--name", display_name
        ]
        
        # Inherit PYTHONPATH and other environment variables
        env = os.environ.copy()
        env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":" + os.getcwd()
        
        try:
            res = subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Subprocess for {display_name} failed with exit code {e.returncode}")
            
    # Collect results
    recalculated_results = {}
    for exp in experiments:
        model_name = exp["model"]
        display_name = exp["name"]
        temp_json = Path("results_correct") / f"{model_name}_metrics.json"
        if temp_json.exists():
            try:
                with open(temp_json, "r") as f:
                    recalculated_results[display_name] = json.load(f)
                temp_json.unlink() # Cleanup
            except Exception as e:
                print(f"⚠️ Failed to read results for {display_name}: {e}")
                
    # Print a final summary table
    print("\n" + "=" * 90)
    print("📝 SUMMARY OF RECALCULATED METRICS & FPS FROM CORRECT FULL IMAGES (DOWNSCALE 4)")
    print("=" * 90)
    print(f"{'Model':<15} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'LPIPS':<8} | {'FPS':<8}")
    print("-" * 90)
    for model_name, metrics in recalculated_results.items():
        print(f"{model_name:<15} | {metrics.get('psnr'):.2f:<10} | {metrics.get('ssim'):.3f:<8} | {metrics.get('lpips'):.3f:<8} | {metrics.get('fps'):.2f:<8}")
    print("=" * 90)

if __name__ == "__main__":
    main()
