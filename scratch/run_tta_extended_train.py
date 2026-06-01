import os
import torch
import sys
import json
import time
from pathlib import Path

# Monkey-patch GNTModel's _load_pretrained to resolve docker workspace paths on host
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
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

from src.dataset.ImageDataset import ImageDataset
from src.nerf.NeRFModel import NerfModel
from src.utils.types import AvailableMetrics, RenderMode

NerfModel.measure_inference_fps = lambda self, *args, **kwargs: 0.0

def run_tta_train(model_name, views):
    base_data_path = Path("assets/data/baseline/nerf_llff_data")
    ds = "fern"
    
    num_views_val = int(views)
    display_name = f"{model_name} TTA N={views} (1500 steps)"
    
    print(f"\n🚀 [Subprocess] Starting Extended TTA Training for {display_name}...")
    dataset_obj = ImageDataset(
        name=ds,
        step="train",
        dataset_path=base_data_path / ds,
    )
    
    try:
        # Instantiate model with 1500 TTA steps
        nerf_model = NerfModel(
            model_name=model_name,
            images=dataset_obj,
            split_strategy="uniform",
            num_views=num_views_val,
            llffhold=8,
            regime="tta",
            tta_steps=1500,
        )
        
        # Train
        print(f"  [Subprocess] 🏋️ Training with 1500 steps...")
        nerf_model.train(data_path=dataset_obj.dataset_path, downscale_factor=4, num_imgs=num_views_val)
        
        # Load the best model (which was just trained)
        ckpt_dir = nerf_model.get_output_path(dataset_obj.dataset_path)
        print(f"  [Subprocess] 🔍 Loading trained model from {ckpt_dir}...")
        loaded = nerf_model.load_from_disk(ckpt_dir, mode="val", downscale_factor=4)
        
        if not loaded:
            print(f"  [Subprocess] ❌ Failed to load checkpoint after training for {display_name}")
            return
            
        print(f"  [Subprocess] ✅ Checkpoint loaded successfully!")
        
        save_dir = Path("results_tta_1500") / f"{model_name}_tta_{views}"
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
            metrics=[AvailableMetrics.PSNR, AvailableMetrics.SSIM, AvailableMetrics.LPIPS],
            downscale_factor=4
        )
        
        if metrics:
            metrics["fps"] = fps
            print(f"  [Subprocess] ✅ Metrics calculated successfully!")
            print(f"  [Subprocess] PSNR: {metrics.get('psnr'):.2f} | SSIM: {metrics.get('ssim'):.3f} | LPIPS: {metrics.get('lpips'):.3f}")
            
            # Save results
            out_json = Path("results_tta_1500") / f"{model_name}_tta_{views}_metrics.json"
            with open(out_json, "w") as f:
                json.dump(metrics, f)
        else:
            print(f"  [Subprocess] ❌ Failed to calculate metrics.")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [Subprocess] ❌ Error: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-model", type=str, default=None)
    parser.add_argument("--views", type=str, default=None)
    args = parser.parse_args()
    
    if args.single_model:
        run_tta_train(args.single_model, args.views)
        return
        
    experiments = [
        {"model": "pixel-nerf", "views": "3"},
        {"model": "pixel-nerf", "views": "6"},
        {"model": "pixel-nerf", "views": "10"},
        {"model": "gnt", "views": "3"},
        {"model": "gnt", "views": "6"},
        {"model": "gnt", "views": "10"},
    ]
    
    Path("results_tta_1500").mkdir(parents=True, exist_ok=True)
    import subprocess
    
    for exp in experiments:
        print(f"\n🔄 Spawning subprocess for {exp['model']} TTA N={exp['views']} (1500 steps)...")
        cmd = [
            sys.executable,
            __file__,
            "--single-model", exp["model"],
            "--views", exp["views"]
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":" + os.getcwd()
        try:
            subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Subprocess failed with exit code {e.returncode}")

if __name__ == "__main__":
    main()
