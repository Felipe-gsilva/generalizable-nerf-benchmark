import os
import sys
import json
import time
import subprocess
from pathlib import Path

from src.dataset.ImageDataset import ImageDataset
from src.nerf.NeRFModel import NerfModel
from src.utils.types import AvailableMetrics, RenderMode

NerfModel.measure_inference_fps = lambda self, *args, **kwargs: 0.0

def run_experiment(model_name, views):
    base_data_path = Path("assets/data/baseline/nerf_llff_data")
    ds = "fern"
    num_views_val = int(views)
    display_name = f"{model_name} N={views} (RTX 4060 Optimized)"
    
    print(f"\n🚀 [Subprocess] Starting 4060 Experiment for {display_name}...")
    dataset_obj = ImageDataset(
        name=ds,
        step="train",
        dataset_path=base_data_path / ds,
    )
    
    extra_args = []
    if model_name == "nerfacto":
        # Add distortion loss to Nerfacto to kill floaters in few-shot
        extra_args = ["--pipeline.model.distortion-loss-mult", "0.01"]
        
    try:
        nerf_model = NerfModel(
            model_name=model_name,
            images=dataset_obj,
            split_strategy="uniform",
            num_views=num_views_val,
            llffhold=8,
            regime="per-scene"
        )
        
        ds_factor = 4 if num_views_val == 0 else 2
        
        print(f"  [Subprocess] 🏋️ Training with downscale {ds_factor}...")
        nerf_model.train(
            data_path=dataset_obj.dataset_path, 
            downscale_factor=ds_factor, 
            num_imgs=num_views_val,
            extra_cmd_args=extra_args
        )
        
        ckpt_dir = nerf_model.get_output_path(dataset_obj.dataset_path)
        print(f"  [Subprocess] 🔍 Loading trained model from {ckpt_dir}...")
        loaded = nerf_model.load_from_disk(ckpt_dir, mode="val", downscale_factor=ds_factor)

        
        if not loaded:
            print(f"  [Subprocess] ❌ Failed to load checkpoint after training for {display_name}")
            return
            
        print(f"  [Subprocess] ✅ Checkpoint loaded successfully!")
        
        save_dir = Path("results_4060") / f"{model_name}_{views}"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [Subprocess] 🎨 Rendering perspective test views to {save_dir}...")
        
        start_time = time.time()
        rendered_imgs = nerf_model.render(
            mode=RenderMode.PERSPECTIVE,
            save_path=save_dir,
            downscale_factor=ds_factor,
        )
        elapsed_time = time.time() - start_time
        num_frames = len(rendered_imgs) if rendered_imgs else 0
        fps = num_frames / elapsed_time if elapsed_time > 0 and num_frames > 0 else 0.0
        print(f"  [Subprocess] ⚡ Rendered {num_frames} perspective images in {elapsed_time:.2f}s ({fps:.2f} FPS)!")
        
        print("  [Subprocess] 📊 Evaluating metrics...")
        metrics = nerf_model.evaluate_all_metrics(
            mode=RenderMode.PERSPECTIVE,
            metrics=[AvailableMetrics.PSNR, AvailableMetrics.SSIM, AvailableMetrics.LPIPS],
            downscale_factor=ds_factor
        )
        
        if metrics:
            metrics["fps"] = fps
            print(f"  [Subprocess] ✅ Metrics calculated successfully!")
            print(f"  [Subprocess] PSNR: {metrics.get('psnr'):.2f} | SSIM: {metrics.get('ssim'):.3f} | LPIPS: {metrics.get('lpips'):.3f}")
            
            out_json = Path("results_4060") / f"{model_name}_{views}_metrics.json"
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
        run_experiment(args.single_model, args.views)
        return
        
    experiments = [
        {"model": "nerfacto", "views": "0"}, # 100% das vistas para o Upper Bound (downscale 4)
        {"model": "nerfacto", "views": "3"},
        {"model": "nerfacto", "views": "6"},
        {"model": "nerfacto", "views": "10"},
        {"model": "instant-ngp", "views": "0"}, # 100% das vistas para o Upper Bound (downscale 4)
        {"model": "instant-ngp", "views": "3"},
        {"model": "instant-ngp", "views": "6"},
        {"model": "instant-ngp", "views": "10"},
    ]
    
    Path("results_4060").mkdir(parents=True, exist_ok=True)
    
    for exp in experiments:
        print(f"\n🔄 Spawning subprocess for {exp['model']} N={exp['views']} (RTX 4060)...")
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
