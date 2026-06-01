import os
import sys
import json
import time
import subprocess
from pathlib import Path

# --- MONKEY PATCHING ---
def patch_models():
    try:
        from src.nerf.PixelNeRFModel import PixelNeRFModel
        def unfrozen_pixelnerf(self):
            print("🚀 [UNFROZEN] PixelNeRF Encoder is fully trainable!")
            pass
        PixelNeRFModel.freeze_net = unfrozen_pixelnerf
    except ImportError:
        pass
        
    try:
        from GNTModel import GNTModel
        def unfrozen_gnt(self):
            print("🚀 [UNFROZEN] GNT Feature Network is fully trainable!")
            pass
        GNTModel._freeze_feature_net = unfrozen_gnt
    except ImportError:
        pass

# Execute the patch right away
patch_models()

from src.dataset.ImageDataset import ImageDataset
from src.nerf.NeRFModel import NerfModel
from src.utils.types import AvailableMetrics, RenderMode

NerfModel.measure_inference_fps = lambda self, *args, **kwargs: 0.0

def run_experiment(model_name, views):
    patch_models() # Make sure it's patched in subprocess
    
    base_data_path = Path("assets/data/baseline/nerf_llff_data")
    ds = "fern"
    num_views_val = int(views)
    display_name = f"{model_name} TTA N={views} (UNFROZEN)"
    
    print(f"\n🚀 [Subprocess] Starting 4060 UNFROZEN TTA for {display_name}...")
    dataset_obj = ImageDataset(
        name=ds,
        step="train",
        dataset_path=base_data_path / ds,
    )
    
    # 1. We override max-num-iterations to 1500 (since TTA regime defaults to 500)
    # 2. We lower train rays to 1024 to prevent OOM during full fine-tuning on 8GB VRAM
    # 3. We lower eval chunk to 512 for the same reason.
    extra_args = [
        "--max-num-iterations", "1500",
        "--pipeline.datamanager.train-num-rays-per-batch", "1024",
        "--pipeline.model.eval-num-rays-per-chunk", "512"
    ]
        
    try:
        nerf_model = NerfModel(
            model_name=model_name,
            images=dataset_obj,
            split_strategy="uniform",
            num_views=num_views_val,
            llffhold=8,
            regime="tta"
        )
        
        print(f"  [Subprocess] 🏋️ Training with downscale 2 and UNFROZEN features...")
        nerf_model.train(
            data_path=dataset_obj.dataset_path, 
            downscale_factor=2, 
            num_imgs=num_views_val,
            extra_cmd_args=extra_args
        )
        
        ckpt_dir = nerf_model.get_output_path(dataset_obj.dataset_path)
        print(f"  [Subprocess] 🔍 Loading trained model from {ckpt_dir}...")
        loaded = nerf_model.load_from_disk(ckpt_dir, mode="val", downscale_factor=2)
        
        if not loaded:
            print(f"  [Subprocess] ❌ Failed to load checkpoint after training for {display_name}")
            return
            
        print(f"  [Subprocess] ✅ Checkpoint loaded successfully!")
        
        save_dir = Path("results_4060_tta_unfrozen") / f"{model_name}_{views}"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [Subprocess] 🎨 Rendering perspective test views to {save_dir}...")
        
        start_time = time.time()
        rendered_imgs = nerf_model.render(
            mode=RenderMode.PERSPECTIVE,
            save_path=save_dir,
            downscale_factor=2,
        )
        elapsed_time = time.time() - start_time
        num_frames = len(rendered_imgs) if rendered_imgs else 0
        fps = num_frames / elapsed_time if elapsed_time > 0 and num_frames > 0 else 0.0
        print(f"  [Subprocess] ⚡ Rendered {num_frames} perspective images in {elapsed_time:.2f}s ({fps:.2f} FPS)!")
        
        print("  [Subprocess] 📊 Evaluating metrics...")
        metrics = nerf_model.evaluate_all_metrics(
            mode=RenderMode.PERSPECTIVE,
            metrics=[AvailableMetrics.PSNR, AvailableMetrics.SSIM, AvailableMetrics.LPIPS],
            downscale_factor=2
        )
        
        if metrics:
            metrics["fps"] = fps
            print(f"  [Subprocess] ✅ Metrics calculated successfully!")
            print(f"  [Subprocess] PSNR: {metrics.get('psnr'):.2f} | SSIM: {metrics.get('ssim'):.3f} | LPIPS: {metrics.get('lpips'):.3f}")
            
            out_json = Path("results_4060_tta_unfrozen") / f"{model_name}_{views}_metrics.json"
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
        {"model": "gnt", "views": "3"},
        {"model": "gnt", "views": "6"},
        {"model": "gnt", "views": "10"},
        {"model": "pixel-nerf", "views": "3"},
        {"model": "pixel-nerf", "views": "6"},
        {"model": "pixel-nerf", "views": "10"},
    ]
    
    Path("results_4060_tta_unfrozen").mkdir(parents=True, exist_ok=True)
    
    for exp in experiments:
        print(f"\n🔄 Spawning subprocess for {exp['model']} N={exp['views']} (RTX 4060 UNFROZEN)...")
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
