import os
import torch
import time
import sys
import json
from pathlib import Path

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

try:
    import GNTModel
    old_load_pretrained = GNTModel.GNTModel._load_pretrained
    def new_load_pretrained(self, ckpt_path):
        if isinstance(ckpt_path, str) and ckpt_path.startswith("/workspace/"):
            ckpt_path = ckpt_path.replace("/workspace/", "/home/felipe-gsilva/dev/cs/nerf-ann-paper/")
        return old_load_pretrained(self, ckpt_path)
    GNTModel.GNTModel._load_pretrained = new_load_pretrained
except:
    pass

from src.dataset.ImageDataset import ImageDataset
from src.nerf.NeRFModel import NerfModel
from src.utils.types import AvailableMetrics, RenderMode

NerfModel.measure_inference_fps = lambda self, *args, **kwargs: 0.0

RANDOM_MAPPING = {
    "pixel-nerf": "assets/data/nerf_checkpoints/fern/fern/pixel-nerf/2026-06-01_054435",
    "gnt": "assets/data/nerf_checkpoints/fern/fern/gnt/2026-05-31_200329",
    "instant-ngp": "assets/data/nerf_checkpoints/fern/fern/instant-ngp/2026-06-01_032214",
    "nerfacto": "assets/data/nerf_checkpoints/fern/fern/nerfacto/2026-06-01_033909",
}

def evaluate_random(model_name):
    ds = "fern"
    ckpt_dir = Path(RANDOM_MAPPING[model_name])
    regime = "tta" if model_name in ["pixel-nerf", "gnt"] else "per-scene"
    
    dataset_obj = ImageDataset(name=ds, step="train", dataset_path=Path("assets/data/baseline/nerf_llff_data") / ds)
    
    nerf_model = NerfModel(
        model_name=model_name,
        images=dataset_obj,
        split_strategy="random",
        num_views=10,
        llffhold=8,
        regime=regime,
    )
    
    loaded = nerf_model.load_from_disk(ckpt_dir, mode="val", downscale_factor=4)
    if not loaded:
        print(f"Failed {model_name}")
        return
        
    save_dir = Path("results_random_10") / f"{model_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = nerf_model.evaluate_all_metrics(
        mode=RenderMode.PERSPECTIVE,
        metrics=[AvailableMetrics.PSNR, AvailableMetrics.SSIM, AvailableMetrics.LPIPS],
        downscale_factor=4
    )
    
    if metrics:
        with open(f"results_random_10/{model_name}_metrics.json", "w") as f:
            json.dump(metrics, f)
        print(f"{model_name} -> PSNR: {metrics['psnr']:.2f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        evaluate_random(sys.argv[1])
    else:
        Path("results_random_10").mkdir(exist_ok=True)
        import subprocess
        for model in RANDOM_MAPPING:
            print(f"Running {model}...")
            env = os.environ.copy()
            env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":" + os.getcwd()
            subprocess.run([sys.executable, __file__, model], env=env)
