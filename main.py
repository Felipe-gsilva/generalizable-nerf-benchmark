from pathlib import Path
from itertools import product
import json
import numpy as np

from src.dataset.ImageDataset import ImageDataset
from src.nerf.NeRFModel import NerfModel
from src.utils.metrics import MetricsLogger
from src.utils.config import config
from utils.types import AvailableMetrics, Plans


def main():
    print("Hello from nerf-ann-paper!")
    datasets = config.datasets_name_list
    sampling_strategies = ["uniform", "random"]
    num_views_options = [3, 6, 10]
    models = config.nerf_models_to_run

    for ds, strategy, model_name, num_views in product(
        datasets, sampling_strategies, models, num_views_options
    ):
        # 1. CORREÇÃO: Nomeamos a variável do dataset como 'dataset_obj' para evitar conflito
        dataset_obj = ImageDataset(
            name=ds, step="train", dataset_path=Path("assets/data/baseline") / ds
        )

        metrics_logger = MetricsLogger(
            model_name=model_name,
            hyperparams={
                "sampling_strategy": strategy,
                "num_views": num_views,
                "dataset": ds,
            },
        )

        print(
            f"🚀 Running {model_name} with {strategy} sampling strategy and {num_views} views."
        )

        nerf_model = NerfModel(
            model_name=model_name,
            images=dataset_obj,
            split_strategy=strategy,
            num_views=num_views,
        )

        nerf_model.train(Path(dataset_obj.dataset_path), downscale_factor=1)

        rendered_images = nerf_model.render(num_slices=10, plan=Plans.AXIAL)

        memory_footprint = nerf_model.get_memory_footprint()

        metrics = nerf_model.evaluate_rendered_images_quality(
            rendered_images=rendered_images,
            metrics=[
                AvailableMetrics.FID,
                AvailableMetrics.PSNR,
                AvailableMetrics.SSIM,
                AvailableMetrics.LPIPS,
            ],
        )

        if metrics:
            metrics_logger.log(**metrics, memory_footprint=memory_footprint)
        else:
            print(f"⚠️ Failed to evaluate metrics for {model_name} ({num_views} views).")

        metrics_logger.close()


if __name__ == "__main__":
    main()
