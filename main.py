from pathlib import Path
from itertools import product

from src.dataset.ImageDataset import ImageDataset
from src.nerf.NeRFModel import NerfModel
from src.utils.metrics import MetricsLogger
from src.utils.config import config
from src.utils.types import AvailableMetrics, RenderMode


def main():
    datasets = config.datasets_name_list
    sampling_strategies = ["uniform", "random"]
    num_views_options = [3, 6, 10]
    models = config.nerf_models_to_run

    for ds, strategy, model_name, num_views in product(
        datasets, sampling_strategies, models, num_views_options
    ):
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

        metrics = nerf_model.evaluate_all_metrics(
            mode=RenderMode.PERSPECTIVE,
            metrics=list(AvailableMetrics) 
        )


        if metrics:
            metrics_logger.log(**metrics)
        else:
            print(f"⚠️ Failed to evaluate metrics for {model_name} ({num_views} views).")

        metrics_logger.close()


if __name__ == "__main__":
    main()
