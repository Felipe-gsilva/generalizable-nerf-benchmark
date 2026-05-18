from pathlib import Path
from itertools import product
import time
import torch
import gc

from src.dataset.ImageDataset import ImageDataset
from src.nerf.NeRFModel import NerfModel
from src.utils.metrics import MetricsLogger
from src.utils.config import config
from src.utils.types import AvailableMetrics, RenderMode


def main():
    datasets = config.datasets_name_list
    sampling_strategies = ["uniform", "random"]
    num_views_options = [3, 6, 10]

    experiments_config = [
        {"model": "instant-ngp", "regime": "per-scene"},
        {"model": "merf-ns", "regime": "per-scene"},
        {"model": "pixel-nerf", "regime": "zero-shot"},
        {"model": "pixel-nerf", "regime": "tta"},
        {"model": "gnt", "regime": "zero-shot"},
        {"model": "gnt", "regime": "tta"},
    ]

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    for ds, strategy, exp, num_views in product(
        datasets, sampling_strategies, experiments_config, num_views_options
    ):
        model_name = exp["model"]
        regime = exp["regime"]

        print("\n" + "=" * 80)
        print(
            f"🚀 Iniciando: {model_name} | Dataset: {ds} | Visões: {num_views} | Estratégia: {strategy}"
        )
        print("=" * 80)

        dataset_obj = ImageDataset(
            name=ds,
            step="train",
            dataset_path=Path("assets/data/baseline") / ds,
            llffhold=8,
            llffhold_split="train",
        )

        metrics_logger = MetricsLogger(
            model_name=f"{model_name}_{regime}",
            hyperparams={
                "sampling_strategy": strategy,
                "num_views": num_views,
                "dataset": ds,
                "regime": regime,
                "device": device_name,
            },
        )

        try:
            nerf_model = NerfModel(
                model_name=model_name,
                images=dataset_obj,
                split_strategy=strategy,
                num_views=num_views,
                llffhold=8,
                regime=regime,
                tta_steps=500,
            )

            # Medição rigorosa de tempo de convergência / preparação
            start_time = time.time()
            nerf_model.train(Path(dataset_obj.dataset_path), downscale_factor=1)
            training_time = time.time() - start_time

            # Avaliação de Métricas de Qualidade Visual e Perceptual (PSNR, SSIM, LPIPS)
            metrics = nerf_model.evaluate_all_metrics(
                mode=RenderMode.PERSPECTIVE, metrics=list(AvailableMetrics)
            )

            if metrics:
                # Injeta métricas de eficiência computacional calculadas no script
                metrics["convergence_time_seconds"] = training_time
                metrics_logger.log(**metrics)
                print(f"✅ Sucesso: Métricas registradas para {model_name}.")
            else:
                print(f"⚠️ Falha: O modelo {model_name} retornou métricas vazias.")

        except Exception as e:
            print(f"❌ Erro crítico ao executar {model_name} no dataset {ds}: {str(e)}")
            continue

        finally:
            metrics_logger.close()
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
