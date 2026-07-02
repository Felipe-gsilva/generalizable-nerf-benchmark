import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
from src.dataset.ImageDataset import ImageDataset
from src.nerf.NerfModel import NerfModel
from src.utils.metrics import MetricsLogger
from src.utils.types import AvailableMetrics, RenderMode

import time
import torch
import gc


def main():
    base_data_path = Path("assets/data/baseline/nerf_llff_data")
    llff_names = [
        "fern",
    ]
    sampling_strategies = ["uniform"
                           ,
                           #"random"
                           ]
    num_views_options = [3, 6, 10]

    experiments_config = [
        {"model": "gnt", "regime": "zero-shot"},
        {"model": "gnt", "regime": "tta"},
    ]

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    for ds in llff_names:
        dataset_timestamp = time.strftime("%Y%m%d_%H%M%S")
        for strategy in sampling_strategies:
            for exp in experiments_config:
                for num_views in num_views_options:
                    model_name = exp["model"]
                    regime = exp["regime"]
                    model_key = f"{model_name}_{regime}"

                    import glob
                    # Verifica se o experimento já foi concluído (se metrics.csv existe)
                    # if glob.glob(f"assets/logs/{model_key}/{ds}/*/{strategy}_{num_views}views/metrics.csv"):
                    #     print(f"⏭️  Pulando: {model_key} | Dataset: {ds} | Visões: {num_views} | Estratégia: {strategy} (Já executado)")
                    #     continue

                    print("\n" + "=" * 80)
                    print(
                        f"🚀 Iniciando: {model_name} | Dataset: {ds} | Visões: {num_views} | Estratégia: {strategy}"
                    )
                    print("=" * 80)

                    dataset_obj = ImageDataset(
                        name=ds,
                        step="train",
                        dataset_path=base_data_path / ds,
                    )

                    metrics_logger = MetricsLogger(
                        model_name=model_key,
                        group_dir=f"{model_key}/{ds}",
                        timestamp=dataset_timestamp,
                        run_id=f"{strategy}_{num_views}views",
                        hyperparams={
                            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
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
                            tta_steps=2500,
                        )
                        current_downscale = 8

                        if dataset_obj.dataset_path is None:
                            print(
                                f"⚠️ Aviso: O dataset {ds} não foi encontrado. Pulando esta configuração."
                            )
                            continue
                        # Medição rigorosa de tempo de convergência / preparação
                        start_time = time.time()
                        try:
                            nerf_model.train(
                                Path(dataset_obj.dataset_path),
                                downscale_factor=current_downscale,
                            )
                        except KeyboardInterrupt:
                            print(f"\n⚠️ Treinamento interrompido pelo usuário para {model_name}. Avaliando métricas parciais...")
                        except Exception as e:
                            print(f"\n⚠️ Treinamento abortado com erro ({e}). Pulando avaliação para evitar reuso de checkpoints corrompidos.")
                            continue
                        
                        training_time = time.time() - start_time

                        # Render perspective views first and save them to disk
                        rendered_views = nerf_model.render(
                            mode=RenderMode.PERSPECTIVE,
                            save_path=Path("assets/results")
                            / model_key
                            / ds
                            / f"{strategy}_{num_views}views",
                            downscale_factor=current_downscale,
                        )

                        # Avaliação de Métricas de Qualidade Visual e Perceptual (PSNR, SSIM, LPIPS)
                        metrics = nerf_model.evaluate_all_metrics(
                            mode=RenderMode.PERSPECTIVE,
                            metrics=list(AvailableMetrics),
                            already_rendered_images=rendered_views,
                            downscale_factor=current_downscale,
                        )

                        if metrics:
                            # Injeta métricas de eficiência computacional calculadas no script
                            metrics["convergence_time_seconds"] = training_time
                            metrics_logger.log(**metrics)
                            print(
                                f"✅ Sucesso: Métricas registradas para {model_name}."
                            )
                        else:
                            print(
                                f"⚠️ Falha: O modelo {model_name} retornou métricas vazias."
                            )

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(
                            f"❌ Erro crítico ao executar {model_name} no dataset {ds}: {str(e)}"
                        )
                        continue

                    finally:
                        metrics_logger.close()
                        if 'nerf_model' in locals():
                            del nerf_model
                        torch.cuda.empty_cache()
                        gc.collect()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Erro inesperado no script principal: {str(e)}")
