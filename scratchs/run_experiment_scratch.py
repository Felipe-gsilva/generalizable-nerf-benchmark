import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from pathlib import Path
from src.dataset.ImageDataset import ImageDataset
from src.nerf.NeRFModel import NerfModel
from src.utils.metrics import MetricsLogger
from src.utils.types import AvailableMetrics, RenderMode

import time
import torch
import gc

def main():
    base_data_path = Path("assets/data/baseline/nerf_llff_data")
    
    # --- Configuração Fixa (Scratch) ---
    ds = "fern"
    strategy = "uniform"
    num_views = 3
    models_to_run = ["gnt"]
    regime = "tta"
    # -----------------------------------

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    dataset_timestamp = time.strftime("%Y%m%d_%H%M%S")

    for model_name in models_to_run:
        model_key = f"{model_name}_{regime}"

        print("\n" + "=" * 80)
        print(f"🚀 Iniciando SCRATCH: {model_name} | Dataset: {ds} | Visões: {num_views} | Estratégia: {strategy}")
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
            run_id=f"scratch_{strategy}_{num_views}views",
            hyperparams={
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sampling_strategy": strategy,
                "num_views": num_views,
                "dataset": ds,
                "regime": regime,
                "device": device_name,
                "is_scratch": True
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
                tta_steps=2500, # Adaptação de 2500 passos
            )
            current_downscale = 8

            if dataset_obj.dataset_path is None:
                print(f"⚠️ Aviso: O dataset {ds} não foi encontrado.")
                continue

            start_time = time.time()
            
            # print(f">> Fase 1: Treinamento / Adaptação (TTA) para {model_name}")
            # nerf_model.train(
            #     Path(dataset_obj.dataset_path),
            #     downscale_factor=current_downscale,
            # )
            
            training_time = time.time() - start_time
            # print(f">> Fase 1 Concluída para {model_name}. Tempo de convergência: {training_time:.2f}s")

            print(f">> Fase 2: Avaliação de Métricas Finais para {model_name}")
            metrics = nerf_model.evaluate_all_metrics(
                mode=RenderMode.PERSPECTIVE, metrics=list(AvailableMetrics),
                downscale_factor=current_downscale
            )

            if metrics:
                metrics["convergence_time_seconds"] = training_time
                metrics_logger.log(**metrics)
                print(f"✅ Sucesso: Métricas registradas para {model_name}!")
            else:
                print(f"⚠️ Falha: O modelo {model_name} retornou métricas vazias.")

            # print(f">> Fase 3: Renderização de View de Teste para {model_name}")
            # nerf_model.render(
            #     mode=RenderMode.PERSPECTIVE,
            #     save_path=Path("results") / model_key / ds / f"scratch_{strategy}_{num_views}views",
            #     downscale_factor=current_downscale,
            # )

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Erro crítico no modelo {model_name}: {str(e)}")

        finally:
            metrics_logger.close()
            if 'nerf_model' in locals():
                del nerf_model
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    main()
