import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from pathlib import Path
from src.dataset.ImageDataset import ImageDataset
from src.nerf.NerfModel import NerfModel
from src.utils.metrics import MetricsLogger
from src.utils.types import AvailableMetrics, RenderMode

import time
import torch
import gc
import glob

def main():
    base_data_path = Path("assets/data/baseline/nerf_llff_data")
    llff_names = [
        "room",
        "trex",
    ]
    sampling_strategies = ["uniform"]
    num_views_options = [3, 6, 10]

    # Models to test
    models = ["gnt", "pixel-nerf"]
    
    # LRs to test: 1e-4, 5e-4, and default (None)
    # The user said 10e-4 which usually means 1e-4 in their context, but literally means 1e-3. 
    # I will include 1e-3, 1e-4, 5e-4 and default just to be safe.
    lrs = {
        "default": None,
        "1e-4": 1e-4,
        "5e-4": 5e-4,
        "1e-3": 1e-3
    }

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    for ds in llff_names:
        for num_views in num_views_options:
            for strategy in sampling_strategies:
                for model_name in models:
                    for lr_name, lr_val in lrs.items():
                        
                        regime = "tta"
                        model_key = f"{model_name}_{regime}_{lr_name}"
                        
                        # Check if already executed
                        if glob.glob(f"assets/logs/{model_key}/{ds}/*/{strategy}_{num_views}views/metrics.csv"):
                            print(f"⏭️  Pulando: {model_key} | Dataset: {ds} | Visões: {num_views} (Já executado)")
                            continue
                            
                        print("\n" + "=" * 80)
                        print(f"🚀 Iniciando Ablation: {model_key} | Dataset: {ds} | Visões: {num_views}")
                        print("=" * 80)

                        dataset_timestamp = time.strftime("%Y%m%d_%H%M%S")
                        
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
                                "lr_ablation": lr_name,
                                "lr_val": lr_val,
                                "device": device_name,
                            },
                        )
                        
                        extra_args = []
                        if lr_val is not None:
                            if model_name == "gnt":
                                extra_args += [
                                    "--optimizers.network.optimizer.lr", str(lr_val),
                                    "--optimizers.feature-net.optimizer.lr", str(lr_val),
                                ]
                            elif model_name == "pixel-nerf":
                                extra_args += [
                                    "--optimizers.encoder.optimizer.lr", str(lr_val),
                                    "--optimizers.nerf.optimizer.lr", str(lr_val),
                                ]

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
                                print(f"⚠️ Aviso: O dataset {ds} não foi encontrado.")
                                continue
                                
                            start_time = time.time()
                            try:
                                nerf_model.train(
                                    Path(dataset_obj.dataset_path),
                                    downscale_factor=current_downscale,
                                    extra_cmd_args=extra_args
                                )
                            except KeyboardInterrupt:
                                print(f"\n⚠️ Interrompido pelo usuário.")
                            except Exception as e:
                                print(f"\n⚠️ Abortado com erro ({e}). Avaliando último checkpoint...")
                            
                            training_time = time.time() - start_time

                            metrics = nerf_model.evaluate_all_metrics(
                                mode=RenderMode.PERSPECTIVE, 
                                metrics=list(AvailableMetrics),
                                downscale_factor=current_downscale
                            )

                            if metrics:
                                metrics["convergence_time_seconds"] = training_time
                                metrics_logger.log(**metrics)
                                print(f"✅ Sucesso: Métricas registradas para {model_key}.")
                            else:
                                print(f"⚠️ Falha: Métricas vazias para {model_key}.")

                            nerf_model.render(
                                mode=RenderMode.PERSPECTIVE,
                                save_path=Path("assets/results") / model_key / ds / f"{strategy}_{num_views}views",
                                downscale_factor=current_downscale,
                            )

                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            print(f"❌ Erro crítico: {str(e)}")
                            continue

                        finally:
                            metrics_logger.close()
                            if 'nerf_model' in locals():
                                del nerf_model
                            torch.cuda.empty_cache()
                            gc.collect()

if __name__ == "__main__":
    main()
