import os
import shutil
import time
import glob
import gc
from pathlib import Path

# Configurações do ambiente para evitar bloqueios do PyTorch e otimizar VRAM
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Auto-detecção do CUDA toolkit para compilações JIT e extensões (gsplat / tiny-cuda-nn)
if "CUDA_HOME" not in os.environ:
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        os.environ["CUDA_HOME"] = str(Path(nvcc_path).resolve().parent.parent)
    else:
        for candidate in ["/usr/local/cuda", "/usr/local/cuda-12.6", "/usr/local/cuda-12", "/opt/cuda"]:
            if os.path.exists(candidate):
                os.environ["CUDA_HOME"] = candidate
                break

if "CUDA_HOME" in os.environ:
    cuda_home = os.environ["CUDA_HOME"]
    cuda_bin = f"{cuda_home}/bin"
    cuda_lib = f"{cuda_home}/lib64"
    current_path = os.environ.get("PATH", "")
    if cuda_bin not in current_path.split(":"):
        os.environ["PATH"] = f"{cuda_bin}:{current_path}"
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if cuda_lib not in current_ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib}:{current_ld}"

import torch
from src.dataset.ImageDataset import ImageDataset
from src.nerf.NerfModel import NerfModel
from src.utils.metrics import MetricsLogger
from src.utils.types import AvailableMetrics, RenderMode


def main():
    base_data_path = Path("assets/data/baseline/nerf_llff_data")
    llff_names = [
        "fern",
        "flower",
        "fortress",
        "horns",
        "leaves",
        "orchids",
        "room",
        "trex",
    ]
    sampling_strategies = [
        "uniform",  # "random"
    ]
    num_views_options = [3, 6, 10]

    # Benchmark expandido com Splatfacto (Gaussian Splatting) e Nerfbusters
    experiments_config = [
        {"model": "splatfacto", "regime": "per-scene"},
        {"model": "nerfbusters", "regime": "per-scene"},
        # Modelos de referência do benchmark anterior (descomente conforme necessário):
        # {"model": "nerfacto", "regime": "per-scene"},
        # {"model": "instant-ngp", "regime": "per-scene"},
        # {"model": "gnt", "regime": "zero-shot"},
        # {"model": "gnt", "regime": "tta"},
        # {"model": "pixel-nerf", "regime": "zero-shot"},
        # {"model": "pixel-nerf", "regime": "tta"},
    ]

    # Pular experimentos já concluídos com sucesso (metrics.csv existente e não-vazio)
    skip_existing = True

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    for ds in llff_names:
        dataset_timestamp = time.strftime("%Y%m%d_%H%M%S")
        for strategy in sampling_strategies:
            for exp in experiments_config:
                for num_views in num_views_options:
                    model_name = exp["model"]
                    regime = exp["regime"]
                    model_key = f"{model_name}_{regime}"

                    # Verifica se o experimento já foi concluído anteriormente
                    if skip_existing:
                        existing_logs = glob.glob(
                            f"assets/logs/{model_key}/{ds}/*/{strategy}_{num_views}views/metrics.csv"
                        )
                        if any(os.path.getsize(f) > 0 for f in existing_logs):
                            print(
                                f"⏭️  Pulando: {model_key} | Dataset: {ds} | Visões: {num_views} | Estratégia: {strategy} (Já executado)"
                            )
                            continue

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

                        extra_args = []
                        if regime == "tta":
                            lr_val = 5e-4
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
                        elif regime == "per-scene":
                            if model_name in ["splatfacto", "nerfbusters", "nerfacto"]:
                                # Em condições de sparse views (N <= 10), desativar o otimizador de câmera evita colapso da pose
                                extra_args += ["--pipeline.model.camera-optimizer.mode", "off"]

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
                                extra_cmd_args=extra_args
                            )
                        except KeyboardInterrupt:
                            print(f"\n⚠️ Treinamento interrompido pelo usuário para {model_name}. Avaliando métricas parciais...")
                        except Exception as e:
                            print(f"\n❌ Treinamento abortado com erro para {model_name} ({e}). Pulando avaliação para evitar reuso de checkpoints corrompidos.")
                            continue

                        training_time = time.time() - start_time

                        # Renderização de visões em perspectiva (salvas em disco)
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
