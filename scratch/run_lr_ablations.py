import os
import time
from pathlib import Path
from src.dataset.ImageDataset import ImageDataset
from src.nerf.NeRFModel import NerfModel
from src.utils.types import AvailableMetrics, RenderMode
import torch
import gc

def run_experiment(model_name, regime, ds, num_views, strategy, extra_args):
    dataset_timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_data_path = Path("assets/data/baseline/nerf_llff_data")
    
    print("\n" + "=" * 80)
    print(f"🚀 Iniciando Ablation: {model_name} | Dataset: {ds} | LR Extra Args: {extra_args}")
    print("=" * 80)

    dataset_obj = ImageDataset(
        name=ds,
        step="train",
        dataset_path=base_data_path / ds,
    )

    try:
        nerf_model = NerfModel(
            model_name=model_name,
            images=dataset_obj,
            split_strategy=strategy,
            num_views=num_views,
            llffhold=8,
            regime=regime,
            tta_steps=1000, # Using 1000 steps to be safe and give it time to converge
        )
        current_downscale = 8

        start_time = time.time()
        
        # Inject the learning rate through extra_cmd_args
        nerf_model.train(
            Path(dataset_obj.dataset_path),
            downscale_factor=current_downscale,
            extra_cmd_args=extra_args
        )
        
        training_time = time.time() - start_time

        # Avaliação de Métricas de Qualidade Visual e Perceptual (PSNR, SSIM, LPIPS)
        metrics = nerf_model.evaluate_all_metrics(
            mode=RenderMode.PERSPECTIVE, metrics=list(AvailableMetrics),
            downscale_factor=current_downscale
        )

        if metrics:
            print(f"✅ Sucesso: Métricas para {model_name}: PSNR = {metrics.get('psnr', 0):.2f}")
        else:
            print(f"⚠️ Falha: O modelo {model_name} retornou métricas vazias.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Erro crítico ao executar {model_name}: {str(e)}")

    finally:
        if 'nerf_model' in locals():
            del nerf_model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    # Experimento 1: GNT com LR 1e-4
    run_experiment(
        model_name="gnt", 
        regime="tta", 
        ds="fern", 
        num_views=10, 
        strategy="uniform", 
        extra_args=[
            "--optimizers.network.optimizer.lr", "1e-4", 
            "--optimizers.feature-net.optimizer.lr", "1e-4"
        ]
    )

    # Experimento 2: PixelNeRF com LR 1e-4
    run_experiment(
        model_name="pixel-nerf", 
        regime="tta", 
        ds="fern", 
        num_views=10, 
        strategy="uniform", 
        extra_args=[
            "--optimizers.encoder.optimizer.lr", "1e-4", 
            "--optimizers.nerf.optimizer.lr", "1e-4"
        ]
    )
