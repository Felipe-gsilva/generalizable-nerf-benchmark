#!/bin/bash
set -e

echo "=========================================================="
echo "🚀 Building Docker image for RTX 4060 (Compute Capability 89)"
echo "=========================================================="
# A RTX 4060 usa arquitetura Ada Lovelace (89). Compilar o TCNN e Colmap para 89 garante performance máxima.
docker build -t nerf-ann-paper:4060 --build-arg GPU_ARCH="89" .

echo ""
echo "=========================================================="
echo "🏃 Running 4060 Optimized Experiments via Docker..."
echo "=========================================================="
# O volume (-v) repassa as pastas atuais (assets, results_4060, etc.) para o container.
docker run --gpus all --rm -it \
    -v "$(pwd)":/workspace \
    -w /workspace \
    -e PYTHONPATH=/workspace \
    -e TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    nerf-ann-paper:4060 \
    python scratch/run_4060_experiments.py
    
echo ""
echo "=========================================================="
echo "🔥 Running 4060 TTA UNFROZEN Experiments via Docker..."
echo "=========================================================="
docker run --gpus all --rm -it \
    -v "$(pwd)":/workspace \
    -w /workspace \
    -e PYTHONPATH=/workspace \
    -e TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    nerf-ann-paper:4060 \
    python scratch/run_4060_tta_unfrozen.py

echo "✅ Todos os experimentos finalizados! Resultados salvos nas pastas results_4060/ e results_4060_tta_unfrozen/"
