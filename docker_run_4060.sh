#!/bin/bash
set -e

echo "=========================================================="
echo "🚀 Building Docker image for RTX 4060 (Compute Capability 89)"
echo "=========================================================="
# A RTX 4060 usa arquitetura Ada Lovelace (89). Compilar o TCNN e Colmap para 89 garante performance máxima.
docker build -t nerf-ann-paper:4060 --build-arg GPU_ARCH="89" .
    
echo ""
echo "=========================================================="
echo "🔥 Running Scratch Experiment via Docker..."
echo "=========================================================="
docker run --gpus all --rm -it \
    -v "$(pwd)":/workspace \
    -w /workspace \
    -e PYTHONPATH=/workspace \
    -e TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    -e OMP_NUM_THREADS=1 \
    nerf-ann-paper:4060 \
    python scratchs/run_experiment_scratch.py

echo ""
echo "✅ Experimento finalizado!"
