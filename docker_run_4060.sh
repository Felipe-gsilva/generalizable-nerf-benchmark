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
    bash -c '
set -e
echo "Verificando modelo pré-treinado do GNT..."
if [ ! -f "assets/pretrained/gnt_pretrained.pth" ]; then
    echo "Baixando modelo pré-treinado..."
    mkdir -p assets/pretrained
    python -c "from gdown import download; download('\''https://drive.google.com/file/d/1YvOJXa5eGpKgoMYcxC2ma7prB1n5UwRn/'\'', '\''assets/pretrained/gnt_pretrained.pth'\'', quiet=False)"
fi

echo "Ajustando caminhos absolutos no config.yml do GNT..."
CONFIG_PATH="assets/data/nerf_checkpoints/fern/fern/gnt/2026-06-02_001426/config.yml"
if [ -f "$CONFIG_PATH" ]; then
    python -c "
import re
with open('\''$CONFIG_PATH'\'', '\''r+'\'') as f:
    text = f.read()
    # Substitui caminhos absolutos do host para /workspace
    text = re.sub(r'\''/[a-zA-Z0-9_\\-\\./]+/nerf-ann-paper'\'', '\''/workspace'\'', text)
    f.seek(0)
    f.write(text)
    f.truncate()
"
    echo "Avaliando métricas do GNT (TTA)..."
    ns-eval --load-config "$CONFIG_PATH" --output-path results_gnt_tta.json
else
    echo "ERRO: O arquivo $CONFIG_PATH não foi encontrado!"
fi
'

echo ""
echo "✅ Experimento finalizado!"
