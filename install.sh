#!/bin/bash

PROJECT_PATH=$(pwd)

# --- UV Setup ---
if ! command -v uv &>/dev/null; then
  echo "You need uv for this project"
  echo "Want to install it now? (y/n)"
  read answer
  if [ "$answer" != "${answer#[Yy]}" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
  else
    echo "Exiting..."
    exit 1
  fi
fi

echo "Do you want to install the LLFF dataset (y/n)"
read answer_llff
if [ "$answer_llff" != "${answer_llff#[Yy]}" ]; then
  echo "Downloading LLFF dataset..."
  # Assuming Kaggle CLI is authenticated
  if command -v kaggle &>/dev/null; then
      mkdir -p assets/data/baseline/nerf_llff_data
      kaggle datasets download -d arenagrenade/llff-dataset-full -p assets/data/baseline/nerf_llff_data --unzip
  else
      echo "Kaggle CLI not found. Please download the LLFF dataset manually: https://www.kaggle.com/api/v1/datasets/download/arenagrenade/llff-dataset-full"
  fi
fi

# --- Environment ---
echo "Setting up unified environment..."
uv venv -p 3.11 .venv
source .venv/bin/activate
uv sync --active --prerelease=allow \
--upgrade-package nerfstudio --upgrade-package tinycudann --upgrade-package merf

# --- CUDA Extension Warmup ---
# Force nerfacc and gsplat to compile their CUDA kernels now, with the correct nvcc,
# so they don't fail on first import during training.
echo ""
echo "🔧 Pre-compiling CUDA extensions (nerfacc, gsplat)..."

# Auto-detect the newest local CUDA toolkit
CUDA_HOME_DETECTED=$(ls -d $HOME/.local/cuda-* /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
if [ -n "$CUDA_HOME_DETECTED" ]; then
  export CUDA_HOME=$CUDA_HOME_DETECTED
  export PATH=$CUDA_HOME/bin:$PATH
  echo "   Using CUDA toolkit: $CUDA_HOME"
else
  echo "⚠️  No local CUDA toolkit found. Extensions will compile at first runtime."
  echo "   Install CUDA 12.4 to ~/.local/cuda-12.4 to avoid this."
fi

python -c "
import sys
libs = [('nerfacc', 'nerfacc.cuda'), ('gsplat', 'gsplat.cuda')]
for name, mod in libs:
    try:
        __import__(mod)
        print(f'  ✅ {name} CUDA extension compiled successfully.')
    except ImportError as e:
        print(f'  ⚠️  {name} not installed, skipping: {e}')
    except Exception as e:
        print(f'  ❌ {name} compilation failed: {e}', file=sys.stderr)
        sys.exit(1)
"
deactivate
echo ""
echo "✅ Installation complete."
