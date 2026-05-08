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
uv sync --upgrade --prerelease=allow
