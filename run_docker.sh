#!/bin/bash

# Auto-detect GPU architecture if not set
if [ -z "$GPU_ARCH" ]; then
    if command -v nvidia-smi &> /dev/null; then
        export GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | sed 's/\.//g')
        echo "🚀 Auto-detected GPU Architecture: $GPU_ARCH"
    else
        echo "⚠️ nvidia-smi not found, defaulting GPU_ARCH to 86 (RTX 30xx/A100/etc.)"
        export GPU_ARCH=86
    fi
fi

# Build and run
COMPOSE_BAKE=true docker compose up --build -d
