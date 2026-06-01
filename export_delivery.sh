#!/bin/bash

# Define the output zip file name
OUTPUT_FILE="nerf_experiment_delivery.zip"

echo "🧹 Cleaning up unnecessary cache folders..."
rm -rf __pycache__ .pytest_cache .ruff_cache scratch .venv/lib/python*/site-packages/__pycache__

echo "📦 Zipping the project into $OUTPUT_FILE..."

# Zip the project, excluding large data folders, virtual environments, and Git history
zip -r $OUTPUT_FILE . \
    -x "*.git*" \
    -x "*.venv*" \
    -x "*__pycache__*" \
    -x "*.pytest_cache*" \
    -x "*.ruff_cache*" \
    -x "assets/data/*" \
    -x "assets/nerf_checkpoints/*" \
    -x "assets/logs/*" \
    -x "assets/pretrained/*" \
    -x "results/*" \
    -x "*.DS_Store" \
    -x "scratch/*" \
    -x "uv.lock"

echo "✅ Delivery package created successfully: $OUTPUT_FILE"
