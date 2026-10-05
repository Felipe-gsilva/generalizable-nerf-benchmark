#!/usr/bin/env bash
# ==============================================================================
# Automated Native Environment Installer for Generalizable NeRF Benchmark
# Replicates the EXACT steps, dependencies, flags, and build environment of Dockerfile
# Supported Distributions: Ubuntu / Debian & Arch Linux
# ==============================================================================

set -euo pipefail

log_info() {
    echo -e "\033[1;34m[INFO]\033[0m $*"
}

log_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $*"
}

log_warn() {
    echo -e "\033[1;33m[WARNING]\033[0m $*"
}

log_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $*" >&2
}

# Determine privilege escalation tool
SUDO=""
if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    if command -v sudo &>/dev/null; then
        SUDO="sudo"
    else
        log_error "This script requires superuser privileges to install system packages. Please run as root or install sudo."
        exit 1
    fi
fi

# ------------------------------------------------------------------------------
# 0. OS Detection
# ------------------------------------------------------------------------------
log_info "Detecting Operating System..."
OS_TYPE=""
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_ID="${ID:-}"
    OS_LIKE="${ID_LIKE:-}"
else
    log_error "Could not determine Linux distribution (/etc/os-release not found)."
    exit 1
fi

if [[ "$OS_ID" =~ ^(ubuntu|debian|linuxmint|pop)$ ]] || [[ "$OS_LIKE" =~ (ubuntu|debian) ]]; then
    OS_TYPE="debian"
    log_info "Detected Debian/Ubuntu-based distribution: $PRETTY_NAME"
elif [[ "$OS_ID" =~ ^(arch|manjaro|endeavouros|garuda|archarm)$ ]] || [[ "$OS_LIKE" =~ arch ]]; then
    OS_TYPE="arch"
    log_info "Detected Arch Linux-based distribution: $PRETTY_NAME"
else
    log_error "Unsupported distribution: $OS_ID ($PRETTY_NAME). Supported: Ubuntu/Debian and Arch Linux."
    exit 1
fi

# ------------------------------------------------------------------------------
# 1. Environment & GPU Architecture Detection
# ------------------------------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export UV_LINK_MODE=copy

if [ -z "${GPU_ARCH:-}" ]; then
    if command -v nvidia-smi &>/dev/null; then
        GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | sed 's/\.//g')
        log_info "🚀 Auto-detected GPU Architecture: $GPU_ARCH"
    else
        log_warn "nvidia-smi not found. Defaulting GPU_ARCH to 86 (RTX 30xx/A100/etc.)"
        GPU_ARCH=86
    fi
fi
export GPU_ARCH

# ------------------------------------------------------------------------------
# 2. System Dependencies Installation (Exact Dockerfile Step 2)
# ------------------------------------------------------------------------------
log_info "📦 Installing system dependencies..."

if [ "$OS_TYPE" = "debian" ]; then
    $SUDO apt-get update
    $SUDO apt-get install -y \
        intel-mkl \
        libceres-dev \
        libsuitesparse-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libfreeimage-dev \
        liblz4-dev \
        libmetis-dev \
        libeigen3-dev \
        libflann-dev \
        libsqlite3-dev \
        libglew-dev \
        libcgal-dev \
        qt6-base-dev \
        ffmpeg xvfb \
        git build-essential cmake ninja-build \
        libgl1 libglib2.0-0 \
        libsm6 libxext6 libxrender1 \
        curl ca-certificates || {
            log_warn "Some Debian package names may vary on your release, retrying with fallback packages..."
            $SUDO apt-get install -y \
                libmkl-dev || true
        }

elif [ "$OS_TYPE" = "arch" ]; then
    $SUDO pacman -Syu --needed --noconfirm \
        base-devel \
        cmake \
        ninja \
        git \
        ceres-solver \
        suitesparse \
        boost \
        gflags \
        google-glog \
        lz4 \
        eigen \
        flann \
        sqlite \
        glew \
        cgal \
        qt6-base \
        ffmpeg \
        xorg-server-xvfb \
        libglvnd \
        glib2 \
        libsm \
        libxext \
        libxrender \
        curl \
        ca-certificates \
        cuda

    # Install MKL or BLAS/LAPACK on Arch (equivalent to intel-mkl)
    $SUDO pacman -S --needed --noconfirm intel-oneapi-mkl 2>/dev/null || $SUDO pacman -S --needed --noconfirm blas lapack || true

    # Install FreeImage and METIS (exact equivalents to libfreeimage-dev & libmetis-dev)
    if ! pacman -Qi freeimage &>/dev/null || ! pacman -Qi metis &>/dev/null; then
        log_info "Installing FreeImage and METIS for COLMAP on Arch Linux..."
        if command -v yay &>/dev/null; then
            yay -S --needed --noconfirm freeimage metis
        elif command -v paru &>/dev/null; then
            paru -S --needed --noconfirm freeimage metis
        else
            if ! pacman -Qi metis &>/dev/null; then
                log_info "Compiling METIS from source..."
                rm -rf /tmp/metis-build
                git clone --depth 1 https://github.com/KarypisLab/METIS.git /tmp/metis-build
                make -C /tmp/metis-build config shared=1
                make -C /tmp/metis-build -j2
                $SUDO make -C /tmp/metis-build install
                rm -rf /tmp/metis-build
            fi
            if ! pacman -Qi freeimage &>/dev/null; then
                log_info "Building FreeImage from AUR..."
                rm -rf /tmp/freeimage-aur
                git clone https://aur.archlinux.org/freeimage.git /tmp/freeimage-aur
                (cd /tmp/freeimage-aur && makepkg -si --noconfirm) || log_warn "Could not auto-install freeimage. If COLMAP fails, install 'freeimage' from AUR manually."
                rm -rf /tmp/freeimage-aur
            fi
        fi
    fi
fi

log_success "System dependencies installed."

# ------------------------------------------------------------------------------
# 3. Setup CUDA Toolkit Path
# ------------------------------------------------------------------------------
if [ -z "${CUDA_HOME:-}" ]; then
    if [ -d "/opt/cuda" ]; then
        export CUDA_HOME="/opt/cuda"
    elif [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    elif command -v nvcc &>/dev/null; then
        export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    fi
fi

if [ -n "${CUDA_HOME:-}" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    log_info "⚡ Using CUDA_HOME=$CUDA_HOME"
fi

# ------------------------------------------------------------------------------
# 4. Compile and Install COLMAP 3.10 with CUDA (Exact Dockerfile Step 3)
# ------------------------------------------------------------------------------
if command -v colmap &>/dev/null && colmap --help | grep -i "3.10" &>/dev/null; then
    log_info "COLMAP 3.10 is already installed on the system. Skipping compilation."
else
    log_info "Building and installing COLMAP 3.10 (CUDA enabled, GPU_ARCH=${GPU_ARCH})..."
    rm -rf /tmp/colmap
    git clone --depth 1 --branch 3.10 https://github.com/colmap/colmap.git /tmp/colmap

    cmake \
        -S /tmp/colmap \
        -B /tmp/colmap/build \
        -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${GPU_ARCH}" \
        -DCUDA_ENABLED=ON \
        -DGUI_ENABLED=OFF \
        -DTESTS_ENABLED=OFF \
        -DUSE_OPENMP=ON \
        -DFETCHCONTENT_FULLY_DISCONNECTED=OFF \
        -DOPENIMAGEIO_ENABLED=OFF

    $SUDO cmake --build /tmp/colmap/build --target install -- -j2
    rm -rf /tmp/colmap
    $SUDO ldconfig || true
    log_success "COLMAP 3.10 installed successfully."
fi

# ------------------------------------------------------------------------------
# 5. Install uv (Exact Dockerfile Step 4)
# ------------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    log_info "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
log_success "uv is available at: $(which uv)"

# ------------------------------------------------------------------------------
# 6. Python 3.11 & Virtual Environment Setup (Exact Dockerfile Step 5)
# ------------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VIRTUAL_ENV="${VIRTUAL_ENV:-$PROJECT_ROOT/.venv}"
export VIRTUAL_ENV
export UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV"

log_info "🐍 Setting up Python 3.11 virtual environment at: $VIRTUAL_ENV..."
uv python install 3.11
uv venv --python 3.11 "$VIRTUAL_ENV"

export PATH="$VIRTUAL_ENV/bin:$PATH"
log_success "Virtual environment activated: $(python3 --version)"

# ------------------------------------------------------------------------------
# 7. Project Dependencies Synchronization (Exact Dockerfile Step 6)
# ------------------------------------------------------------------------------
log_info "🔄 Synchronizing project dependencies via uv sync..."
uv sync --prerelease=allow --no-install-project \
    --upgrade-package=nerfstudio-gnt \
    --upgrade-package=nerfstudio-pixelnerf

# ------------------------------------------------------------------------------
# 8. Tiny-CUDA-NN Build & Install (Exact Dockerfile Step 7)
# ------------------------------------------------------------------------------
log_info "⚡ Compiling Tiny-CUDA-NN bindings (GPU_ARCH=${GPU_ARCH})..."
export TCNN_CUDA_ARCHITECTURES="$GPU_ARCH"
export MAX_JOBS="${MAX_JOBS:-2}"

uv pip install pip "setuptools<70.0.0" wheel ninja
uv pip install -v git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation
log_success "Tiny-CUDA-NN compiled and installed."

# ------------------------------------------------------------------------------
# 9. Install Current Project in Editable Mode (Exact Dockerfile Step 8)
# ------------------------------------------------------------------------------
log_info "📦 Installing current project in editable mode..."
uv pip install --no-deps -e .
log_success "Project installed in editable mode."

# ------------------------------------------------------------------------------
# 10. Optional: Dataset Download
# ------------------------------------------------------------------------------
if [ -t 0 ]; then
    echo ""
    read -r -p "Do you want to download the LLFF dataset now? (y/N): " answer_llff
    if [[ "$answer_llff" =~ ^[Yy]$ ]]; then
        log_info "Downloading LLFF dataset via Kaggle..."
        if command -v kaggle &>/dev/null; then
            mkdir -p assets/data/baseline/
            kaggle datasets download -d arenagrenade/llff-dataset-full -p assets/data/baseline/ --unzip
            log_success "LLFF dataset downloaded to assets/data/baseline/"
        else
            log_warn "Kaggle CLI not found. Download manually from: https://www.kaggle.com/datasets/arenagrenade/llff-dataset-full"
        fi
    fi
fi

echo ""
log_success "🎉 Installation completed successfully!"
echo "To activate your environment, run:"
echo "    source $VIRTUAL_ENV/bin/activate"
echo "To launch the benchmark:"
echo "    python main.py"
