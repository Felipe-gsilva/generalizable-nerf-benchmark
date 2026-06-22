# 1. Base Image (Removed the unused colmap-bin stage)
FROM nvidia/cuda:12.6.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV IS_DOCKER=1

# 2. System Dependencies (Removed apt 'colmap' to avoid binary conflicts)
RUN apt-get update && apt-get install -y \
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
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG GPU_ARCH="86"
RUN git clone --depth 1 --branch 3.10 \
      https://github.com/colmap/colmap.git /tmp/colmap \
 && cmake \
      -S /tmp/colmap \
      -B /tmp/colmap/build \
      -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=${GPU_ARCH} \
      -DCUDA_ENABLED=ON \
      -DGUI_ENABLED=OFF \
      -DTESTS_ENABLED=OFF \
      -DUSE_OPENMP=ON \
      -DFETCHCONTENT_FULLY_DISCONNECTED=OFF \
      -DOPENIMAGEIO_ENABLED=OFF \
 && cmake --build /tmp/colmap/build --target install -- -j2 \
 && rm -rf /tmp/colmap \
 && ldconfig

# 4. Install uv via Docker
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

# 5. Python & Virtual Environment Setup
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
RUN uv python install 3.11
RUN uv venv --python 3.11 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 6. Project Dependencies (Cache Layer)
COPY pyproject.toml ./
RUN uv sync --prerelease=allow --no-install-project --upgrade-package=nerfstudio-gnt --upgrade-package=nerfstudio-pixelnerf

# 7. Tiny-CUDA-NN (Heavy Compilation)
ENV TCNN_CUDA_ARCHITECTURES=$GPU_ARCH
ENV MAX_JOBS=2

RUN uv pip install pip "setuptools<70.0.0" wheel ninja
RUN uv pip install -v git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation

# 8. Source Code (Place at the very end so code changes don't trigger recompilations)
COPY . .
RUN uv pip install -e .
