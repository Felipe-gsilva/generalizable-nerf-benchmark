FROM colmap/colmap:latest AS colmap-bin

# 1. Subimos a base para o Ubuntu 24.04 (mesma versão do colmap:latest)
FROM nvidia/cuda:12.6.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV IS_DOCKER=1

# 2. Sistema: instalamos o colmap (para as dependências) e o intel-mkl
# Removemos o python do apt, pois o 'uv' vai cuidar disso!
RUN apt-get update && apt-get install -y \
    colmap ffmpeg xvfb \
    intel-mkl \
    git build-essential cmake ninja-build \
    libgl1 libglib2.0-0 \
    libsm6 libxext6 libxrender1 \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 3. Instalando o uv da forma mais elegante possível via Docker
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

# 4. Usamos o uv para instalar o Python 3.11 explicitamente e criar o venv
RUN uv python install 3.11
RUN uv venv --python 3.11 .venv

# 5. Dependências do Projeto (Cache Layer)
COPY pyproject.toml ./
RUN uv sync --prerelease=allow --no-install-project --upgrade-package=nerfstudio-gnt --upgrade-package=nerfstudio-pixelnerf

# 6. Tiny-CUDA-NN (Heavy Compilation)
ARG GPU_ARCH="86"
ENV TCNN_CUDA_ARCHITECTURES=$GPU_ARCH
ENV MAX_JOBS=2

# Garante que as ferramentas de build estão no venv
RUN uv pip install pip "setuptools<70.0.0" wheel ninja

# Compila o TCNN
RUN uv pip install -v git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation

# 7. Código Fonte
COPY . .
RUN uv pip install -e .

# 8. Entrypoint e Injeção do COLMAP
ENV PATH="/workspace/.venv/bin:$PATH"

# Agora sim! Copiamos o executável da GPU.
# Como ambos (colmap:latest e este container) são Ubuntu 24.04, o GLIBC bate perfeitamente.
COPY --from=colmap-bin /usr/local/ /usr/local/
RUN ldconfig

CMD ["python", "main.py"]
