FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV IS_DOCKER=1

# 🔧 Sistema
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git build-essential cmake ninja-build \
    libgl1 libglib2.0-0 \
    libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# 🔑 Toolchain Python
RUN python -m pip install --upgrade pip setuptools wheel

# uv
RUN pip install uv

WORKDIR /workspace

# -----------------------------------
# 1. 📦 DEPENDÊNCIAS (CACHE LAYER)
# -----------------------------------

# Copia só pyproject.toml para travar a base do ambiente
COPY pyproject.toml ./

# 🔥 Ignora instalação do projeto (CRÍTICO para cache)
RUN uv sync --prerelease=allow --no-install-project --upgrade-package=nerfstudio-gnt --upgrade-package=nerfstudio-pixelnerf

# -----------------------------------
# 2. 🔨 TINY-CUDA-NN (HEAVY COMPILATION)
# -----------------------------------
# Instalado ANTES do código fonte. O Docker fará cache dessa etapa permanentemente.

RUN uv pip install pip "setuptools<70.0.0" wheel ninja
ENV MAX_JOBS=2

# 1. Accept the architecture passed from the bash script (default to 86 if missing)
ARG GPU_ARCH="86"

# 2. Assign it to the environment variable tiny-cuda-nn looks for
ENV TCNN_CUDA_ARCHITECTURES=$GPU_ARCH

# 3. Build!
RUN .venv/bin/python -m pip install \
  -v git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch \
  --no-build-isolation

# -----------------------------------
# 3. 📁 CÓDIGO FONTE (MUDANÇAS FREQUENTES)
# -----------------------------------
# Qualquer edição no seu código só invalida o cache daqui para baixo!

COPY . .

# Agora instala seu projeto corretamente
RUN uv pip install -e .

# -----------------------------------
# 🚀 ENTRYPOINT
# -----------------------------------

# Injeta o ambiente virtual diretamente no PATH do sistema.
# Isso elimina a necessidade de usar o comando "uv run" para rodar scripts!
ENV PATH="/workspace/.venv/bin:$PATH"

# Comando padrão caso você rode o container sem argumentos
CMD ["python", "main.py"]
