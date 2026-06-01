# NeRF ANN Paper - Experiment Pipeline

Este repositório contém a infraestrutura e os modelos para rodar experimentos de Neural Radiance Fields (NeRFs), com foco na adaptação e métricas de qualidade.

## Estrutura do Projeto

- `src/`: Código-fonte principal com a definição dos modelos e rotinas de métricas.
  - `src/nerf/`: Implementação e configurações dos modelos NeRF (PixelNeRF, GNT, etc.).
  - `src/utils/`: Funções utilitárias e loggers (ex: cálculo de PSNR, SSIM, LPIPS).
  - `src/validation/`: Scripts de validação e avaliação.
- `main.py`: Ponto de entrada do experimento, instanciando datasets e iterando pelas estratégias de treino.
- `docker-compose.yml` e `Dockerfile`: Configurações de contêiner contendo toda a base necessária de dependências (Nerfstudio, COLMAP, Pytorch, etc).
- `pyproject.toml` e `uv.lock`: Definição e trava de dependências de Python modernas gerenciadas via `uv`.

## Como Executar

A infraestrutura foi preparada para execução em Docker, abstraindo a instalação de dependências como CUDA, COLMAP e Tiny-CUDA-NN.

```bash
# Para iniciar o container e executar os experimentos definidos em main.py:
docker-compose up --build
```

Os resultados de métricas e os logs das runs serão gerados nas pastas `assets/` e `results/`.
