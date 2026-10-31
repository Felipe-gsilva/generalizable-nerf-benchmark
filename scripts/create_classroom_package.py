#!/usr/bin/env python3
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "entrega_aprendizado_profundo"
ZIP_FILE = ROOT_DIR / "entrega_aprendizado_profundo.zip"


def run_cmd(cmd, cwd=None):
    res = subprocess.run(cmd, shell=True, cwd=cwd or ROOT_DIR, text=True, capture_output=True)
    if res.returncode != 0:
        print(f"Erro na execução: {res.stderr}")
    return res


def compile_paper():
    latex_dir = ROOT_DIR / "docs" / "final-2027"
    run_cmd("latexmk -pdf -outdir=build main.tex && cp build/main.pdf main.pdf", cwd=latex_dir)


def prepare_directory_structure():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    dirs = [
        OUTPUT_DIR / "artigo" / "formato_editavel" / "img",
        OUTPUT_DIR / "apresentacao",
        OUTPUT_DIR / "codigo",
        OUTPUT_DIR / "dados_teste" / "fern",
        OUTPUT_DIR / "resultados" / "metricas",
        OUTPUT_DIR / "resultados" / "graficos",
        OUTPUT_DIR / "resultados" / "tabelas",
        OUTPUT_DIR / "resultados" / "renderizacoes",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def copy_paper_files():
    latex_dir = ROOT_DIR / "docs" / "final-2027"

    # PDF compilado
    shutil.copy2(latex_dir / "main.pdf", OUTPUT_DIR / "artigo" / "artigo.pdf")

    # Formato editável
    edit_dir = OUTPUT_DIR / "artigo" / "formato_editavel"
    extensions = {".tex", ".bib", ".cls", ".sty", ".bst"}
    for f in latex_dir.iterdir():
        if f.is_file() and f.suffix in extensions:
            shutil.copy2(f, edit_dir / f.name)

    # Imagens do artigo
    img_src = latex_dir / "img"
    if img_src.exists():
        shutil.copytree(img_src, edit_dir / "img", dirs_exist_ok=True)

    # Instruções de compilação mínimas
    (edit_dir / "README.md").write_text(
        "# Compilação do Artigo\n\n"
        "Para compilar o artigo via latexmk:\n"
        "```bash\n"
        "latexmk -pdf main.tex\n"
        "```\n\n"
        "Ou usando pdflatex:\n"
        "```bash\n"
        "pdflatex main.tex\n"
        "bibtex main\n"
        "pdflatex main.tex\n"
        "pdflatex main.tex\n"
        "```\n"
    )


def copy_presentation_file():
    pres_file = ROOT_DIR / "Nerf sob Vistas Esparsas - Apresentação.pdf"
    if pres_file.exists():
        shutil.copy2(pres_file, OUTPUT_DIR / "apresentacao" / "apresentacao.pdf")


def copy_source_code():
    dst = OUTPUT_DIR / "codigo"

    def ignore_patterns(path, names):
        return [n for n in names if n in {"__pycache__", ".pytest_cache", ".ruff_cache", ".egg-info"} or n.endswith(".pyc")]

    shutil.copytree(ROOT_DIR / "src", dst / "src", ignore=ignore_patterns, dirs_exist_ok=True)
    shutil.copytree(ROOT_DIR / "scripts", dst / "scripts", ignore=ignore_patterns, dirs_exist_ok=True)

    files_to_copy = [
        "main.py",
        "pyproject.toml",
        "Dockerfile",
        "docker-compose.yml",
        "install.sh",
        "run_docker.sh",
        "README.md",
    ]
    for f in files_to_copy:
        p = ROOT_DIR / f
        if p.exists():
            shutil.copy2(p, dst / f)


def copy_test_dataset():
    src_data = ROOT_DIR / "assets" / "data" / "baseline" / "nerf_llff_data" / "fern"
    dst_data = OUTPUT_DIR / "dados_teste" / "fern"
    if src_data.exists():
        shutil.copytree(src_data, dst_data, dirs_exist_ok=True)

    (OUTPUT_DIR / "dados_teste" / "README.md").write_text(
        "# Dados de Teste\n\n"
        "O benchmark avaliou todas as 8 cenas do dataset LLFF (*fern*, *flower*, *fortress*, *horns*, *leaves*, *orchids*, *room*, *trex*).\n\n"
        "Para manter o pacote leve, incluímos a cena `fern` como amostra pronta para teste imediato com o split `llffhold=8`.\n"
        "Para baixar e configurar todas as 8 cenas do LLFF para testes completos, basta rodar o `./install.sh` na pasta de código.\n"
    )


def copy_results():
    res_dir = OUTPUT_DIR / "resultados"

    # CSVs de métricas
    agg_src = ROOT_DIR / "assets" / "results" / "aggregated_metrics"
    if agg_src.exists():
        shutil.copytree(agg_src, res_dir / "metricas", dirs_exist_ok=True)

    # Gráficos
    plots_src = ROOT_DIR / "assets" / "results" / "plots"
    if plots_src.exists():
        shutil.copytree(plots_src, res_dir / "graficos", dirs_exist_ok=True)

    ablation_src = ROOT_DIR / "assets" / "reports" / "tta_steps_ablation"
    if ablation_src.exists():
        for p in ablation_src.glob("*.png"):
            shutil.copy2(p, res_dir / "graficos" / p.name)
        for c in ablation_src.glob("*.csv"):
            shutil.copy2(c, res_dir / "metricas" / c.name)

    # Tabelas LaTeX
    tables_src = ROOT_DIR / "assets" / "reports" / "tables"
    if tables_src.exists():
        shutil.copytree(tables_src, res_dir / "tabelas", dirs_exist_ok=True)

    # Amostras de renderização qualitativa
    grid_src = ROOT_DIR / "docs" / "final-2027" / "img" / "grid5x5"
    if grid_src.exists():
        shutil.copytree(grid_src, res_dir / "renderizacoes" / "grid_comparativo", dirs_exist_ok=True)

    sample_models = [
        "splatfacto_per-scene",
        "gnt_pft-decoder",
        "gnt_peft-lora",
        "gnt_bitfit",
        "pixel-nerf_pft-backbone",
    ]
    for model_name in sample_models:
        model_fern = ROOT_DIR / "assets" / "results" / model_name / "fern" / "uniform_3views"
        if model_fern.exists():
            dst_sub = res_dir / "renderizacoes" / model_name
            dst_sub.mkdir(parents=True, exist_ok=True)
            for png in model_fern.glob("*.png"):
                shutil.copy2(png, dst_sub / png.name)

    (res_dir / "README.md").write_text(
        "# Resultados\n\n"
        "- `metricas/`: Arquivos CSV com métricas de PSNR, SSIM, LPIPS, tempo, FPS e VRAM para todas as cenas.\n"
        "- `graficos/`: Figuras de convergência, ablação de passos e trade-offs.\n"
        "- `tabelas/`: Tabelas em formato LaTeX utilizadas no artigo.\n"
        "- `renderizacoes/`: Novas vistas renderizadas pelos modelos avaliados.\n"
    )


def create_root_readme():
    readme_path = OUTPUT_DIR / "README.md"
    readme_path.write_text(
        "# Trabalho Prático - Aprendizado Profundo\n\n"
        "**Aluno:** Felipe Gomes da Silva  \n"
        "**Artigo:** *NeRF Architectures under Sparse-View Conditions: A Systematic Benchmarking Methodology* (7 páginas, formato IEEE)  \n\n"
        "## Estrutura dos Arquivos\n\n"
        "- `artigo/`: Artigo em PDF (`artigo.pdf`) e fontes em LaTeX (`formato_editavel/`).\n"
        "- `apresentacao/`: Slides da apresentação do trabalho (`apresentacao.pdf`).\n"
        "- `codigo/`: Código-fonte completo com os modelos (PixelNeRF, GNT, Instant-NGP, Nerfacto, Splatfacto), wrappers e scripts de benchmark.\n"
        "- `dados_teste/`: Amostra de teste do dataset LLFF (`fern`) com imagens e poses COLMAP (`llffhold=8`). O benchmark completo utilizou todas as 8 cenas do LLFF (detalhes abaixo).\n"
        "- `resultados/`: Métricas consolidadas (CSVs), gráficos de ablação/convergência, tabelas LaTeX e imagens renderizadas.\n\n"
        "## Como Executar e Baixar Datasets\n\n"
        "Para instalar dependências e baixar automaticamente o dataset LLFF completo com as 8 cenas para teste:\n"
        "```bash\n"
        "./install.sh\n"
        "```\n\n"
        "Alternativamente, utilizando Docker:\n"
        "```bash\n"
        "docker compose up --build\n"
        "```\n\n"
        "Para executar treinos e avaliações:\n"
        "```bash\n"
        "python main.py --help\n"
        "```\n"
    )


def create_zip_archive():
    if ZIP_FILE.exists():
        ZIP_FILE.unlink()

    with zipfile.ZipFile(ZIP_FILE, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(OUTPUT_DIR):
            for file in files:
                full_path = Path(root) / file
                arcname = full_path.relative_to(ROOT_DIR)
                zipf.write(full_path, arcname)

    size_mb = ZIP_FILE.stat().st_size / (1024 * 1024)
    print(f"Pacote {ZIP_FILE.name} gerado com sucesso ({size_mb:.2f} MB).")


def main():
    print("Compilando artigo e preparando entrega...")
    compile_paper()
    prepare_directory_structure()
    copy_paper_files()
    copy_presentation_file()
    copy_source_code()
    copy_test_dataset()
    copy_results()
    create_root_readme()
    create_zip_archive()
    print("Concluído.")


if __name__ == "__main__":
    main()
