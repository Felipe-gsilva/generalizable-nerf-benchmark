import os
import zipfile
from pathlib import Path


def create_zip(output_filename: str = "benchmark_code.zip"):
    """
    Cria um pacote ZIP compacto contendo todo o código-fonte, scripts e
    arquivos de configuração necessários para rodar o benchmark em servidores remotos,
    garantindo que todas as configs (inclusive do Nerfbusters/Splatfacto) sejam empacotadas
    e excluindo pastas pesadas de dados, logs e pesos baixados dinamicamente.
    """
    print(f"📦 Criando pacote ZIP: {output_filename}...")

    # Diretórios inteiros a serem ignorados
    EXCLUDED_DIRS = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "scratch",
        ".system_generated",
        "docs",  # Documentação latex não é necessária para execução no servidor
    }

    # Subpastas de assets que contém gigabytes de dados/logs e devem ser ignoradas
    EXCLUDED_ASSET_SUBDIRS = {
        os.path.join("assets", "data"),
        os.path.join("assets", "logs"),
        os.path.join("assets", "nerf_checkpoints"),
        os.path.join("assets", "results"),
        os.path.join("assets", "nerf_aug"),
    }

    # Extensões pesadas de pesos binários (são baixados sob demanda pelo NerfModel)
    EXCLUDED_EXTENSIONS = {
        ".ckpt",
        ".pth",
        ".pt",
        ".bin",
        ".tar",
        ".gz",
        ".DS_Store",
        ".aux",
        ".log",
    }

    # Identificadores de arquivos de configuração
    CONFIG_EXTENSIONS = {".yaml", ".yml", ".json", ".toml", ".sh", ".conf", ".cfg"}
    EXPLICIT_CONFIG_FILES = {
        "Dockerfile",
        "docker-compose.yml",
        "pyproject.toml",
        "install.sh",
        "run_docker.sh",
        "export_delivery.sh",
    }

    included_configs = []
    included_files_count = 0

    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("."):
            rel_root = os.path.relpath(root, ".")
            if rel_root == ".":
                rel_root = ""

            # Filtra diretórios para evitar percorrer árvores pesadas
            dirs[:] = [
                d
                for d in dirs
                if d not in EXCLUDED_DIRS
                and os.path.join(rel_root, d) not in EXCLUDED_ASSET_SUBDIRS
            ]

            for file in files:
                rel_file_path = os.path.join(rel_root, file) if rel_root else file

                # Ignora zips, caches e arquivos binários pesados
                if (
                    file == output_filename
                    or file.endswith(".zip")
                    or any(file.endswith(ext) for ext in EXCLUDED_EXTENSIONS)
                ):
                    continue

                # Registra se é arquivo de configuração para verificação
                is_config = (
                    any(file.endswith(ext) for ext in CONFIG_EXTENSIONS)
                    or file in EXPLICIT_CONFIG_FILES
                    or "config" in file.lower()
                )

                if is_config:
                    included_configs.append(rel_file_path)

                zipf.write(rel_file_path, rel_file_path)
                included_files_count += 1

    zip_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
    print(f"\n✅ Concluído! {output_filename} criado com sucesso:")
    print(f"   • Tamanho total: {zip_size_mb:.2f} MB")
    print(f"   • Total de arquivos: {included_files_count}")

    print("\n📋 Arquivos de configuração incluídos no ZIP:")
    for cfg in sorted(included_configs):
        print(f"   ✓ {cfg}")


if __name__ == "__main__":
    create_zip()
