import os
import csv
from pathlib import Path

def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)

def format_row(row, widths):
    return " | ".join(f"{str(item).ljust(w)}" for item, w in zip(row, widths))

def print_table(headers, data):
    # Calculate column widths
    widths = [len(str(h)) for h in headers]
    for row in data:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))
            
    separator = "-+-".join("-" * w for w in widths)
    
    print(format_row(headers, widths))
    print(separator)
    for row in data:
        # Format floats to 4 decimal places where applicable
        formatted_row = []
        for val in row:
            if isinstance(val, float):
                formatted_row.append(f"{val:.4f}")
            else:
                formatted_row.append(val)
        print(format_row(formatted_row, widths))

def main():
    log_dir = Path("assets/logs")
    if not log_dir.exists():
        print(f"O diretório {log_dir} não existe.")
        return

    csv_files = list(log_dir.rglob("metrics.csv"))
    
    data = []
    
    for file in csv_files:
        parts = file.parts
        if len(parts) < 6:
            continue
            
        model = parts[-5]
        scene = parts[-4]
        
        try:
            with open(file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    continue
                # Pega a última linha para as métricas finais
                row = rows[-1]
                row["Model"] = model
                row["Scene"] = scene
                data.append(row)
        except Exception as e:
            print(f"Erro ao ler {file}: {e}")
            
    if not data:
        print("Nenhum dado válido de metrics.csv foi encontrado em assets/logs.")
        return
        
    metrics = {
        "psnr": "PSNR",
        "ssim": "SSIM",
        "lpips": "LPIPS",
        "convergence_time_seconds": "Train Time (s)",
        "inference_fps": "Inference FPS",
        "cuda_memory_usage_bytes": "VRAM Peak (MB)"
    }
    
    # Pre-process numeric data
    for row in data:
        for key in metrics.keys():
            if key in row and row[key] != '':
                row[key] = float(row[key])
                if key == "cuda_memory_usage_bytes":
                    row[key] = row[key] / (1024 * 1024)
            else:
                row[key] = None

    headers = list(metrics.values())
    
    # Super Table (Geral)
    super_values = {k: [] for k in metrics.keys()}
    for row in data:
        for k in metrics.keys():
            if row[k] is not None:
                super_values[k].append(row[k])
                
    super_means = [mean(super_values[k]) for k in metrics.keys()]
    
    print("\n" + "="*80)
    print("🌟 SUPER TABELA (Média Geral) 🌟")
    print("="*80)
    print_table(headers, [super_means])
    
    # Por Cena
    scenes = set(row["Scene"] for row in data)
    scene_data = []
    for scene in sorted(scenes):
        scene_rows = [r for r in data if r["Scene"] == scene]
        means = []
        for k in metrics.keys():
            vals = [r[k] for r in scene_rows if r[k] is not None]
            means.append(mean(vals))
        scene_data.append([scene] + means)
        
    print("\n" + "="*80)
    print("🎬 DADOS AGLOMERADOS POR CENA (Média de todos os modelos) 🎬")
    print("="*80)
    print_table(["Scene"] + headers, scene_data)
    
    # Por Modelo
    models = set(row["Model"] for row in data)
    model_data = []
    for model in sorted(models):
        model_rows = [r for r in data if r["Model"] == model]
        means = []
        for k in metrics.keys():
            vals = [r[k] for r in model_rows if r[k] is not None]
            means.append(mean(vals))
        model_data.append([model] + means)
        
    print("\n" + "="*80)
    print("🤖 DADOS AGLOMERADOS POR MODELO (Média de todas as cenas) 🤖")
    print("="*80)
    print_table(["Model"] + headers, model_data)

    # Export to CSV manually
    output_dir = Path("assets/results/aggregated_metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "super_table.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(super_means)
        
    with open(output_dir / "scene_table.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Scene"] + headers)
        writer.writerows(scene_data)
        
    with open(output_dir / "model_table.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model"] + headers)
        writer.writerows(model_data)

    print(f"\n📁 Tabelas CSV exportadas para: {output_dir}/")

if __name__ == "__main__":
    main()
