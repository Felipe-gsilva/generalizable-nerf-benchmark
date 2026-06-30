import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def mean(lst):
    if not lst: return 0.0
    return sum(lst)/len(lst)

def format_model_name(name):
    mapping = {
        "gnt_zero-shot": "GNT (ZS)",
        "gnt_tta": "GNT (TTA)",
        "pixel-nerf_zero-shot": "PixelNeRF (ZS)",
        "pixel-nerf_tta": "PixelNeRF (TTA)",
        "instant-ngp_per-scene": "Instant-NGP (Per-Scene)",
        "nerfacto_per-scene": "Nerfacto (Per-Scene)"
    }
    return mapping.get(name, name)

def load_data():
    log_dir = Path("assets/logs")
    metrics_files = list(log_dir.rglob("metrics.csv"))
    
    data = []
    for f in metrics_files:
        parts = f.parts
        if len(parts) < 6: continue
        model = parts[-5]
        scene = parts[-4]
        
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                rows = list(csv.DictReader(fp))
                if not rows: continue
                row = rows[-1]
                psnr = float(row.get("psnr", 0))
                ssim = float(row.get("ssim", 0))
                lpips = float(row.get("lpips", 0))
                train_time = float(row.get("convergence_time_seconds", 0))
                fps = float(row.get("inference_fps", 0))
                vram = float(row.get("cuda_memory_usage_bytes", 0))/(1024*1024)
                
                data.append({
                    "model_raw": model,
                    "model_formatted": format_model_name(model),
                    "scene": scene, 
                    "psnr": psnr, 
                    "ssim": ssim,
                    "lpips": lpips, 
                    "train_time": train_time,
                    "fps": fps, 
                    "vram": vram
                })
        except:
            pass
    return data


def plot_radar(data, output_dir):
    # Selecionar os modelos (usando os nomes padronizados)
    models_raw = ["instant-ngp_per-scene", "nerfacto_per-scene", "gnt_tta", "pixel-nerf_tta"]
    models = [format_model_name(m) for m in models_raw]
    
    metrics = ["PSNR", "SSIM", "1 / LPIPS", "1 / Train Time", "FPS"]
    
    raw_avgs = {}
    for m in models:
        m_data = [d for d in data if d["model_formatted"] == m]
        if not m_data: continue
        raw_avgs[m] = {
            "PSNR": mean([d["psnr"] for d in m_data]),
            "SSIM": mean([d["ssim"] for d in m_data]),
            "1 / LPIPS": 1.0 / (mean([d["lpips"] for d in m_data]) + 1e-5),
            "1 / Train Time": 1.0 / (mean([d["train_time"] for d in m_data]) + 1e-5),
            "FPS": mean([d["fps"] for d in m_data])
        }
    
    if not raw_avgs: return
    
    normalized = {}
    for m in raw_avgs:
        normalized[m] = []
        for key in metrics:
            all_vals = [raw_avgs[mx][key] for mx in raw_avgs]
            min_v = min(all_vals)
            max_v = max(all_vals)
            norm = (raw_avgs[m][key] - min_v) / (max_v - min_v + 1e-5)
            normalized[m].append(max(norm, 0.05)) # floor em 0.05 para não sumir no centro
            
    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    for i, m in enumerate(raw_avgs):
        values = normalized[m]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=m, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
        
    plt.xticks(angles[:-1], metrics, size=11)
    ax.set_yticklabels([])
    plt.title('Radar Chart: Overall Architecture Balance\n(Normalized Values)', size=15, y=1.1)
    
    # Legenda descritiva e padronizada
    plt.legend(title='Evaluated Models', loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig(output_dir / "plot_radar.png", dpi=300)
    plt.close('all')


def plot_boxplot(data, output_dir):
    plt.figure(figsize=(10, 6))
    
    # Listar os modelos únicos formatados em ordem alfabética para consistência
    models = sorted(list(set(d["model_formatted"] for d in data)))
    plot_data = []
    labels = []
    
    for m in models:
        m_psnr = [d["psnr"] for d in data if d["model_formatted"] == m]
        if m_psnr:
            plot_data.append(m_psnr)
            # Quebrar a string no meio caso tenha (ZS) ou (TTA) para encaixar melhor no eixo X
            if " (" in m:
                label_parts = m.split(" (")
                labels.append(f"{label_parts[0]}\n({label_parts[1]}")
            else:
                labels.append(m)
            
    plt.boxplot(plot_data, tick_labels=labels, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black'),
                medianprops=dict(color='red', linewidth=2))
                
    plt.title('Performance Variance Across Evaluated Scenes', fontsize=14)
    plt.xlabel('Architectures and Regimes', fontsize=12)
    plt.ylabel('Visual Quality: PSNR (dB)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_boxplot.png", dpi=300)
    plt.close()

def main():
    data = load_data()
    if not data:
        print("No data found.")
        return
        
    output_dir = Path("assets/results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_radar(data, output_dir)
    plot_boxplot(data, output_dir)
    
    print(f"✅ Generated Radar and Box plots in {output_dir}/")

if __name__ == "__main__":
    main()
