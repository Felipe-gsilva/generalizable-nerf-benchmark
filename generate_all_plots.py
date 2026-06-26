import csv
import sys
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def mean(lst):
    if not lst: return 0.0
    return sum(lst)/len(lst)

def load_data():
    log_dir = Path("assets/logs")
    metrics_files = list(log_dir.rglob("metrics.csv"))
    
    data = []
    for f in metrics_files:
        parts = f.parts
        if len(parts) < 6: continue
        model = parts[-5]
        scene = parts[-4]
        strategy_views = parts[-2]
        
        views = 3
        if "6views" in strategy_views: views = 6
        elif "10views" in strategy_views: views = 10
        
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
                
                regime = "Per-Scene"
                base_model = model
                if "zero-shot" in model:
                    regime = "Zero-Shot"
                    base_model = model.replace("_zero-shot", "")
                elif "tta" in model:
                    regime = "TTA"
                    base_model = model.replace("_tta", "")
                elif "per-scene" in model:
                    base_model = model.replace("_per-scene", "")
                    
                data.append({
                    "model_full": model,
                    "base_model": base_model,
                    "regime": regime,
                    "scene": scene, 
                    "views": views, 
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

def plot_1_scaling_law(data, output_dir):
    plt.figure(figsize=(10, 6))
    
    for model in ["gnt_zero-shot", "gnt_tta", "pixel-nerf_zero-shot", "pixel-nerf_tta"]:
        views_dict = {3:[], 6:[], 10:[]}
        for d in data:
            if d["model_full"] == model and d["views"] in views_dict:
                views_dict[d["views"]].append(d["psnr"])
        
        x = [3, 6, 10]
        y = [mean(views_dict[v]) for v in x]
        
        if any(y):
            marker = 'o' if 'zero-shot' in model else '^'
            linestyle = '--' if 'zero-shot' in model else '-'
            plt.plot(x, y, marker=marker, linestyle=linestyle, linewidth=2, markersize=8, label=model)
            
    plt.title('Scaling Law: Visual Quality vs Number of Views', fontsize=14)
    plt.xlabel('Number of Input Views (N)', fontsize=12)
    plt.ylabel('Average PSNR (dB)', fontsize=12)
    plt.xticks([3, 6, 10])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot1_scaling_law.png", dpi=300)
    plt.close()

def plot_2_bubble_chart(data, output_dir):
    plt.figure(figsize=(11, 7))
    
    models = sorted(list(set(d["model_full"] for d in data)))
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    for i, m in enumerate(models):
        m_data = [d for d in data if d["model_full"] == m]
        if not m_data: continue
        avg_fps = mean([d["fps"] for d in m_data])
        avg_psnr = mean([d["psnr"] for d in m_data])
        avg_time = mean([d["train_time"] for d in m_data])
        
        # Scale area of bubble based on train time
        s = np.clip(avg_time, 50, 3000) 
        
        plt.scatter(avg_fps, avg_psnr, s=s, color=colors[i%len(colors)], alpha=0.7, edgecolors='k', label=m)
        plt.annotate(m, (avg_fps, avg_psnr), xytext=(12, 12), textcoords='offset points')
        
    plt.xscale('log')
    plt.title('Viability Trade-off: Inference FPS vs PSNR\n(Bubble Size = Training Time)', fontsize=14)
    plt.xlabel('Inference FPS (Log Scale)', fontsize=12)
    plt.ylabel('Average PSNR (dB)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / "plot2_bubble_viability.png", dpi=300)
    plt.close()

def plot_3_grouped_bar(data, output_dir):
    base_models = ["gnt", "pixel-nerf"]
    zero_shot_psnr = []
    tta_psnr = []
    
    for m in base_models:
        zs_data = [d["psnr"] for d in data if d["base_model"] == m and d["regime"] == "Zero-Shot"]
        tta_data = [d["psnr"] for d in data if d["base_model"] == m and d["regime"] == "TTA"]
        
        zero_shot_psnr.append(mean(zs_data) if zs_data else 0)
        tta_psnr.append(mean(tta_data) if tta_data else 0)
        
    x = np.arange(len(base_models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, zero_shot_psnr, width, label='Zero-Shot', color='#1f77b4', edgecolor='k')
    rects2 = ax.bar(x + width/2, tta_psnr, width, label='Test-Time Adaptation (TTA)', color='#ff7f0e', edgecolor='k')
    
    ax.set_ylabel('Average PSNR (dB)')
    ax.set_title('Visual Quality: Zero-Shot vs TTA (Architectural Inertia)')
    ax.set_xticks(x)
    ax.set_xticklabels(["GNT", "PixelNeRF"])
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom')
                        
    plt.tight_layout()
    plt.savefig(output_dir / "plot3_grouped_bar.png", dpi=300)
    plt.close('all')

def plot_4_radar(data, output_dir):
    models = ["instant-ngp_per-scene", "nerfacto_per-scene", "gnt_tta", "pixel-nerf_tta"]
    metrics = ["PSNR", "SSIM", "1 / LPIPS", "1 / Train Time", "FPS"]
    
    raw_avgs = {}
    for m in models:
        m_data = [d for d in data if d["model_full"] == m]
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
            normalized[m].append(max(norm, 0.05)) # floor at 0.05 for visibility
            
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
        
    plt.xticks(angles[:-1], metrics)
    ax.set_yticklabels([])
    plt.title('Radar Chart: Overall Architecture Balance\n(Normalized Values)', size=14, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig(output_dir / "plot4_radar.png", dpi=300)
    plt.close('all')

def plot_5_boxplot(data, output_dir):
    plt.figure(figsize=(10, 6))
    
    models = sorted(list(set(d["model_full"] for d in data)))
    plot_data = []
    labels = []
    
    for m in models:
        m_psnr = [d["psnr"] for d in data if d["model_full"] == m]
        if m_psnr:
            plot_data.append(m_psnr)
            labels.append(m.replace("_per-scene", "").replace("_zero-shot", "\n(ZS)").replace("_tta", "\n(TTA)"))
            
    plt.boxplot(plot_data, tick_labels=labels, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black'),
                medianprops=dict(color='red', linewidth=2))
                
    plt.title('Performance Variance Across Evaluated Scenes', fontsize=14)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "plot5_boxplot.png", dpi=300)
    plt.close()

def main():
    data = load_data()
    if not data:
        print("No data found.")
        return
        
    output_dir = Path("assets/results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_1_scaling_law(data, output_dir)
    plot_2_bubble_chart(data, output_dir)
    plot_3_grouped_bar(data, output_dir)
    plot_4_radar(data, output_dir)
    plot_5_boxplot(data, output_dir)
    
    print(f"✅ Generated 5 plots in {output_dir}/")

if __name__ == "__main__":
    main()
