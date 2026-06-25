import csv
import sys
import re
from pathlib import Path

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.lines as mlines
except ImportError:
    print("Error: The 'numpy' and/or 'matplotlib' libraries are not installed.")
    sys.exit(1)

def main():
    log_dir = Path("assets/logs")
    csv_files = list(log_dir.rglob("metrics.csv"))
    
    if not csv_files:
        print("No metrics.csv files found in assets/logs.")
        return

    data = []

    for file in csv_files:
        parts = file.parts
        if len(parts) < 6:
            continue
            
        model = parts[-5]
        scene = parts[-4]
        run_id = parts[-2]
        
        # Extract num_views
        match = re.search(r'(\d+)views', run_id)
        num_views = int(match.group(1)) if match else 0
        
        try:
            with open(file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    continue
                row = rows[-1]
                
                vram_bytes = row.get("cuda_memory_usage_bytes", "")
                psnr_val = row.get("psnr", "")
                
                if vram_bytes and psnr_val:
                    vram_mb = float(vram_bytes) / (1024 * 1024)
                    psnr = float(psnr_val)
                    data.append({
                        "Model": model, 
                        "Scene": scene,
                        "Views": num_views,
                        "VRAM": vram_mb, 
                        "PSNR": psnr
                    })
        except Exception as e:
            continue

    if not data:
        print("No valid data extracted.")
        return

    # Grouping unique models and views
    models_set = sorted(list(set(d["Model"] for d in data)))
    views_set = sorted(list(set(d["Views"] for d in data)))
    
    colors = list(mcolors.TABLEAU_COLORS.values())
    markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Mapping
    model_colors = {m: colors[i % len(colors)] for i, m in enumerate(models_set)}
    view_markers = {v: markers[i % len(markers)] for i, v in enumerate(views_set)}

    plt.figure(figsize=(12, 8))
    
    # Plot each data point
    for d in data:
        plt.scatter(
            d["VRAM"], 
            d["PSNR"], 
            color=model_colors[d["Model"]], 
            marker=view_markers[d["Views"]], 
            s=90, 
            alpha=0.75, 
            edgecolors='k', 
            linewidth=0.5
        )

    # Plot linear regression for each model
    for model in models_set:
        model_data = [d for d in data if d["Model"] == model]
        vrams = [d["VRAM"] for d in model_data]
        psnrs = [d["PSNR"] for d in model_data]
        
        # Only plot regression if there are at least 2 points
        if len(vrams) > 1:
            try:
                # Calculate linear regression (degree 1)
                z = np.polyfit(vrams, psnrs, 1)
                p = np.poly1d(z)
                
                # Plot the trend line over the range of x values for this model
                # To make it extend just a tiny bit beyond the dots, we can use linspace
                x_seq = np.linspace(min(vrams), max(vrams), 100)
                plt.plot(x_seq, p(x_seq), color=model_colors[model], linestyle='--', alpha=0.5, linewidth=2)
            except Exception:
                pass

    # English labels for paper integration
    plt.title('Computational Cost vs Visual Quality (All Runs with Trend Lines)', fontsize=14, pad=15)
    plt.xlabel('Computational Cost: Peak VRAM (MB)', fontsize=12)
    plt.ylabel('Visual Quality: PSNR (dB)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Create custom legends
    # 1. Models (Colors)
    model_handles = [
        mlines.Line2D([], [], color=model_colors[m], marker='o', linestyle='None',
                      markersize=9, label=m) 
        for m in models_set
    ]
                                   
    # 2. Number of Views (Markers)
    view_handles = [
        mlines.Line2D([], [], color='gray', marker=view_markers[v], linestyle='None',
                      markersize=9, label=f"{v} images") 
        for v in views_set
    ]
    
    # Add an entry for the Trend Line in the views legend (or separately)
    trend_handle = mlines.Line2D([], [], color='gray', linestyle='--', linewidth=2, label='Trend Line')
    view_handles.append(trend_handle)
                                  
    # Add legends to the plot
    first_legend = plt.legend(handles=model_handles, title='Models', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.gca().add_artist(first_legend)
    plt.legend(handles=view_handles, title='Input Images & Trends', bbox_to_anchor=(1.02, 0.5), loc='upper left', borderaxespad=0.)

    plt.tight_layout()

    # Save the plot
    output_dir = Path("assets/results/aggregated_metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "vram_vs_psnr_all_runs_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ Plot successfully updated and saved at: {plot_path}")

if __name__ == "__main__":
    main()
