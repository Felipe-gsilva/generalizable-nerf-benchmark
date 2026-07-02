import csv
from pathlib import Path
import argparse

def mean(lst):
    if not lst: return 0.0
    return sum(lst)/len(lst)

def print_latex_table(data_summary):
    print("\\begin{table}[!t]")
    print("\\centering")
    print("\\caption{Training and optimization with recommended range values of learning rate}")
    print("\\label{tab:ablation_lr}")
    print("\\resizebox{\\columnwidth}{!}{%}")
    print("\\begin{tabular}{llcc}")
    print("\\toprule")
    print("\\textbf{Model} & \\textbf{Regime} & \\textbf{Learning Rate ($\\eta$)} & \\textbf{PSNR (dB) $\\uparrow$}  \\\\")
    print("\\midrule")
    
    # PixelNeRF
    print("\\multirow{4}{*}{PixelNeRF} ")
    p_1e4 = data_summary.get("pixel-nerf_tta_1e-4", "N/A")
    p_5e4 = data_summary.get("pixel-nerf_tta_5e-4", "N/A")
    p_1e3 = data_summary.get("pixel-nerf_tta_1e-3", "N/A")
    p_zs = data_summary.get("pixel-nerf_zero-shot", "N/A")
    print(f"               & TTA         & $10^{{-4}}$ & {p_1e4} \\\\")
    print(f"               & TTA         & $5 \\times 10^{{-4}}$ & {p_5e4} \\\\")
    print(f"               & TTA         & $10^{{-3}}$ & {p_1e3} \\\\")
    print(f"               & Zero-Shot   & N/A         & {p_zs} \\\\")
    print("\\midrule")
    
    # GNT
    print("\\multirow{4}{*}{GNT}")
    g_1e4 = data_summary.get("gnt_tta_1e-4", "N/A")
    g_5e4 = data_summary.get("gnt_tta_5e-4", "N/A")
    g_1e3 = data_summary.get("gnt_tta_1e-3", "N/A")
    g_zs = data_summary.get("gnt_zero-shot", "N/A")
    print(f"               & TTA         & $10^{{-4}}$ & {g_1e4}\\\\")
    print(f"               & TTA         & $5 \\times 10^{{-4}}$ & {g_5e4}\\\\")
    print(f"               & TTA         & $10^{{-3}}$ & {g_1e3}\\\\")
    print(f"               & Zero-Shot   & N/A         & {g_zs} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}%")
    print("}")
    print("\\end{table}")


def main():
    parser = argparse.ArgumentParser(description="Generate LR Ablation Table")
    parser.add_argument("--log-dir", type=str, default="assets/logs", help="Directory containing the ablation logs (metrics.csv files)")
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: Directory {log_dir} does not exist.")
        return
        
    metrics_files = list(log_dir.rglob("metrics.csv"))
    
    data = []
    
    target_keys = [
        "pixel-nerf_tta_1e-4", "pixel-nerf_tta_5e-4", "pixel-nerf_tta_1e-3", "pixel-nerf_zero-shot",
        "gnt_tta_1e-4", "gnt_tta_5e-4", "gnt_tta_1e-3", "gnt_zero-shot"
    ]
    
    for f in metrics_files:
        parts = f.parts
        
        model_key = None
        for mk in target_keys:
            if mk in parts:
                model_key = mk
                break
                
        if not model_key:
            continue
            
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                rows = list(csv.DictReader(fp))
                if not rows: continue
                row = rows[-1]
                psnr = float(row.get("psnr", 0))
                data.append({"model_key": model_key, "psnr": psnr})
        except Exception:
            pass

    data_summary = {}
    for mk in target_keys:
        m_data = [d["psnr"] for d in data if d["model_key"] == mk]
        if m_data:
            data_summary[mk] = f"{mean(m_data):.2f}"
        else:
            data_summary[mk] = "N/A"
            
    print("--- GERANDO TABELA LATEX ---")
    print_latex_table(data_summary)
    print("----------------------------")
    
if __name__ == "__main__":
    main()
