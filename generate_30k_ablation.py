import csv
import glob
from pathlib import Path
from collections import defaultdict

def main():
    log_dir = Path("assets/logs")
    models = ["pixel-nerf", "gnt"]
    regimes = ["per-scene", "tta"]
    views = [3, 6, 10]
    
    # Nested dictionary: results[model][regime][view] = max_psnr
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: "N/A")))

    for model in models:
        for regime in regimes:
            model_key = f"{model}_{regime}_30k"
            
            for v in views:
                run_id = f"uniform_{v}views"
                search_pattern = log_dir / model_key / "fern" / "*" / run_id / "metrics.csv"
                
                metrics_files = glob.glob(str(search_pattern))
                max_psnr = -1.0
                
                for f in metrics_files:
                    try:
                        with open(f, 'r') as fp:
                            rows = list(csv.DictReader(fp))
                            if not rows:
                                continue
                            psnr = float(rows[-1].get("psnr", 0))
                            if psnr > max_psnr:
                                max_psnr = psnr
                    except Exception:
                        pass
                
                if max_psnr > -1.0:
                    results[model][regime][v] = f"{max_psnr:.2f}"

    latex_code = r"""\begin{table}[!t]
\centering
\caption{Impact of 30,000 steps adaptation on few-shot synthesis for the \textit{fern} scene}
\label{tab:ablation_30k}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llccc}
\toprule
\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Regime}} & \multicolumn{3}{c}{\textbf{PSNR (dB) $\uparrow$}} \\
\cmidrule(lr){3-5}
& & \textbf{3 Views} & \textbf{6 Views} & \textbf{10 Views} \\
\midrule
"""

    model_names_map = {
        "pixel-nerf": "PixelNeRF",
        "gnt": "GNT"
    }
    
    regime_names_map = {
        "tta": "TTA",
        "per-scene": "Per-Scene"
    }

    for i, model in enumerate(models):
        latex_code += f"\\multirow{{2}}{{*}}{{{model_names_map[model]}}}\n"
        for regime in regimes:
            v3 = results[model][regime][3]
            v6 = results[model][regime][6]
            v10 = results[model][regime][10]
            
            latex_code += f" & {regime_names_map[regime]} & {v3} & {v6} & {v10} \\\\\n"
        
        if i < len(models) - 1:
            latex_code += "\\midrule\n"

    latex_code += r"""\bottomrule
\end{tabular}%
}
\end{table}
"""

    print("=== Tabela LaTeX Gerada ===")
    print(latex_code)

if __name__ == "__main__":
    main()
