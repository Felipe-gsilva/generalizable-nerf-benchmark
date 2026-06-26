import csv
from pathlib import Path

def print_markdown_table(headers, rows):
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "|-" + "-|-".join(["-" * len(h) for h in headers]) + "-|"
    print(header_line)
    print(sep_line)
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print()

def mean(lst):
    if not lst: return 0.0
    return sum(lst)/len(lst)

def main():
    log_dir = Path("assets/logs")
    metrics_files = list(log_dir.rglob("metrics.csv"))
    
    data = []
    for f in metrics_files:
        parts = f.parts
        if len(parts) < 6: continue
        model = parts[-5]
        scene = parts[-4]
        strategy_views = parts[-2]
        
        strategy = "uniform" if "uniform" in strategy_views else "random"
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
                
                data.append({
                    "model": model, "scene": scene, "strategy": strategy,
                    "views": views, "psnr": psnr, "ssim": ssim,
                    "lpips": lpips, "train_time": train_time,
                    "fps": fps, "vram": vram
                })
        except:
            pass

    # 1. Tabela de Qualidade Visual e Desempenho e Memoria (Geral / Overall Averages for Fern)
    fern_all = [d for d in data if d["scene"] == "fern"]
    models = sorted(list(set(d["model"] for d in fern_all)))
    
    vq_rows = []
    mem_rows = []
    for m in models:
        m_data = [d for d in fern_all if d["model"] == m]
        m_psnr = mean([d["psnr"] for d in m_data])
        m_ssim = mean([d["ssim"] for d in m_data])
        m_lpips = mean([d["lpips"] for d in m_data])
        m_train = mean([d["train_time"] for d in m_data])
        m_fps = mean([d["fps"] for d in m_data])
        m_vram = mean([d["vram"] for d in m_data])
        
        regime = "Per-Scene"
        if "zero-shot" in m: regime = "Zero-Shot"
        elif "tta" in m: regime = "TTA"
        
        vq_rows.append([regime, m, f"{m_psnr:.2f}", f"{m_ssim:.3f}", f"{m_lpips:.3f}"])
        mem_rows.append([regime, m, f"{m_train:.2f}", f"{m_fps:.3f}", f"{m_vram:.2f}"])
        
    print("### 👁️ Tabela de Qualidade Visual (Cena: fern, média total)")
    print_markdown_table(["Regime", "Modelo", "PSNR (dB) $\\uparrow$", "SSIM $\\uparrow$", "LPIPS $\\downarrow$"], vq_rows)
    
    print("### 🧠 Tabela de Desempenho e Memória (Cena: fern, média total)")
    print_markdown_table(["Regime", "Modelo", "Tempo Treino (s) $\\downarrow$", "FPS Inferência $\\uparrow$", "Pico VRAM (MB) $\\downarrow$"], mem_rows)
    
    # 2. Impact of Views (N) for TTA and Zero-Shot models (using fern)
    gen_models = [m for m in models if "tta" in m or "zero-shot" in m]
    impact_rows = []
    for m in gen_models:
        row = [m]
        for v in [3, 6, 10]:
            v_data = [d for d in data if d["model"] == m and d["views"] == v and d["scene"] == "fern"]
            if v_data:
                avg_psnr = mean([d["psnr"] for d in v_data])
                row.append(f"{avg_psnr:.2f}")
            else:
                row.append("N/A")
        impact_rows.append(row)
    print("### 📈 Escalonamento por Número de Vistas de Treino ($N$) no PSNR")
    print_markdown_table(["Modelo", "N = 3", "N = 6", "N = 10"], impact_rows)
    
    # 3. Sampling Ablation (Uniform vs Random) in Fern
    ablation_rows = []
    for m in models:
        u_data = [d for d in data if d["model"] == m and d["scene"] == "fern" and d["strategy"] == "uniform"]
        r_data = [d for d in data if d["model"] == m and d["scene"] == "fern" and d["strategy"] == "random"]
        
        if u_data and r_data:
            u_psnr = mean([d["psnr"] for d in u_data])
            r_psnr = mean([d["psnr"] for d in r_data])
            delta = u_psnr - r_psnr
            ablation_rows.append([m, f"{u_psnr:.2f}", f"{r_psnr:.2f}", f"{delta:+.2f}"])
    
    if ablation_rows:
        print("### 🎲 Ablação de Estratégia de Amostragem (Uniform vs Random)")
        print_markdown_table(["Modelo", "Uniform PSNR", "Random PSNR", "$\\Delta$ PSNR (Unif - Rand)"], ablation_rows)

if __name__ == "__main__":
    main()
