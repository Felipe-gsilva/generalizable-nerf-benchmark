import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data from Table 2 and Table 5
data = {
    'Modelo': ['PixelNeRF (TTA)', 'GNT (TTA)', 'Instant-NGP (Per-Scene)', 'Nerfacto (Per-Scene)'],
    'VRAM (GB)': [3.42, 0.19, 0.15, 0.05],
    'PSNR (dB)': [6.98, 10.34, 15.29, 25.33]
}

df = pd.DataFrame(data)

# Set style
sns.set_theme(style="whitegrid", palette="deep")
plt.figure(figsize=(8, 6))

ax = sns.scatterplot(
    data=df, 
    x='VRAM (GB)', 
    y='PSNR (dB)', 
    hue='Modelo',
    style='Modelo',
    s=200, 
    alpha=0.9,
    markers=['o', 's', 'D', '^']
)

# Annotate points
for i in range(len(df)):
    ax.text(df['VRAM (GB)'][i] + 0.05, df['PSNR (dB)'][i], 
            df['Modelo'][i].split(' ')[0], 
            horizontalalignment='left', 
            size='medium', color='black', weight='semibold')

# Set labels and title
plt.xlabel("Pico de VRAM Dinâmico (GB) $\\downarrow$", fontsize=12, fontweight='bold')
plt.ylabel("Fidelidade Visual - PSNR (dB) $\\uparrow$", fontsize=12, fontweight='bold')
plt.title("Trade-off Computacional vs. Qualidade Visual", fontsize=14, fontweight='bold')

# Remove legend as labels are directly on points
plt.legend([],[], frameon=False)

# Add ideal quadrant highlight
plt.axvspan(-0.1, 0.5, ymin=0.5, ymax=1.0, alpha=0.1, color='green', label='Região Ideal')
plt.text(0.2, 28, "Região Ideal\n(Baixa VRAM, Alta Qualidade)", color='green', alpha=0.7, ha='center', va='center', fontweight='bold', fontsize=10)

# Adjust limits
plt.xlim(-0.2, 4.0)
plt.ylim(5.0, 30.0)

# Save
plt.tight_layout()
plt.savefig('/home/felipe-gsilva/dev/cs/nerf-ann-paper/docs/img/grafico_tradeoff.pdf', format='pdf', dpi=300)
plt.close()
