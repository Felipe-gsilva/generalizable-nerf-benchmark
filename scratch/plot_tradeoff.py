import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data from Table 2 and Table 5
data = {
    'Modelo': ['PixelNeRF (TTA)', 'GNT (TTA)', 'Instant-NGP (Per-Scene)', 'Nerfacto (Per-Scene)'],
    'VRAM (GB)': [1.93, 1.93, 1.30, 1.30],
    'PSNR (dB)': [6.98, 10.34, 13.57, 12.04]
}

df = pd.DataFrame(data)

# Set style
sns.set_theme(style="whitegrid", palette="deep")
plt.figure(figsize=(8, 6))

# Plot
# Add a slight jitter to X axis if points overlap exactly
df['VRAM (GB) Jittered'] = df['VRAM (GB)'] + np.random.normal(0, 0.02, size=len(df))

ax = sns.scatterplot(
    data=df, 
    x='VRAM (GB) Jittered', 
    y='PSNR (dB)', 
    hue='Modelo',
    style='Modelo',
    s=200, 
    alpha=0.9,
    markers=['o', 's', 'D', '^']
)

# Annotate points
for i in range(len(df)):
    ax.text(df['VRAM (GB) Jittered'][i] + 0.02, df['PSNR (dB)'][i], 
            df['Modelo'][i].split(' ')[0], 
            horizontalalignment='left', 
            size='medium', color='black', weight='semibold')

# Set labels and title
plt.xlabel("Consumo de VRAM (GB) $\\downarrow$", fontsize=12, fontweight='bold')
plt.ylabel("Fidelidade Visual - PSNR (dB) $\\uparrow$", fontsize=12, fontweight='bold')
plt.title("Trade-off Computacional vs. Qualidade Visual (10 Vistas)", fontsize=14, fontweight='bold')

# Remove legend as labels are directly on points
plt.legend([],[], frameon=False)

# Add ideal quadrant highlight
plt.axvspan(1.0, 1.6, ymin=0.5, ymax=1.0, alpha=0.1, color='green', label='Região Ideal')
plt.text(1.3, 13.8, "Região Ideal\n(Baixa VRAM, Alta Qualidade)", color='green', alpha=0.7, ha='center', va='center', fontweight='bold', fontsize=10)


# Adjust limits
plt.xlim(1.0, 2.2)
plt.ylim(5.0, 15.0)

# Save
plt.tight_layout()
plt.savefig('/home/felipe-gsilva/dev/cs/nerf-ann-paper/docs/img/grafico_tradeoff.pdf', format='pdf', dpi=300)
plt.close()
