"""
Distribución de grado del grafo CyL con ajuste log-log para clasificación topológica.
Determina si la red sigue una ley de potencias (scale-free) o distribución exponencial.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_CYL = os.path.join(BASE, '..', 'datos', 'cyl')
FIGS_CYL = os.path.join(BASE, '..', 'figuras', 'cyl')

os.makedirs(FIGS_CYL, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'figure.dpi': 150,
})

# Cargar grafo
adj = pd.read_csv(os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv'), index_col=0)
G = nx.from_pandas_adjacency(adj)
degree_sequence = [d for n, d in G.degree()]

# ── Ajuste log-log de distribución de grado ────────────────────────────────────
k_vals, counts = np.unique(degree_sequence, return_counts=True)
pk_vals = counts / counts.sum()

# Usar solo k >= 2 para el ajuste (evitar ruido en cola baja)
mask = (k_vals >= 2) & (pk_vals > 0)
log_k  = np.log10(k_vals[mask].astype(float))
log_pk = np.log10(pk_vals[mask].astype(float))

slope, intercept, r_val, p_val, se = linregress(log_k, log_pk)
alpha = -slope
r2    = r_val ** 2

print(f"Exponente de escala libre: α = {alpha:.3f}")
print(f"Coeficiente de determinación: R² = {r2:.4f}")
if r2 >= 0.80 and 2.0 <= alpha <= 3.5:
    print("  → Distribución compatible con ley de potencias (red scale-free)")
else:
    print("  → Distribución no compatible con ley de potencias (red geográfica/exponencial)")

# ── Figura con histograma + ajuste ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel izquierdo: histograma lineal
ax = axes[0]
ax.hist(degree_sequence, bins=30, color='steelblue', edgecolor='black', alpha=0.8)
ax.set_title("Distribución del grado — CyL (100 nodos)")
ax.set_xlabel("Grado $k$")
ax.set_ylabel("Frecuencia")
ax.grid(axis='y', alpha=0.3)

# Panel derecho: escala log-log con ajuste
ax = axes[1]
ax.scatter(k_vals, pk_vals, color='steelblue', s=40, zorder=5, label='Datos empíricos')

# Curva ajustada
k_fit = np.linspace(max(k_vals[mask].min(), 1), k_vals.max(), 200)
pk_fit = 10 ** (intercept + slope * np.log10(k_fit))
ax.plot(k_fit, pk_fit, 'r--', lw=1.8,
        label=rf'Ajuste: $P(k) \propto k^{{-\alpha}}$, $\alpha={alpha:.2f}$, $R^2={r2:.2f}$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title("Distribución de grado (escala log-log)")
ax.set_xlabel("Grado $k$")
ax.set_ylabel("$P(k)$")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

fig.tight_layout()

for ext in ('png', 'pdf', 'svg'):
    path = os.path.join(FIGS_CYL, f'distribucion_grado_grafo.{ext}')
    fig.savefig(path, dpi=300 if ext == 'png' else 150, bbox_inches='tight')
    print(f"Guardado: {path}")

plt.close(fig)
