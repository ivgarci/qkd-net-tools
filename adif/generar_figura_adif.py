"""Genera adif_resiliencia.pdf con el estilo de las figuras de resiliencia existentes."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Datos ──────────────────────────────────────────────────────────────────────
with open('/Users/igarcia/doctorado/2025_2026/mapas/resultados_adif_junctions.json') as f:
    data = json.load(f)

p      = [v * 100 for v in data['attack_degree']['p_values']]
S_deg  = [v * 100 for v in data['attack_degree']['S_values']]
S_cb   = [v * 100 for v in data['attack_cb']['S_values']]

p_star_deg = data['attack_degree']['p_star'] * 100   # 5 %
p_star_cb  = data['attack_cb']['p_star']    * 100   # 10 %

# Valor S en p*
S_at_pstar_deg = S_deg[int(p_star_deg)]
S_at_pstar_cb  = S_cb[int(p_star_cb)]

# ── Estilo ─────────────────────────────────────────────────────────────────────
COLOR_DEG  = '#1f77b4'   # azul (coherente con CyL en figuras existentes)
COLOR_CB   = '#d95f02'   # naranja tostado
COLOR_THR  = '#555555'   # gris umbral
LINEWIDTH  = 2.0
FONT_TITLE = 13
FONT_LABEL = 11
FONT_TICK  = 10
FONT_LEG   = 10

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#cccccc',
    'grid.linewidth': 0.6,
    'grid.linestyle': '-',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
})

fig, ax = plt.subplots(figsize=(8.5, 4.4))

# ── Curvas principales ─────────────────────────────────────────────────────────
ax.plot(p, S_deg, color=COLOR_DEG, linewidth=LINEWIDTH,
        linestyle='-', label='Ataque dirigido por grado ($C_D$)', zorder=3)
ax.plot(p, S_cb,  color=COLOR_CB,  linewidth=LINEWIDTH,
        linestyle='--', label='Ataque dirigido por intermediación ($C_B$)', zorder=3)

# ── Umbral S = 0.5 ────────────────────────────────────────────────────────────
ax.axhline(50, color=COLOR_THR, linewidth=1.2, linestyle='--',
           label='Umbral $S = 0{,}5$', zorder=2)

# ── Líneas verticales p* ──────────────────────────────────────────────────────
ax.axvline(p_star_deg, ymin=0, ymax=S_at_pstar_deg / 100,
           color=COLOR_DEG, linewidth=1.2, linestyle=':', zorder=2)
ax.axvline(p_star_cb,  ymin=0, ymax=S_at_pstar_cb  / 100,
           color=COLOR_CB,  linewidth=1.2, linestyle=':', zorder=2)

# ── Marcadores en el cruce ────────────────────────────────────────────────────
ax.plot(p_star_deg, S_at_pstar_deg, 'o',
        color=COLOR_DEG, markersize=7, zorder=5,
        label=f'$p^\\star = {p_star_deg:.0f}\\%$ (grado)')
ax.plot(p_star_cb,  S_at_pstar_cb,  's',
        color=COLOR_CB,  markersize=7, zorder=5,
        label=f'$p^\\star = {p_star_cb:.0f}\\%$ ($C_B$)')

# ── Anotaciones de p* en el eje X ─────────────────────────────────────────────
ax.annotate(f'{p_star_deg:.0f}%',
            xy=(p_star_deg, 0), xytext=(p_star_deg + 0.3, -8),
            fontsize=FONT_TICK, color=COLOR_DEG, ha='left')
ax.annotate(f'{p_star_cb:.0f}%',
            xy=(p_star_cb, 0), xytext=(p_star_cb + 0.3, -8),
            fontsize=FONT_TICK, color=COLOR_CB, ha='left')

# ── Ejes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(0, 49)
ax.set_ylim(0, 105)
ax.set_xlabel('Fracción de nodos eliminados $p$ (%)', fontsize=FONT_LABEL)
ax.set_ylabel('Tamaño relativo de la\ncomponente gigante $S(G_p)$', fontsize=FONT_LABEL)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.tick_params(labelsize=FONT_TICK)

# ── Título ────────────────────────────────────────────────────────────────────
ax.set_title('Degradación bajo ataques dirigidos — Red ADIF (grafo de junctions, $|V|=485$)',
             fontsize=FONT_TITLE, pad=10)

# ── Leyenda ───────────────────────────────────────────────────────────────────
ax.legend(fontsize=FONT_LEG, loc='upper right', framealpha=0.9,
          edgecolor='#cccccc')

fig.tight_layout()

out_pdf = '/Users/igarcia/doctorado/2025_2026/697937f94a86c11bc36ad509/Figures/adif_resiliencia.pdf'
out_png = '/Users/igarcia/doctorado/2025_2026/697937f94a86c11bc36ad509/Figures/adif_resiliencia.png'
fig.savefig(out_pdf, format='pdf', dpi=150, bbox_inches='tight')
fig.savefig(out_png, format='png', dpi=150, bbox_inches='tight')
print(f'Guardado: {out_pdf}')
print(f'Guardado: {out_png}')
