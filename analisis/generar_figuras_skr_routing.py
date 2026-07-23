"""
Figures for QKD-SKR-Routing paper.

Generates:
  figuras/qkd_skr_routing/skr_vs_distancia.pdf
      SKR vs fibre distance curve (Fig 1 in paper): ideal asymptotic BB84
      model with declared parameters (η_det=0.10, μ=0.5,
      α=0.2 dB/km, p_dark=1e-6, e_det=0.015, f_EC=1.16).
      Shaded region marks d > Δ = 45 km.

  figuras/qkd_skr_routing/benchmarks_qkd_metricas.pdf
      2-panel (Fig 4 in paper):
        (a) Edge-distance distribution for Spain QKD network.
        (b) SKR distribution (histogram) across all 5681 edges,
            computed with the canonical ideal asymptotic model (η_det=0.10).

  figuras/qkd_skr_routing/esp_topologia.png
      Geographic Spain QKD topology with edges coloured by distance.

The SKR formula is not duplicated here. It is imported from
``protocols/skr_bb84.py`` so figures and tabular results use the same canonical
Lo--Ma--Chen ideal asymptotic model, including the BB84 sifting factor q=1/2.
It is not a finite-decoy implementation or an experimental calibration.

Run with:
    python analisis/generar_figuras_skr_routing.py
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_ESP  = os.path.join(BASE, '..', 'datos', 'espana')
REPO_ROOT = os.path.abspath(os.path.join(BASE, '..'))
PAPER_DIR = os.environ.get(
    'QKD_SKR_FIGURE_DIR',
    os.path.join(REPO_ROOT, 'figuras', 'qkd_skr_routing'),
)
os.makedirs(PAPER_DIR, exist_ok=True)

# ── Canonical physical model ─────────────────────────────────────────────────
sys.path.insert(0, REPO_ROOT)
from protocols.skr_bb84 import (  # noqa: E402
    ALPHA_DB_KM,
    E_DETECTOR as E_DET,
    ETA_DET,
    F_EC,
    MU,
    P_DARK,
    Q_SIFT,
    skr_bb84_asymptotic,
)

DELTA_KM    = 45.0     # QKD network design threshold (km)
RHO_F       = 1.25     # Routing factor (Haversine → fibre distance)


def skr_vec(d_array: np.ndarray, **kwargs) -> np.ndarray:
    """Vectorised wrapper for the canonical ideal asymptotic model."""
    return np.array([skr_bb84_asymptotic(float(d), **kwargs)
                     for d in d_array])


# ── Haversine distance ────────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (km)."""
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1.0 - a))


# ── Load Spain network data ───────────────────────────────────────────────────

print('Loading Spain adjacency matrix...')
adj_df = pd.read_csv(os.path.join(DATA_ESP, 'AdjacencyMatrixNamed45.csv'),
                     index_col=0)
G = nx.from_pandas_adjacency(adj_df)
if not nx.is_connected(G):
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
print(f'  |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}')

print('Loading coordinates...')
coords_raw = pd.read_csv(os.path.join(DATA_ESP, 'peninsula_1000.csv'),
                         sep=';', decimal=',')
coords = {}
for _, row in coords_raw.iterrows():
    name = str(row['Población']).strip()
    try:
        coords[name] = (float(str(row['Latitud']).replace(',', '.')),
                        float(str(row['Longitud']).replace(',', '.')))
    except (ValueError, KeyError):
        pass

# ── Compute edge distances and SKR using the canonical model ─────────────────

print('Computing edge distances and SKR (canonical ideal asymptotic model)...')
edge_data = []
pos = {}
for n in G.nodes():
    if n in coords:
        lat, lon = coords[n]
        pos[n] = (lon, lat)

for u, v in G.edges():
    if u in coords and v in coords:
        d_hav = haversine(coords[u][0], coords[u][1],
                          coords[v][0], coords[v][1])
        d_fibre = d_hav * RHO_F   # apply routing factor
        skr_val = skr_bb84_asymptotic(d_fibre)
        edge_data.append({'u': u, 'v': v,
                          'dist_km': d_fibre,
                          'skr': skr_val})

df_edges = pd.DataFrame(edge_data)
distances = df_edges['dist_km'].values
skr_vals  = df_edges['skr'].values
p25, p50, p75, p90 = (np.percentile(distances, q) for q in (25, 50, 75, 90))
print(f'  Edges with distance: {len(df_edges)} / {G.number_of_edges()}')
print(f'  Median: {p50:.1f} km  |  P90: {p90:.1f} km  |  Max: {distances.max():.1f} km')

# ── Figure 1: SKR vs distance (skr_vs_distancia.pdf) ─────────────────────────
# Paper caption: SKR vs fibre distance with shaded region d > Δ = 45 km,
# declared parameters α=0.2, η_det=0.10, μ=0.5, e_det=0.015,
# p_dark=1e-6, f_EC=1.16.

print('\nGenerating Fig 1: skr_vs_distancia...')
d_range = np.linspace(0.1, 200.0, 600)
r_range = skr_vec(d_range)
positive = r_range > 0

fig, ax = plt.subplots(figsize=(8, 5))

# Main SKR curve
ax.semilogy(d_range[positive], r_range[positive],
            color='steelblue', lw=2.2,
            label='Ideal asymptotic BB84 (exact decoy estimate)')

# Shaded region: d > Δ = 45 km  (excluded by network design threshold)
d_max_plot = d_range[positive][-1] if positive.any() else 200.0
ax.axvspan(DELTA_KM, d_max_plot, alpha=0.12, color='red',
           label=rf'$d > \Delta = {DELTA_KM:.0f}$ km (excluded region)')
ax.axvline(DELTA_KM, color='red', lw=1.4, ls='--', alpha=0.8)

# Annotate the Δ = 45 km operating point
r_delta = skr_bb84_asymptotic(DELTA_KM)
ax.scatter([DELTA_KM], [r_delta], color='red', zorder=6, s=60)
ax.annotate(
    rf'$\Delta = {DELTA_KM:.0f}$ km'
    f'\n$R = {r_delta:.2e}$ bits/pulse',
    xy=(DELTA_KM, r_delta),
    xytext=(DELTA_KM + 18, r_delta * 4),
    fontsize=8.5, color='red',
    arrowprops=dict(arrowstyle='->', color='red', lw=0.9)
)

# Annotate a few key operating points
for d_pt, label in [(10, '10 km'), (20, '20 km'), (30, '30 km')]:
    r_pt = skr_bb84_asymptotic(d_pt)
    ax.scatter([d_pt], [r_pt], color='darkorange', zorder=5, s=40, alpha=0.8)

ax.set_xlabel('Fibre distance $d$ (km)', fontsize=12)
ax.set_ylabel('SKR $R(d)$ (bits/pulse)', fontsize=12)
ax.set_title(
    r'Secret Key Rate vs. Fibre Distance \textemdash Ideal Asymptotic BB84' '\n'
    r'($\alpha=0.2$ dB/km, $\eta_{\rm det}=0.10$, $\mu=0.5$, '
    r'$e_{\rm det}=0.015$, $p_{\rm dark}=10^{-6}$, '
    r'$f_{\rm EC}=1.16$, $q=1/2$)',
    fontsize=10
)
ax.set_xlim(0, 200)
ax.legend(fontsize=9, loc='lower left')
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()

for ext in ('pdf', 'png'):
    out = os.path.join(PAPER_DIR, f'skr_vs_distancia.{ext}')
    metadata = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else None
    fig.savefig(out, dpi=150, bbox_inches='tight', metadata=metadata)
    print(f'  Saved: {out}')
plt.close(fig)

# Verify key operating points computed from the canonical declared equations.
print('\n  Verification of canonical operating points:')
for d_chk, r_expected in [(10, 6.4664382800499395e-3),
                           (45, 1.2834511689390389e-3),
                           (50, 1.0188456107580379e-3),
                           (100, 9.996046924682088e-5)]:
    r_got = skr_bb84_asymptotic(d_chk)
    match = 'OK' if abs(r_got - r_expected) / r_expected < 0.05 else 'MISMATCH'
    print(f'  d={d_chk:3d} km: R={r_got:.3e} (expected {r_expected:.2e}) [{match}]')


# ── Figure 4: Physical metrics 2-panel (benchmarks_qkd_metricas.pdf) ─────────
# Paper caption (Fig 4): Top: edge distance distribution.
#                        Bottom: SKR distribution.
# Both computed with the canonical ideal asymptotic model, η_det=0.10.

print('\nGenerating Fig 4: benchmarks_qkd_metricas...')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))

# Panel (a): edge-distance distribution
ax1.hist(distances, bins=55, color='#2166ac', alpha=0.78,
         edgecolor='white', linewidth=0.4)
for pct, label, ls in [(p50, f'Median {p50:.0f} km', ':'),
                       (p90, f'P90 = {p90:.0f} km', '--')]:
    ax1.axvline(pct, color='#d95f02', lw=1.4, ls=ls, label=label)
ax1.axvline(DELTA_KM, color='red', lw=1.2, ls='-.',
            label=rf'$\Delta = {DELTA_KM:.0f}$ km threshold')
ax1.set_xlabel('Edge distance $d$ (km)', fontsize=11)
ax1.set_ylabel('Edge count', fontsize=11)
ax1.set_title(
    rf'(a) Distance distribution — Spain QKD network ($|E|={len(distances):,}$ edges)',
    fontsize=10
)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.25)

# Panel (b): SKR distribution (histogram) — as described in paper caption
# "Bottom: SKR distribution."
# Filter to positive SKR values (all edges should have SKR > 0 since d ≤ Δ)
skr_positive = skr_vals[skr_vals > 0]
ax2.hist(skr_positive, bins=55, color='#4393c3', alpha=0.78,
         edgecolor='white', linewidth=0.4)
skr_delta = skr_bb84_asymptotic(DELTA_KM)
skr_p50   = np.percentile(skr_positive, 50)
ax2.axvline(skr_delta, color='red', lw=1.2, ls='--',
            label=rf'$R(\Delta={DELTA_KM:.0f}$ km$) = {skr_delta:.2e}$ bits/pulse')
ax2.axvline(skr_p50,   color='#d95f02', lw=1.4, ls=':',
            label=f'Median $R = {skr_p50:.2e}$ bits/pulse')
ax2.set_xlabel('SKR $R(d)$ (bits/pulse)', fontsize=11)
ax2.set_ylabel('Edge count', fontsize=11)
ax2.set_title(
    r'(b) SKR distribution — ideal asymptotic BB84, '
    r'$\eta_{\rm det}=0.10$, $\mu=0.5$, $q=1/2$',
    fontsize=10
)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.25)

fig.tight_layout()
for ext in ('pdf', 'png'):
    out = os.path.join(PAPER_DIR, f'benchmarks_qkd_metricas.{ext}')
    metadata = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else None
    fig.savefig(out, dpi=150, bbox_inches='tight', metadata=metadata)
    print(f'  Saved: {out}')
plt.close(fig)


# ── Figure 3: Spain QKD topology (esp_topologia.png) ─────────────────────────

print('\nGenerating Fig 3: esp_topologia...')
if len(pos) < 100:
    print('  Not enough coordinates for geographic plot — skipping topology figure.')
else:
    from matplotlib.lines import Line2D
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(9, 7))

    # Build edge lookup for distance
    dist_lookup = {frozenset([r['u'], r['v']]): r['dist_km']
                   for _, r in df_edges.iterrows()}
    cmap = plt.cm.RdYlBu_r   # blue=short/high-SKR, red=long/low-SKR

    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue
        key = frozenset([u, v])
        d = dist_lookup.get(key, np.nan)
        if np.isnan(d):
            color = '#cccccc'
            alpha = 0.3
            lw = 0.3
        else:
            norm_d = d / DELTA_KM   # normalise to [0, 1] relative to threshold
            norm_d = min(norm_d, 1.0)
            color  = cmap(norm_d)
            alpha  = 0.55
            lw     = 0.4 + norm_d * 0.8
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=color, alpha=alpha, lw=lw, solid_capstyle='round')

    # Draw nodes
    nodes_geo = [n for n in G.nodes() if n in pos]
    deg = dict(G.degree())
    node_sizes  = [3 + 2.5 * deg.get(n, 1) for n in nodes_geo]
    node_colors = ['#d73027' if deg.get(n, 0) >= 8 else '#ffffff'
                   for n in nodes_geo]
    node_ec     = ['#800026' if deg.get(n, 0) >= 8 else '#2166ac'
                   for n in nodes_geo]
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_geo, ax=ax,
                           node_size=node_sizes, node_color=node_colors,
                           edgecolors=node_ec, linewidths=0.5, alpha=0.9)

    sm = ScalarMappable(cmap=cmap,
                        norm=Normalize(vmin=0, vmax=DELTA_KM))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('Edge fibre distance (km)', fontsize=9)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d73027',
               markeredgecolor='#800026', markersize=7,
               label='High-degree hub (deg $\\geq$ 8)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='#2166ac', markersize=5,
               label='Standard node'),
    ]
    ax.legend(handles=legend_elements, loc='lower left',
              fontsize=8.5, framealpha=0.9)
    ax.set_xlabel('Longitude ($^\\circ$)', fontsize=10)
    ax.set_ylabel('Latitude ($^\\circ$)', fontsize=10)
    ax.set_title(
        f'Spain national QKD topology: $|V|={G.number_of_nodes()}$ nodes, '
        f'$|E|={G.number_of_edges()}$ edges, $\\Delta={DELTA_KM:.0f}$ km\n'
        'Edge colour: blue (short, high-SKR) $\\to$ red (long, low-SKR)',
        fontsize=10
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    for ext in ('png', 'pdf'):
        out = os.path.join(PAPER_DIR, f'esp_topologia.{ext}')
        metadata = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else None
        fig.savefig(out, dpi=150, bbox_inches='tight', metadata=metadata)
        print(f'  Saved: {out}')
    plt.close(fig)

print('\nDone.')
