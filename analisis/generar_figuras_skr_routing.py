"""
Figures for QKD-SKR-Routing paper — unique replacements for shared figures.

Generates:
  QKD-SKR-Routing/Figures/benchmarks_qkd_metricas.pdf
      2-panel: (a) edge-distance distribution for Spain QKD network,
               (b) estimated SKR distribution using BB84 physical model.
  QKD-SKR-Routing/Figures/esp_topologia.png
      Geographic Spain QKD topology with edges coloured by SKR feasibility
      (feasible ≤ 50 km vs relay-required > 50 km).
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_ESP  = os.path.join(BASE, '..', 'datos', 'espana')
PAPER_DIR = os.path.join(BASE, '..', '..', '..', 'articulos',
                         'QKD-SKR-Routing', 'Figures')
os.makedirs(PAPER_DIR, exist_ok=True)

DELTA_KM  = 50.0          # QKD feasibility threshold (km)
L_ATT     = 22.0          # effective attenuation length (km) for α = 0.2 dB/km
R0_bpp    = 1.0           # normalised SKR at d = 0 (bits per pulse)

# ── helpers ──────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ = radians(lat2 - lat1)
    Δλ = radians(lon2 - lon1)
    a = sin(Δφ/2)**2 + cos(φ1)*cos(φ2)*sin(Δλ/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def skr_model(d_km):
    """Simplified BB84 SKR model: R(d) = R0 * exp(-d / L_att)."""
    return R0_bpp * np.exp(-d_km / L_ATT)


# ── load data ─────────────────────────────────────────────────────────────────

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

# ── compute edge distances ────────────────────────────────────────────────────

print('Computing edge distances...')
edge_data = []
pos = {}
for n in G.nodes():
    if n in coords:
        lat, lon = coords[n]
        pos[n] = (lon, lat)

for u, v in G.edges():
    if u in coords and v in coords:
        d = haversine(coords[u][0], coords[u][1],
                      coords[v][0], coords[v][1])
        edge_data.append({'u': u, 'v': v, 'dist_km': d, 'skr': skr_model(d)})

df_edges = pd.DataFrame(edge_data)
distances = df_edges['dist_km'].values
skr_vals  = df_edges['skr'].values
p25, p50, p75, p90 = (np.percentile(distances, q) for q in (25, 50, 75, 90))
print(f'  Edges with distance: {len(df_edges)} / {G.number_of_edges()}')
print(f'  Median: {p50:.1f} km  |  P90: {p90:.1f} km  |  Max: {distances.max():.1f} km')

# ── Figure 1: physical metrics (2-panel) ─────────────────────────────────────

print('Generating benchmarks figure...')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

# Panel (a): edge-distance distribution
ax1.hist(distances, bins=55, color='#2166ac', alpha=0.78,
         edgecolor='white', linewidth=0.4)
for pct, label, ls in [(p50, f'Median {p50:.0f} km', ':'),
                       (p90, f'P90 = {p90:.0f} km', '--')]:
    ax1.axvline(pct, color='#d95f02', lw=1.4, ls=ls, label=label)
ax1.set_xlabel('Edge distance $d$ (km)', fontsize=11)
ax1.set_ylabel('Edge count', fontsize=11)
ax1.set_title('(a) Distance distribution\n'
              rf'Spain QKD network ($|E|={len(distances):,}$ edges)', fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.25)

# Panel (b): SKR vs distance scatter + theoretical curve
d_curve = np.linspace(0, distances.max(), 300)
r_curve = skr_model(d_curve)
ax2.scatter(distances, skr_vals, s=1.5, color='#4393c3', alpha=0.35, label='Network edges')
ax2.plot(d_curve, r_curve, color='#d73027', lw=2.0, zorder=5,
         label=r'$R(d)=R_0\,e^{-d/L_{\rm att}}$')
ax2.axvline(p50, color='grey', lw=0.9, ls=':', alpha=0.7, label=f'Median {p50:.0f} km')
ax2.set_xlabel('Edge distance $d$ (km)', fontsize=11)
ax2.set_ylabel('Normalised SKR $R(d)/R_0$', fontsize=11)
ax2.set_title(r'(b) SKR vs distance — $L_{\rm att}=22$ km' '\n'
              r'(BB84, $\alpha=0.2$ dB/km, $\eta_{\rm det}=0.2$)', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.25)

fig.tight_layout()
for ext in ('pdf', 'png'):
    out = os.path.join(PAPER_DIR, f'benchmarks_qkd_metricas.{ext}')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  Saved: {out}')
plt.close(fig)

# ── Figure 2: Spain QKD topology — edges coloured by distance quantile ────────

print('Generating Spain topology figure...')
if len(pos) < 100:
    print('  Not enough coordinates for geographic plot — skipping topology figure.')
else:
    from matplotlib.lines import Line2D
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(9, 7))

    # Build edge lookup for distance
    dist_lookup = {frozenset([r['u'], r['v']]): r['dist_km'] for _, r in df_edges.iterrows()}
    cmap = plt.cm.plasma_r

    # Draw edges coloured by distance
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
            norm_d = d / distances.max()
            color  = cmap(norm_d)
            alpha  = 0.55
            lw     = 0.5 + norm_d * 1.0
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=color, alpha=alpha, lw=lw, solid_capstyle='round')

    # Draw nodes
    nodes_geo = [n for n in G.nodes() if n in pos]
    deg = dict(G.degree())
    node_sizes  = [3 + 2.5 * deg.get(n, 1) for n in nodes_geo]
    node_colors = ['#d73027' if deg.get(n, 0) >= 8 else '#ffffff' for n in nodes_geo]
    node_ec     = ['#800026' if deg.get(n, 0) >= 8 else '#2166ac' for n in nodes_geo]
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_geo, ax=ax,
                           node_size=node_sizes, node_color=node_colors,
                           edgecolors=node_ec, linewidths=0.5, alpha=0.9)

    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=distances.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('Edge distance (km)', fontsize=9)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d73027',
               markeredgecolor='#800026', markersize=7, label='High-degree hub (deg $\\geq$ 8)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='#2166ac', markersize=5, label='Standard node'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8.5, framealpha=0.9)
    ax.set_xlabel('Longitude (°)', fontsize=10)
    ax.set_ylabel('Latitude (°)', fontsize=10)
    ax.set_title(f'Spain national QKD topology: $|V|={G.number_of_nodes()}$ nodes, '
                 f'$|E|={G.number_of_edges()}$ edges\n'
                 'Edge colour $\\propto$ link distance (darker = longer)',
                 fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    for ext in ('png', 'pdf'):
        out = os.path.join(PAPER_DIR, f'esp_topologia.{ext}')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  Saved: {out}')
    plt.close(fig)

print('Done.')
