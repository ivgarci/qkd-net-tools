"""
k-core decomposition of the ADIF junction graph — ADIF-QKD paper figure.

Two-panel layout:
  Left  — geographic map with nodes coloured by k-core shell index.
  Right — horizontal bar chart: nodes per k-shell (reveals quasi-tree structure).

Saves: ADIF-QKD/Figures/k_core_adif.pdf / .png
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_ADIF = os.path.join(BASE, '..', 'datos', 'adif')

if len(sys.argv) > 1:
    FIGS_OUT = sys.argv[1]
else:
    FIGS_OUT = os.path.join(BASE, '..', '..', '..', 'articulos', 'ADIF-QKD', 'Figures')
os.makedirs(FIGS_OUT, exist_ok=True)


# ── build junction graph (same procedure as other ADIF scripts) ───────────────

def load_adif_junction_graph():
    nodes_df = pd.read_csv(
        os.path.join(DATA_ADIF, 'nodos_red_adif.csv'), quotechar='"', on_bad_lines='skip'
    )
    adj_df = pd.read_csv(
        os.path.join(DATA_ADIF, 'adyacencia_red_adif.csv'), quotechar='"', on_bad_lines='skip'
    )
    G_full = nx.Graph()
    connected = nodes_df[nodes_df['conectado'] == 'SI'].copy()
    for _, row in connected.iterrows():
        G_full.add_node(str(row['cod']),
                        nombre=str(row['nombre']),
                        lat=float(row['lat']),
                        lon=float(row['lon']))
    seen = set()
    for _, row in adj_df.iterrows():
        u, v = str(row['cod']), str(row['vecino_cod'])
        key = frozenset([u, v])
        if key in seen:
            continue
        seen.add(key)
        if G_full.has_node(u) and G_full.has_node(v):
            try:
                G_full.add_edge(u, v, dist_km=float(row['dist_km']))
            except (ValueError, TypeError, KeyError):
                continue
    comps = sorted(nx.connected_components(G_full), key=len, reverse=True)
    G_lcc = G_full.subgraph(comps[0]).copy()
    keep = {n for n in G_lcc.nodes() if G_lcc.degree(n) != 2}
    J = nx.Graph()
    for n in keep:
        J.add_node(n, **G_lcc.nodes[n])
    visited = set()
    for start in keep:
        for nbr in list(G_lcc.neighbors(start)):
            ek = frozenset([start, nbr])
            if ek in visited:
                continue
            visited.add(ek)
            acc = G_lcc[start][nbr].get('dist_km', 0.0) or 0.0
            prev, cur = start, nbr
            while cur not in keep:
                neighbors = list(G_lcc.neighbors(cur))
                nxt = neighbors[0] if neighbors[1] == prev else neighbors[1]
                acc += G_lcc[cur][nxt].get('dist_km', 0.0) or 0.0
                visited.add(frozenset([cur, nxt]))
                prev, cur = cur, nxt
            if cur != start and not J.has_edge(start, cur):
                J.add_edge(start, cur, dist_km=acc)
    jcomps = list(nx.connected_components(J))
    if len(jcomps) > 1:
        J = J.subgraph(max(jcomps, key=len)).copy()
    return J


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading ADIF junction graph...')
    J = load_adif_junction_graph()
    V, E = J.number_of_nodes(), J.number_of_edges()
    print(f'  |V|={V}, |E|={E}')

    # k-core decomposition
    core = nx.core_number(J)
    k_max = max(core.values())
    shells = {}
    for n, k in core.items():
        shells.setdefault(k, []).append(n)
    print(f'  k_max = {k_max}')
    for k in sorted(shells):
        print(f'  k={k}: {len(shells[k])} nodes')

    # geographic positions
    pos = {n: (J.nodes[n]['lon'], J.nodes[n]['lat']) for n in J.nodes()}

    # ── colour palette: one distinct colour per k-shell ───────────────────────
    # k_max is small (quasi-tree → typically 2–4), so discrete colours work well
    shell_vals = sorted(shells.keys())
    n_shells   = len(shell_vals)
    palette    = ['#d1e5f0', '#74add1', '#4575b4', '#313695',
                  '#fee090', '#f46d43', '#d73027', '#a50026']
    shell_colors = {k: palette[i % len(palette)] for i, k in enumerate(shell_vals)}

    node_colors = [shell_colors[core[n]] for n in J.nodes()]
    node_sizes  = [8 + 18 * core[n] for n in J.nodes()]

    # bridges (k=1 only → articulation points that are also degree-1 in k-core)
    bridges = list(nx.bridges(J))

    # ── figure ────────────────────────────────────────────────────────────────
    fig, (ax_map, ax_bar) = plt.subplots(
        1, 2, figsize=(12, 5.5),
        gridspec_kw={'width_ratios': [2.2, 1]}
    )

    # --- left: geographic map ------------------------------------------------
    # Draw non-bridge edges
    bridge_set = set(map(frozenset, bridges))
    normal_edges = [(u, v) for u, v in J.edges() if frozenset([u, v]) not in bridge_set]
    nx.draw_networkx_edges(J, pos, edgelist=normal_edges, ax=ax_map,
                           edge_color='#aaaaaa', width=0.6, alpha=0.55)
    # Highlight bridges
    if bridges:
        nx.draw_networkx_edges(J, pos, edgelist=bridges, ax=ax_map,
                               edge_color='#d95f02', width=1.4, alpha=0.85,
                               style='dashed')

    # Draw nodes per shell (outermost first so inner shells appear on top)
    for k in shell_vals:
        ns = shells[k]
        nx.draw_networkx_nodes(J, pos, nodelist=ns, ax=ax_map,
                               node_color=shell_colors[k],
                               node_size=[node_sizes[list(J.nodes()).index(n)] for n in ns],
                               alpha=0.92, edgecolors='white', linewidths=0.4)

    # Legend for k-shells + bridges
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=shell_colors[k], markersize=7 + k,
               markeredgecolor='white',
               label=f'$k$-shell {k}  ({len(shells[k])} nodes)')
        for k in shell_vals
    ]
    legend_elements.append(
        Line2D([0], [0], color='#d95f02', lw=1.5, ls='--',
               label=f'Bridge ({len(bridges)} edges)')
    )
    ax_map.legend(handles=legend_elements, loc='lower left',
                  fontsize=8.5, framealpha=0.92, edgecolor='#cccccc')

    ax_map.set_xlabel('Longitude (°)', fontsize=10)
    ax_map.set_ylabel('Latitude (°)',  fontsize=10)
    ax_map.set_title(
        rf'(a) $k$-core decomposition — ADIF junction graph'
        '\n'
        rf'$|V_J|={V}$,  $|E_J|={E}$,  $k_{{\max}}={k_max}$  (quasi-tree)',
        fontsize=10
    )
    ax_map.spines['top'].set_visible(False)
    ax_map.spines['right'].set_visible(False)

    # --- right: nodes-per-shell bar chart ------------------------------------
    ks    = shell_vals
    counts = [len(shells[k]) for k in ks]
    colors = [shell_colors[k] for k in ks]

    bars = ax_bar.barh([f'$k={k}$' for k in ks], counts,
                        color=colors, edgecolor='white', linewidth=0.6, alpha=0.92)
    for bar, cnt in zip(bars, counts):
        ax_bar.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                    str(cnt), va='center', ha='left', fontsize=9)

    ax_bar.set_xlabel('Node count', fontsize=10)
    ax_bar.set_title('(b) Nodes per $k$-shell', fontsize=10)
    ax_bar.invert_yaxis()
    ax_bar.grid(True, axis='x', alpha=0.25)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.set_xlim(0, max(counts) * 1.15)

    # Annotation: quasi-tree characterisation
    pct_k1 = 100 * len(shells.get(1, [])) / V
    ax_bar.text(0.97, 0.05,
                f'{pct_k1:.0f}% nodes\nin $k$-shell 1\n(quasi-tree)',
                transform=ax_bar.transAxes, ha='right', va='bottom',
                fontsize=8.5, style='italic', color='#555555')

    fig.tight_layout(pad=1.5)

    for ext in ('pdf', 'png'):
        out = os.path.join(FIGS_OUT, f'k_core_adif.{ext}')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved: {out}')
    plt.close(fig)
    print('Done.')
