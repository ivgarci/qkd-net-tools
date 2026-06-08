"""
Betweenness centrality distribution for the ADIF junction graph.
Generates: centralidades_intermediacion.pdf/.png  (English labels)
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_ADIF = os.path.join(BASE, '..', 'datos', 'adif')

# Output directory: pass as first argument or default to ../../articulos/ADIF-QKD/Figures
if len(sys.argv) > 1:
    FIGS_OUT = sys.argv[1]
else:
    FIGS_OUT = os.path.join(BASE, '..', '..', '..', 'articulos', 'ADIF-QKD', 'Figures')
os.makedirs(FIGS_OUT, exist_ok=True)


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
        G_full.add_node(str(row['cod']), lat=float(row['lat']), lon=float(row['lon']))
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
    # Degree-2 contraction → junction graph
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


if __name__ == '__main__':
    print('Loading ADIF junction graph...')
    J = load_adif_junction_graph()
    print(f'  |V|={J.number_of_nodes()}, |E|={J.number_of_edges()}')

    print('Computing betweenness centrality...')
    cb = nx.betweenness_centrality(J, normalized=True)
    vals = list(cb.values())

    cb_max_node = max(cb, key=cb.get)
    v_max = cb[cb_max_node]
    label_max = str(cb_max_node).replace('BIF. ', 'Bif. ').replace('BIF.', 'Bif.')
    print(f'  C_B max = {v_max:.4f}  ({label_max})')

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.hist(vals, bins=40, color='#2166ac', alpha=0.75,
            edgecolor='white', linewidth=0.5)
    ax.axvline(v_max, color='black', lw=1.0, ls='--', alpha=0.7)
    ax.text(v_max + 0.004, ax.get_ylim()[1] * 0.65,
            f'max={v_max:.3f}\n({label_max[:22]})',
            fontsize=7.5, va='top')

    ax.set_xlabel('Normalised betweenness centrality $C_B(v)$', fontsize=10)
    ax.set_ylabel('Node count', fontsize=10)
    ax.set_title(r'ADIF junction graph ($|V_J|=485$, $|E_J|=633$)', fontsize=10)
    ax.grid(True, which='major', alpha=0.25)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        out = os.path.join(FIGS_OUT, f'centralidades_intermediacion.{ext}')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved: {out}')
    plt.close(fig)
    print('Done.')
