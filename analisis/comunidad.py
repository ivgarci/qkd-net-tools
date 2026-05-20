"""
Detección de comunidades por el método de Louvain (networkx ≥3.0).
Opera sobre la red CyL (AdjacencyMatrixNamed45.csv).
"""

import os
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_CYL = os.path.join(BASE, '..', 'datos', 'cyl')
FIGS_CYL = os.path.join(BASE, '..', 'figuras', 'cyl')

os.makedirs(FIGS_CYL, exist_ok=True)


def load_graph(adj_csv: str) -> nx.Graph:
    adj = pd.read_csv(adj_csv, index_col=0)
    return nx.from_pandas_adjacency(adj)


def plot_louvain_communities(G: nx.Graph, out_dir: str) -> int:
    communities = nx_comm.louvain_communities(G, seed=42)
    partition = {}
    for comm_id, nodes in enumerate(communities):
        for node in nodes:
            partition[node] = comm_id

    num_communities = len(communities)
    modularity = nx_comm.modularity(G, communities)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
    })

    cmap = matplotlib.colormaps['tab20'].resampled(num_communities)
    node_colors = [cmap(partition[node]) for node in G.nodes()]

    pos = nx.spring_layout(G, seed=42, k=1.5)

    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color='lightgray', width=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=80,
                           alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=5)
    ax.set_title(
        f'Estructura de comunidades Louvain — CyL\n'
        f'{num_communities} comunidades, modularidad Q={modularity:.3f}'
    )
    ax.axis('off')

    fig.tight_layout()

    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f'comunidades_louvain.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Guardado: {path}")

    plt.close(fig)
    return num_communities


if __name__ == '__main__':
    adj_csv = os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv')
    G = load_graph(adj_csv)
    print(f"Grafo cargado: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    n = plot_louvain_communities(G, FIGS_CYL)
    print(f"Comunidades detectadas (Louvain): {n}")
