"""
Distribución del coeficiente de agrupamiento (clustering) por nodo.
Opera sobre la red CyL (AdjacencyMatrixNamed45.csv).
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_CYL = os.path.join(BASE, '..', 'datos', 'cyl')
FIGS_CYL = os.path.join(BASE, '..', 'figuras', 'cyl')

os.makedirs(FIGS_CYL, exist_ok=True)


def load_graph(adj_csv: str) -> nx.Graph:
    adj = pd.read_csv(adj_csv, index_col=0)
    return nx.from_pandas_adjacency(adj)


def plot_clustering_distribution(G: nx.Graph, out_dir: str) -> float:
    avg_cc = nx.average_clustering(G)
    node_cc = nx.clustering(G)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(list(node_cc.values()), bins=np.linspace(0, 1, 20),
            color='steelblue', edgecolor='black', alpha=0.8)
    ax.axvline(avg_cc, color='firebrick', lw=1.5, ls='--',
               label=f'Media = {avg_cc:.3f}')
    ax.set_title('Distribución del coeficiente de agrupamiento')
    ax.set_xlabel('Coeficiente de agrupamiento')
    ax.set_ylabel('Número de nodos')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()

    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f'coeficiente_clustering.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Guardado: {path}")

    plt.close(fig)
    return avg_cc


if __name__ == '__main__':
    adj_csv = os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv')
    G = load_graph(adj_csv)
    print(f"Grafo cargado: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    avg = plot_clustering_distribution(G, FIGS_CYL)
    print(f"Coeficiente de agrupamiento medio: {avg:.4f}")
