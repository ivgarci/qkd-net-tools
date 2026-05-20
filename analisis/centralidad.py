"""
Visualización de las top-10 nodos por centralidad de grado, intermediación y cercanía.
Opera sobre la red CyL (AdjacencyMatrixNamed45.csv).
"""

import os
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


def plot_top_centrality(G: nx.Graph, out_dir: str, top_n: int = 10) -> None:
    degree_c = nx.degree_centrality(G)
    betweenness_c = nx.betweenness_centrality(G)
    closeness_c = nx.closeness_centrality(G)

    df = pd.DataFrame({
        'Centralidad de grado': degree_c,
        'Centralidad de intermediación': betweenness_c,
        'Centralidad de cercanía': closeness_c,
    })

    colors = ['steelblue', 'firebrick', 'seagreen']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 9,
    })

    for ax, col, color in zip(axes, df.columns, colors):
        top = df[col].sort_values(ascending=False).head(top_n)
        top.plot(kind='bar', ax=ax, color=color, edgecolor='white', linewidth=0.5)
        ax.set_title(col)
        ax.set_xlabel('Nodo')
        ax.set_ylabel('Valor de centralidad')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()

    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f'centralidad_top10.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Guardado: {path}")

    plt.close(fig)


if __name__ == '__main__':
    adj_csv = os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv')
    G = load_graph(adj_csv)
    print(f"Grafo cargado: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    plot_top_centrality(G, FIGS_CYL)
    print("Listo.")
