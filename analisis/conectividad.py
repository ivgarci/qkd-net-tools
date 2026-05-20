"""
Visualización de la red con layout geográfico o spring layout.
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


def plot_connectivity(G: nx.Graph, out_dir: str) -> None:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
    })

    degree_map = dict(G.degree())
    node_sizes = [50 + 20 * degree_map[n] for n in G.nodes()]
    node_colors = [degree_map[n] for n in G.nodes()]

    pos = nx.spring_layout(G, seed=42, k=1.5)

    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, edge_color='gray', width=0.8)
    sc = nx.draw_networkx_nodes(G, pos, ax=ax,
                                node_size=node_sizes,
                                node_color=node_colors,
                                cmap=plt.cm.YlOrRd,
                                alpha=0.85)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=5)
    plt.colorbar(sc, ax=ax, label='Grado del nodo', shrink=0.7)
    ax.set_title(f'Conectividad de la red CyL — |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}')
    ax.axis('off')

    fig.tight_layout()

    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f'conectividad_red.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Guardado: {path}")

    plt.close(fig)

    # Métricas básicas de conectividad
    components = list(nx.connected_components(G))
    print(f"Componentes conexas: {len(components)}")
    print(f"Puntos de articulación: {len(list(nx.articulation_points(G)))}")
    print(f"Puentes: {len(list(nx.bridges(G)))}")
    if nx.is_connected(G):
        print(f"Diámetro: {nx.diameter(G)}")
        print(f"Longitud media de camino: {nx.average_shortest_path_length(G):.4f}")


if __name__ == '__main__':
    adj_csv = os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv')
    G = load_graph(adj_csv)
    print(f"Grafo cargado: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    plot_connectivity(G, FIGS_CYL)
    print("Listo.")
