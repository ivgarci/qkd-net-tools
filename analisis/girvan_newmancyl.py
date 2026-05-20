"""
Detección de comunidades Girvan-Newman para CyL (y opcionalmente España).
k (número de comunidades) se puede pasar como argumento de línea de comandos.

Uso:
    python girvan_newmancyl.py         # k=8 por defecto, caso CyL
    python girvan_newmancyl.py 10      # k=10, caso CyL
"""

import os
import sys
import pandas as pd
import networkx as nx
from networkx.algorithms.community import girvan_newman
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_CYL = os.path.join(BASE, '..', 'datos', 'cyl')
FIGS_CYL = os.path.join(BASE, '..', 'figuras', 'cyl')

os.makedirs(FIGS_CYL, exist_ok=True)

COLORS = [
    'skyblue', 'lightgreen', 'coral', 'gold', 'lightpink',
    'lightgrey', 'lightblue', 'orange', 'purple', 'brown', 'cyan', 'lime',
]


def run_girvan_newman(adj_csv, coords_csv, k, out_dir, caso_name, sep=','):
    adj = pd.read_csv(adj_csv, index_col=0)
    G = nx.from_pandas_adjacency(adj)

    coords_df = pd.read_csv(coords_csv, delimiter=sep)
    col_pob = 'Población' if 'Población' in coords_df.columns else coords_df.columns[0]
    node_positions = {
        row[col_pob]: (row['Longitud'], row['Latitud'])
        for _, row in coords_df.iterrows()
        if row[col_pob] in G.nodes()
    }

    communities_generator = girvan_newman(G)
    for _ in range(k - 1):
        next(communities_generator)
    top_communities = sorted(map(sorted, next(communities_generator)))

    color_map = []
    for node in G:
        for idx, community in enumerate(top_communities):
            if node in community:
                color_map.append(COLORS[idx % len(COLORS)])
                break

    fig = plt.figure(figsize=(12, 12))
    nx.draw(G, pos=node_positions, with_labels=True, node_size=500,
            node_color=color_map, font_size=8, font_weight='bold', edge_color='gray')
    plt.title(f"Comunidades Girvan-Newman — {caso_name} ({k} comunidades)")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")

    base_name = f"girvan_newman_{caso_name}_{k}"
    for ext in ('png', 'pdf', 'svg'):
        path = os.path.join(out_dir, f"{base_name}.{ext}")
        plt.savefig(path, format=ext, dpi=300 if ext == 'png' else None,
                    bbox_inches='tight')
        print(f"Guardado: {path}")
    plt.close(fig)

    communities_df = pd.DataFrame({
        'Community': range(len(top_communities)),
        'Nodes': top_communities,
        'Size': [len(c) for c in top_communities],
    })
    print(f"\nComunidades Girvan-Newman ({caso_name}, k={k}):")
    print(communities_df.to_string(index=False))
    return top_communities


if __name__ == '__main__':
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(f"Ejecutando Girvan-Newman con k={k} comunidades — caso CyL")
    run_girvan_newman(
        adj_csv=os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv'),
        coords_csv=os.path.join(DATA_CYL, 'cyl_1000.csv'),
        k=k,
        out_dir=FIGS_CYL,
        caso_name='cyl',
        sep=';',
    )
