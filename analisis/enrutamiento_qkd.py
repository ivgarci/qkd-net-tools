"""
Enrutamiento consciente de QKD (key-aware routing).

Compara dos estrategias de enrutamiento entre pares de nodos:
  1. Ruta más corta en saltos (Dijkstra clásico — sin física QKD)
  2. Ruta máxima-SKR (maximizar la SKR agregada del camino)

La SKR agregada de una ruta es el mínimo de los SKR de las aristas
que la componen (cuello de botella del canal cuántico).

Genera:
  figuras/comparacion_rutas_qkd.pdf/.png
  datos/enrutamiento_qkd_bottleneck.csv — top-10 pares más limitados por SKR
"""

import os
import math
import heapq
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_CYL  = os.path.join(BASE, '..', 'datos', 'cyl')
DATA_ESP  = os.path.join(BASE, '..', 'datos', 'espana')
FIGS_OUT  = os.path.join(BASE, '..', 'figuras')
os.makedirs(FIGS_OUT, exist_ok=True)

# Importar modelo SKR (relativo al paquete protocols/)
import sys
sys.path.insert(0, os.path.join(BASE, '..'))
from protocols.skr_bb84 import skr_bb84_decoy, _haversine


# ---------------------------------------------------------------------------
# Construcción del grafo con atributos de distancia y SKR
# ---------------------------------------------------------------------------

def build_qkd_graph(adj_csv, coords_csv, coords_sep=','):
    """Carga grafo, asigna dist_km y SKR a cada arista."""
    adj = pd.read_csv(adj_csv, index_col=0)
    G = nx.from_pandas_adjacency(adj)

    if coords_csv and os.path.exists(coords_csv):
        coords_df = pd.read_csv(coords_csv, delimiter=coords_sep)
        col_pob = 'Población' if 'Población' in coords_df.columns else coords_df.columns[0]
        coords = {row[col_pob]: (row['Latitud'], row['Longitud'])
                  for _, row in coords_df.iterrows()
                  if row[col_pob] in G.nodes()}

        for u, v in G.edges():
            if u in coords and v in coords:
                lat1, lon1 = coords[u]
                lat2, lon2 = coords[v]
                dist = _haversine(lat1, lon1, lat2, lon2)
                skr  = skr_bb84_decoy(dist)
                G[u][v]['dist_km'] = dist
                G[u][v]['SKR']     = skr
            else:
                G[u][v]['dist_km'] = 45.0
                G[u][v]['SKR']     = skr_bb84_decoy(45.0)

    return G


# ---------------------------------------------------------------------------
# Algoritmo de ruta máxima-SKR (variante de Dijkstra)
# ---------------------------------------------------------------------------

def max_skr_path(G, source, target):
    """
    Ruta entre source y target que maximiza la SKR mínima del camino.
    Utiliza un heap máximo (negamos los valores para usar heapq).
    Devuelve (skr_bottleneck, path). skr_bottleneck=0 si no hay ruta.
    """
    # {node: best_min_skr_reached}
    best = {source: float('inf')}
    # heap: (-min_skr_so_far, node, path)
    heap = [(-float('inf'), source, [source])]
    visited = set()

    while heap:
        neg_skr, node, path = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)

        if node == target:
            return -neg_skr, path

        for nbr in G.neighbors(node):
            if nbr in visited:
                continue
            edge_skr = G[node][nbr].get('SKR', 0.0)
            path_skr = min(-neg_skr, edge_skr)
            if path_skr > best.get(nbr, 0.0):
                best[nbr] = path_skr
                heapq.heappush(heap, (-path_skr, nbr, path + [nbr]))

    return 0.0, []


# ---------------------------------------------------------------------------
# Comparación de rutas para un grafo
# ---------------------------------------------------------------------------

def compare_routing(G, sample_pairs=None, max_pairs=200):
    """
    Compara Dijkstra (saltos) vs max-SKR para una muestra de pares.
    Devuelve DataFrame con métricas por par.
    """
    nodes = list(G.nodes())
    if sample_pairs is None:
        # Muestra aleatoria reproducible
        rng = np.random.default_rng(42)
        idx = rng.choice(len(nodes), size=min(len(nodes), 50), replace=False)
        sample_nodes = [nodes[i] for i in idx]
        pairs = [(u, v) for i, u in enumerate(sample_nodes)
                 for v in sample_nodes[i+1:]][:max_pairs]
    else:
        pairs = sample_pairs

    rows = []
    for u, v in pairs:
        # Ruta corta en saltos
        try:
            sp_hops = nx.shortest_path(G, u, v)
            sp_hops_n = len(sp_hops) - 1
            sp_dist = sum(G[sp_hops[i]][sp_hops[i+1]].get('dist_km', 0)
                          for i in range(len(sp_hops)-1))
            sp_skr_btl = min(G[sp_hops[i]][sp_hops[i+1]].get('SKR', 0)
                             for i in range(len(sp_hops)-1)) if sp_hops_n > 0 else 0.0
        except nx.NetworkXNoPath:
            continue

        # Ruta máxima-SKR
        mqr_skr, mqr_path = max_skr_path(G, u, v)
        mqr_hops = len(mqr_path) - 1
        mqr_dist = sum(G[mqr_path[i]][mqr_path[i+1]].get('dist_km', 0)
                       for i in range(len(mqr_path)-1))

        rows.append({
            'origen': str(u),
            'destino': str(v),
            'sp_hops': sp_hops_n,
            'sp_dist_km': round(sp_dist, 1),
            'sp_skr_bottleneck': sp_skr_btl,
            'mqr_hops': mqr_hops,
            'mqr_dist_km': round(mqr_dist, 1),
            'mqr_skr_bottleneck': mqr_skr,
            'skr_gain': mqr_skr / sp_skr_btl if sp_skr_btl > 0 else np.inf,
            'hop_overhead': mqr_hops - sp_hops_n,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------

def plot_routing_comparison(df, label, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel izquierdo: SKR Dijkstra vs max-SKR
    ax = axes[0]
    valid = df[(df['sp_skr_bottleneck'] > 0) & (df['mqr_skr_bottleneck'] > 0)]
    if not valid.empty:
        ax.scatter(valid['sp_skr_bottleneck'], valid['mqr_skr_bottleneck'],
                   alpha=0.5, s=20, color='steelblue')
        lim_min = min(valid['sp_skr_bottleneck'].min(),
                      valid['mqr_skr_bottleneck'].min()) * 0.5
        lim_max = max(valid['sp_skr_bottleneck'].max(),
                      valid['mqr_skr_bottleneck'].max()) * 2
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                'k--', lw=0.8, alpha=0.5, label='Igualdad')
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel('SKR_bottleneck ruta corta (bits/pulso)')
    ax.set_ylabel('SKR_bottleneck ruta máx-SKR (bits/pulso)')
    ax.set_title(f'SKR de cuello de botella — {label}')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    # Panel derecho: distribución de skr_gain
    ax2 = axes[1]
    gains = df['skr_gain'].replace([np.inf, -np.inf], np.nan).dropna()
    if not gains.empty:
        ax2.hist(gains[gains <= 20], bins=30, color='steelblue', alpha=0.7,
                 edgecolor='white')
        ax2.axvline(1.0, color='black', lw=0.8, ls='--',
                    label='Sin mejora (ratio=1)')
        ax2.set_xlabel('Ratio SKR max-SKR / Dijkstra')
        ax2.set_ylabel('Número de pares')
        ax2.set_title('Mejora de SKR — ruta consciente vs saltos mínimos')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Enrutamiento consciente de QKD — {label}', fontsize=12, y=1.01)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f'comparacion_rutas_qkd.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Guardado: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Enrutamiento consciente de QKD — CyL")
    print("=" * 60)

    adj_csv    = os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv')
    coords_csv = os.path.join(DATA_CYL, 'cyl_1000.csv')

    if not os.path.exists(adj_csv):
        print(f"No encontrado: {adj_csv}")
        exit(1)

    G = build_qkd_graph(adj_csv, coords_csv, coords_sep=';')
    print(f"Grafo CyL: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")

    print("Comparando rutas (Dijkstra vs max-SKR)...")
    df = compare_routing(G, max_pairs=300)

    if not df.empty:
        print(f"\nPares analizados: {len(df)}")
        print(f"Mejora media SKR (ratio): {df['skr_gain'].replace([np.inf], np.nan).mean():.2f}x")
        print(f"Overhead de saltos (media): {df['hop_overhead'].mean():.2f}")

        # Top-10 pares más limitados por SKR (cuello de botella más bajo)
        bottleneck = df.nsmallest(10, 'sp_skr_bottleneck')[
            ['origen', 'destino', 'sp_dist_km', 'sp_skr_bottleneck',
             'mqr_skr_bottleneck', 'skr_gain']
        ]
        print("\nTop-10 pares más limitados (SKR_bottleneck Dijkstra):")
        print(bottleneck.to_string(index=False))

        out_csv = os.path.join(BASE, '..', 'datos', 'enrutamiento_qkd_bottleneck.csv')
        df.to_csv(out_csv, index=False)
        print(f"\nGuardado: {out_csv}")

        plot_routing_comparison(df, 'CyL', FIGS_OUT)

    print("Done.")
