"""
Construcción de los tres grafos QKD con atributo 'skr' por arista.

Redes:
  - cyl    : 100 nodos, 254 aristas  (AdjacencyMatrixNamed45 + skr_per_link.csv)
  - espana : 950 nodos, 5681 aristas (AdjacencyMatrixNamed45 + skr_per_link.csv)
  - adif   : 485 nodos, 633 aristas  (grafo de junctions: contracción de
             cadenas de grado 2 de la LCC de la red ADIF; SKR = BB84+decoy
             sobre la distancia acumulada del tramo contraído, suelo 1e-15
             para tramos fuera de rango)

El SKR de CyL/España procede de datos/skr_per_link.csv (generado con
protocols/skr_bb84.skr_bb84_decoy, η_det = 0.10). Para ADIF se evalúa la
misma función SKR(d) sobre dist_km de la arista contraída.
"""

import os
import sys

import pandas as pd
import networkx as nx

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, '..'))
sys.path.insert(0, ROOT)

from protocols.skr_bb84 import skr_bb84_decoy  # noqa: E402

DATA = os.path.join(ROOT, 'datos')
SKR_FLOOR = 1e-15  # mismo suelo que analisis/enrutamiento_adif_completo.py


def _build_from_adjacency(adj_csv, caso_skr):
    """Grafo desde matriz de adyacencia nominal + SKR de skr_per_link.csv."""
    adj = pd.read_csv(adj_csv, index_col=0)
    G = nx.from_pandas_adjacency(adj)

    skr_df = pd.read_csv(os.path.join(DATA, 'skr_per_link.csv'))
    skr_df = skr_df[skr_df['caso'] == caso_skr]
    lookup = {}
    for _, r in skr_df.iterrows():
        lookup[frozenset((r['nodo_u'], r['nodo_v']))] = (
            float(r['SKR_bits_pulso']), float(r['dist_km']))

    missing = []
    for u, v in G.edges():
        key = frozenset((u, v))
        if key in lookup:
            skr, dist = lookup[key]
            G[u][v]['skr'] = max(skr, SKR_FLOOR)
            G[u][v]['dist_km'] = dist
        else:
            missing.append((u, v))
    if missing:
        raise ValueError(
            f"{caso_skr}: {len(missing)} aristas sin SKR en skr_per_link.csv "
            f"(p.ej. {missing[:3]})")

    nx.set_node_attributes(G, {n: n for n in G.nodes()}, 'nombre')
    return G


def _build_adif_junctions():
    """
    Grafo de junctions ADIF (485 nodos, 633 aristas).
    Réplica de adif/analisis_adif_junctions.py:
      1. nodos con conectado == 'SI'
      2. aristas deduplicadas (tabla bidireccional)
      3. LCC del grafo completo
      4. contracción de cadenas de grado 2 (dist_km acumulada;
         caminos paralelos → distancia mínima)
      5. LCC del grafo de junctions
    SKR por arista = skr_bb84_decoy(dist_km acumulada), suelo 1e-15.
    """
    nodes_df = pd.read_csv(os.path.join(DATA, 'adif', 'nodos_red_adif.csv'),
                           dtype={'cod': str}, quotechar='"',
                           on_bad_lines='skip')
    adj_df = pd.read_csv(os.path.join(DATA, 'adif', 'adyacencia_red_adif.csv'),
                         dtype={'cod': str, 'vecino_cod': str},
                         quotechar='"', on_bad_lines='skip')

    G = nx.Graph()
    for _, r in nodes_df[nodes_df['conectado'] == 'SI'].iterrows():
        G.add_node(str(r['cod']), nombre=str(r['nombre']))

    seen = set()
    for _, r in adj_df.iterrows():
        u, v = str(r['cod']), str(r['vecino_cod'])
        key = frozenset((u, v))
        if key in seen:
            continue
        seen.add(key)
        if not (G.has_node(u) and G.has_node(v)):
            continue
        try:
            d = float(r['dist_km'])
        except (ValueError, TypeError):
            continue
        G.add_edge(u, v, dist_km=d)

    lcc = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc).copy()

    keep = {n for n in G.nodes() if G.degree(n) != 2}
    J = nx.Graph()
    for n in keep:
        J.add_node(n, **G.nodes[n])

    visited = set()
    for start in keep:
        for nbr in list(G.neighbors(start)):
            ek = frozenset((start, nbr))
            if ek in visited:
                continue
            visited.add(ek)
            acc = G[start][nbr].get('dist_km', 0.0) or 0.0
            prev, cur = start, nbr
            while cur not in keep:
                nbs = list(G.neighbors(cur))
                nxt = nbs[0] if nbs[1] == prev else nbs[1]
                acc += G[cur][nxt].get('dist_km', 0.0) or 0.0
                visited.add(frozenset((cur, nxt)))
                prev, cur = cur, nxt
            if cur == start:
                continue
            if not J.has_edge(start, cur) or acc < J[start][cur]['dist_km']:
                J.add_edge(start, cur, dist_km=acc)

    lcc_j = max(nx.connected_components(J), key=len)
    J = J.subgraph(lcc_j).copy()

    for u, v in J.edges():
        J[u][v]['skr'] = max(skr_bb84_decoy(J[u][v]['dist_km']), SKR_FLOOR)

    return J


def build_graph(red):
    """red ∈ {'cyl', 'espana', 'adif'} → nx.Graph con atributo 'skr'."""
    if red == 'cyl':
        return _build_from_adjacency(
            os.path.join(DATA, 'cyl', 'AdjacencyMatrixNamed45.csv'), 'CyL')
    if red == 'espana':
        return _build_from_adjacency(
            os.path.join(DATA, 'espana', 'AdjacencyMatrixNamed45.csv'),
            'España')
    if red == 'adif':
        return _build_adif_junctions()
    raise ValueError(f"Red desconocida: {red}")


if __name__ == '__main__':
    for red in ('cyl', 'espana', 'adif'):
        G = build_graph(red)
        skrs = [d['skr'] for _, _, d in G.edges(data=True)]
        print(f"{red}: |V|={G.number_of_nodes()} |E|={G.number_of_edges()} "
              f"SKR[{min(skrs):.3e}, {max(skrs):.3e}]")
