#!/usr/bin/env python3
"""
delta_stability_cyl.py — Análisis de estabilidad de Δ para QKD-Communities (§8.2)

Para Δ ∈ {30,35,40,45,50,55,60,65,70} km:
  - Construye el grafo sobre los 100 nodos de la red CyL base
  - Ejecuta Girvan-Newman (k=9 cortes)
  - Calcula el alignment score A(Δ) usando la asignación nodo→provincia
    derivada del caso verificado Δ=45km (A=1 confirmado)

La asignación nodo→provincia se establece una sola vez con GN+Hungarian en
Δ=45km (resultado verificado empíricamente) y se usa como referencia fija.

Uso:
    python analisis/delta_stability_cyl.py

Salida: tabla por pantalla + datos/resultados_papers/delta_stability_cyl.csv
"""

import os
import math
import csv
import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm
from scipy.optimize import linear_sum_assignment

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_CYL = os.path.join(BASE, '..', 'datos', 'cyl')
RESULTS  = os.path.join(BASE, '..', 'datos', 'resultados_papers')
os.makedirs(RESULTS, exist_ok=True)

PROV_CAPITALS = {
    'Ávila':      (40.657, -4.700),
    'Burgos':     (42.343, -3.697),
    'León':       (42.598, -5.571),
    'Palencia':   (42.010, -4.535),
    'Salamanca':  (40.965, -5.664),
    'Segovia':    (40.948, -4.118),
    'Soria':      (41.764, -2.465),
    'Valladolid': (41.652, -4.724),
    'Zamora':     (41.503, -5.745),
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(a))


def load_coords(coords_csv):
    df = pd.read_csv(coords_csv, sep=';', encoding='utf-8-sig')
    df['Latitud']  = df['Latitud'].astype(str).str.replace(',', '.').astype(float)
    df['Longitud'] = df['Longitud'].astype(str).str.replace(',', '.').astype(float)
    return {row['Población']: (float(row['Latitud']), float(row['Longitud']))
            for _, row in df.iterrows()}


def build_graph_for_delta(nodes, coords, delta_km):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    node_list = list(nodes)
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            u, v = node_list[i], node_list[j]
            if u in coords and v in coords:
                d = haversine_km(coords[u][0], coords[u][1],
                                 coords[v][0], coords[v][1])
                if d <= delta_km:
                    G.add_edge(u, v)
    return G


def establish_province_assignment(base_adj_csv, coords, prov_capitals):
    """
    Corre GN en el grafo base (Δ=45km) con k=9 y hace matching húngaro
    para obtener el mapeo ground-truth nodo→provincia.
    Devuelve dict {node_name: province_name}.
    """
    adj = pd.read_csv(base_adj_csv, index_col=0)
    G = nx.from_pandas_adjacency(adj)

    comp_gen = nx_comm.girvan_newman(G)
    partition = None
    for _ in range(8):
        partition = tuple(next(comp_gen))
    communities = list(partition)
    print(f"  GN base (Δ=45km): k={len(communities)}, "
          f"Q={nx_comm.modularity(G, communities):.4f}, "
          f"sizes={sorted([len(c) for c in communities], reverse=True)}")

    prov_names = list(prov_capitals.keys())
    cost = []
    for comm in communities:
        lats = [coords[n][0] for n in comm if n in coords]
        lons = [coords[n][1] for n in comm if n in coords]
        clat = sum(lats) / len(lats) if lats else 41.0
        clon = sum(lons) / len(lons) if lons else -4.5
        row = [haversine_km(clat, clon, *prov_capitals[p]) for p in prov_names]
        cost.append(row)

    row_ind, col_ind = linear_sum_assignment(np.array(cost))
    comm_to_prov = {r: prov_names[c] for r, c in zip(row_ind, col_ind)}

    node_prov = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_prov[node] = comm_to_prov[i]
    return node_prov


def alignment_score(communities, node_province):
    """
    A(Δ) = fracción de nodos cuya comunidad GN es pura (100% de un único
    provincia según el mapeo ground-truth).
    A=1 ↔ cada comunidad contiene exactamente nodos de una sola provincia.
    """
    total = sum(len(c) for c in communities)
    aligned = 0
    for comm in communities:
        provs = set(node_province.get(n) for n in comm if node_province.get(n))
        if len(provs) == 1:
            aligned += len(comm)
    return aligned / total if total > 0 else 0.0


def components_alignment(G, node_province):
    """
    Para grafos desconectados: A calculado dentro de cada componente conexa.
    Devuelve (A_global, n_components, min_component_size).
    """
    components = list(nx.connected_components(G))
    total = G.number_of_nodes()
    aligned = 0
    for comp in components:
        provs = set(node_province.get(n) for n in comp if node_province.get(n))
        if len(provs) == 1:
            aligned += len(comp)
    A = aligned / total if total > 0 else 0.0
    return A, len(components), min(len(c) for c in components)


def run_gn_k9(G):
    comp_gen = nx_comm.girvan_newman(G)
    partition = None
    try:
        for _ in range(8):
            partition = tuple(next(comp_gen))
        return list(partition)
    except StopIteration:
        return list(partition) if partition else [set(G.nodes())]


if __name__ == '__main__':
    coords_cyl = load_coords(os.path.join(DATA_CYL, 'cyl_1000.csv'))

    print("Estableciendo mapeo nodo→provincia desde Δ=45km (ground truth)...")
    node_prov = establish_province_assignment(
        os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv'),
        coords_cyl, PROV_CAPITALS)
    print(f"  Nodos asignados: {len(node_prov)}/100\n")

    base_nodes = list(node_prov.keys())
    DELTAS = [30, 35, 40, 45, 50, 55, 60, 65, 70]

    header = f"{'Δ(km)':>6} | {'|E|':>5} | {'Conn':>5} | {'CC':>4} | {'k':>4} | {'Q':>7} | {'A(Δ)':>7}"
    print(header)
    print("-" * len(header))

    rows = []
    for delta in DELTAS:
        G = build_graph_for_delta(base_nodes, coords_cyl, delta)
        n_edges = G.number_of_edges()
        connected = nx.is_connected(G)

        if not connected:
            A, n_cc, min_cc = components_alignment(G, node_prov)
            print(f"{delta:>6} | {n_edges:>5} | {'No':>5} | {n_cc:>4} | {'—':>4} | {'—':>7} | {A:>7.4f}")
            rows.append({'delta': delta, 'edges': n_edges, 'connected': False,
                         'components': n_cc, 'k': None, 'Q': None, 'A': round(A, 4)})
        else:
            try:
                partition = run_gn_k9(G)
                k = len(partition)
                Q = nx_comm.modularity(G, partition)
                A = alignment_score(partition, node_prov)
                print(f"{delta:>6} | {n_edges:>5} | {'Yes':>5} | {1:>4} | {k:>4} | {Q:>7.4f} | {A:>7.4f}")
                rows.append({'delta': delta, 'edges': n_edges, 'connected': True,
                             'components': 1, 'k': k, 'Q': round(Q, 4), 'A': round(A, 4)})
            except Exception as e:
                print(f"{delta:>6} | {n_edges:>5} | {'Yes':>5} | {1:>4} | ERROR: {e}")
                rows.append({'delta': delta, 'edges': n_edges, 'connected': True,
                             'components': 1, 'k': None, 'Q': None, 'A': None})

    out_csv = os.path.join(RESULTS, 'delta_stability_cyl.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['delta','edges','connected','components','k','Q','A'])
        w.writeheader()
        w.writerows(rows)
    print(f"\nResultados guardados en: {out_csv}")
    print("\nDone.")
