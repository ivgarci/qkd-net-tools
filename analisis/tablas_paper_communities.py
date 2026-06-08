#!/usr/bin/env python3
"""
tablas_paper_communities.py — QKD-Communities (JNCA submission)
Genera los valores exactos para todas las tablas del paper que estaban marcados
como estimados o pendientes de verificación.

Ejecutar con:
    python tablas_paper_communities.py

Salida por pantalla → copiar los valores directamente al LaTeX.
Genera también resultados/tablas_communities.csv

Tablas que cubre:
    - Table 2: nodos por comunidad/provincia (CyL, GN)
    - k_max exacto para CyL y España
    - Louvain comparison (Q, nº comunidades) para CyL y España
    - Puentes inter-comunitarios confirmados (CyL y España)
    - Validación A=1 para CyL
"""

import os
import sys
import json
import csv
import math
import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm
from scipy.optimize import linear_sum_assignment

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_CYL  = os.path.join(BASE, '..', 'datos', 'cyl')
DATA_ESP  = os.path.join(BASE, '..', 'datos', 'espana')
RESULTS   = os.path.join(BASE, '..', 'datos', 'resultados_papers')
os.makedirs(RESULTS, exist_ok=True)

# ---------------------------------------------------------------------------
# Mapeo manual provincia → código INE (CPRO)
# Asignamos cada municipio del grafo CyL a su provincia usando las primeras
# letras del nombre de municipio + conocimiento geográfico.
# La forma más robusta es leer el CSV fuente que tiene coordenadas y comparar.
# ---------------------------------------------------------------------------

# CyL tiene 9 provincias. Cargamos la lista de municipios del CSV fuente
# (cyl_1000.csv) y los de la matriz de adyacencia, luego asignamos provincia
# buscando en el Nomenclátor completo.
# Como no tenemos el CSV completo con CPRO, usamos el siguiente enfoque:
# 1) Sabemos exactamente qué municipios están en cada provincia por los
#    resultados del paper (Girvan-Newman ya detectó 9 comunidades = 9 provincias).
# 2) Para la tabla, necesitamos el CONTEO exacto de nodos por comunidad,
#    que sale directamente de nx.community.girvan_newman(G).
# El nombre de provincia lo asignamos según la posición geográfica de los
# medoides (latitud/longitud del centroide de cada comunidad).

CPRO_CYL = {
    '05': 'Ávila', '09': 'Burgos', '24': 'León', '34': 'Palencia',
    '37': 'Salamanca', '40': 'Segovia', '42': 'Soria',
    '47': 'Valladolid', '49': 'Zamora'
}

# Coordenadas aproximadas de cada capital provincial (para asignar comunidades)
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
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))


def load_graph(adj_csv):
    adj = pd.read_csv(adj_csv, index_col=0)
    return nx.from_pandas_adjacency(adj)


def load_coords(coords_csv):
    """Devuelve dict {nombre_municipio: (lat, lon)} con valores float garantizados."""
    for dec in (',', '.'):
        df = pd.read_csv(coords_csv, sep=';', decimal=dec, encoding='utf-8-sig')
        if pd.api.types.is_numeric_dtype(df['Latitud']):
            return {row['Población']: (float(row['Latitud']), float(row['Longitud']))
                    for _, row in df.iterrows()}
    # fallback: coerción manual
    df = pd.read_csv(coords_csv, sep=';', encoding='utf-8-sig')
    df['Latitud']  = df['Latitud'].astype(str).str.replace(',', '.').astype(float)
    df['Longitud'] = df['Longitud'].astype(str).str.replace(',', '.').astype(float)
    return {row['Población']: (float(row['Latitud']), float(row['Longitud']))
            for _, row in df.iterrows()}


def assign_provinces_1to1(communities, coords, province_centroids):
    """
    Asigna provincias a comunidades con matching 1-a-1 (algoritmo húngaro).
    Funciona cuando len(communities) == len(province_centroids).
    Devuelve list[str] con provincia asignada a cada comunidad (mismo orden).
    """
    prov_names = list(province_centroids.keys())
    n_comm = len(communities)
    n_prov = len(prov_names)

    # Matriz de coste: distancia de cada centroide de comunidad a cada capital
    cost = []
    for comm in communities:
        lats = [coords[n][0] for n in comm if n in coords]
        lons = [coords[n][1] for n in comm if n in coords]
        if not lats:
            cost.append([1e9] * n_prov)
            continue
        clat, clon = sum(lats)/len(lats), sum(lons)/len(lons)
        row = [haversine_km(clat, clon, *province_centroids[p]) for p in prov_names]
        cost.append(row)

    if n_comm == n_prov:
        # Matching exacto 1-a-1
        import numpy as np
        row_ind, col_ind = linear_sum_assignment(np.array(cost))
        result = ['?'] * n_comm
        for r, c in zip(row_ind, col_ind):
            result[r] = prov_names[c]
        return result
    else:
        # Más comunidades que provincias: asignación greedy por cercanía (sin 1-a-1)
        return [prov_names[min(range(n_prov), key=lambda j: cost[i][j])] for i in range(n_comm)]


def run_gn(G, label, max_communities=15):
    """Ejecuta Girvan-Newman y devuelve (communities_list, Q, n_communities)."""
    print(f"\n  Ejecutando Girvan-Newman en {label} (|V|={G.number_of_nodes()}, |E|={G.number_of_edges()})...")
    comp_gen = nx_comm.girvan_newman(G)
    best_partition, best_Q = None, -1.0
    try:
        for _ in range(max_communities):
            partition = tuple(next(comp_gen))
            Q = nx_comm.modularity(G, partition)
            print(f"    k={len(partition)}: Q={Q:.4f}")
            if Q > best_Q:
                best_Q, best_partition = Q, partition
    except StopIteration:
        pass
    return list(best_partition), best_Q


def run_louvain(G, label):
    """Ejecuta Louvain con seed fija."""
    partition = nx_comm.louvain_communities(G, seed=42)
    Q = nx_comm.modularity(G, partition)
    print(f"  Louvain {label}: k={len(partition)}, Q={Q:.4f}")
    return list(partition), Q


def kcore_stats(G, label):
    core_numbers = nx.core_number(G)
    k_max = max(core_numbers.values())
    k1_count = sum(1 for v in core_numbers.values() if v == 1)
    k_max_count = sum(1 for v in core_numbers.values() if v == k_max)
    print(f"  k-core {label}: k_max={k_max} ({k_max_count} nodos), k=1: {k1_count} nodos")
    return k_max, k_max_count, k1_count


def bridges_inter_community(G, partition):
    """Devuelve la lista de puentes del grafo que son además inter-comunitarios."""
    node_to_comm = {}
    for i, comm in enumerate(partition):
        for node in comm:
            node_to_comm[node] = i
    bridges = list(nx.bridges(G))
    inter = [(u, v) for u, v in bridges if node_to_comm.get(u) != node_to_comm.get(v)]
    return bridges, inter


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == '__main__':
    rows_out = []

    # -----------------------------------------------------------------------
    # 1. CyL
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("CyL — AdjacencyMatrixNamed45.csv")
    print("=" * 60)

    G_cyl = load_graph(os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv'))
    coords_cyl = load_coords(os.path.join(DATA_CYL, 'cyl_1000.csv'))

    print(f"Grafo cargado: |V|={G_cyl.number_of_nodes()}, |E|={G_cyl.number_of_edges()}")
    print(f"Conectado: {nx.is_connected(G_cyl)}")
    print(f"Grado medio: {sum(d for _, d in G_cyl.degree())/G_cyl.number_of_nodes():.3f}")
    total_bridges_cyl = len(list(nx.bridges(G_cyl)))
    print(f"Puentes totales: {total_bridges_cyl}")

    # Girvan-Newman CyL
    gn_cyl, Q_gn_cyl = run_gn(G_cyl, 'CyL', max_communities=15)
    print(f"\n  >> GN CyL: {len(gn_cyl)} comunidades, Q={Q_gn_cyl:.4f}")

    # Asignar provincias con matching 1-a-1 (algoritmo húngaro)
    sorted_comms_cyl = sorted(gn_cyl, key=len, reverse=False)
    prov_assigned = assign_provinces_1to1(sorted_comms_cyl, coords_cyl, PROV_CAPITALS)
    print("\n  Asignación de comunidades a provincias (matching 1-a-1):")
    print(f"  {'Comunidad':>10} | {'Provincia':>12} | {'Nodos':>6} | {'% Total':>8}")
    print("  " + "-" * 46)
    community_table = []
    for i, (comm, prov) in enumerate(zip(sorted_comms_cyl, prov_assigned)):
        pct = 100*len(comm)/G_cyl.number_of_nodes()
        print(f"  C{i+1:>8} | {prov:>12} | {len(comm):>6} | {pct:>7.1f}%")
        community_table.append({'comunidad': f'C{i+1}', 'provincia': prov, 'nodos': len(comm), 'pct': round(pct,1)})
        rows_out.append({'paper': 'Communities', 'tabla': 'Table2_CyL', 'comunidad': f'C{i+1}', 'provincia': prov, 'nodos': len(comm)})
    print(f"  Total: {sum(r['nodos'] for r in community_table)} nodos")

    # Puentes inter-comunitarios
    bridges_cyl, inter_cyl = bridges_inter_community(G_cyl, gn_cyl)
    print(f"\n  Puentes totales: {len(bridges_cyl)}")
    print(f"  Puentes INTER-comunitarios: {len(inter_cyl)}")
    for u, v in inter_cyl:
        print(f"    {u} — {v}")
        rows_out.append({'paper': 'Communities', 'tabla': 'Bridges_CyL', 'nodo_u': u, 'nodo_v': v})

    # Louvain CyL
    louvain_cyl, Q_louvain_cyl = run_louvain(G_cyl, 'CyL')
    rows_out.append({'paper': 'Communities', 'tabla': 'Louvain_CyL', 'k': len(louvain_cyl), 'Q': round(Q_louvain_cyl, 4)})

    # k-core CyL
    kmax_cyl, kmax_count_cyl, k1_count_cyl = kcore_stats(G_cyl, 'CyL')
    rows_out.append({'paper': 'Communities', 'tabla': 'Kcore_CyL', 'k_max': kmax_cyl, 'k_max_count': kmax_count_cyl, 'k1_count': k1_count_cyl})

    # -----------------------------------------------------------------------
    # 2. España
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("España — AdjacencyMatrixNamed45.csv")
    print("=" * 60)

    G_esp = load_graph(os.path.join(DATA_ESP, 'AdjacencyMatrixNamed45.csv'))
    print(f"Grafo cargado: |V|={G_esp.number_of_nodes()}, |E|={G_esp.number_of_edges()}")
    print(f"Conectado: {nx.is_connected(G_esp)}")
    print(f"Grado medio: {sum(d for _, d in G_esp.degree())/G_esp.number_of_nodes():.3f}")
    total_bridges_esp = len(list(nx.bridges(G_esp)))
    print(f"Puentes totales: {total_bridges_esp}")

    # Girvan-Newman España (puede tardar varios minutos)
    print("\n  NOTA: GN en España (950 nodos) puede tardar 5-15 min...")
    gn_esp, Q_gn_esp = run_gn(G_esp, 'España', max_communities=20)
    print(f"\n  >> GN España: {len(gn_esp)} comunidades, Q={Q_gn_esp:.4f}")
    # Tamaños de comunidades
    print("\n  Tamaño de cada comunidad España (GN):")
    for i, comm in enumerate(sorted(gn_esp, key=len, reverse=True)):
        print(f"    C{i+1}: {len(comm)} nodos")
        rows_out.append({'paper': 'Communities', 'tabla': 'Table3_ESP', 'comunidad': f'S{i+1}', 'nodos': len(comm)})

    # Puentes inter-comunitarios España
    bridges_esp, inter_esp = bridges_inter_community(G_esp, gn_esp)
    print(f"\n  Puentes totales España: {len(bridges_esp)}")
    print(f"  Puentes INTER-comunitarios: {len(inter_esp)}")
    for u, v in inter_esp:
        print(f"    {u} — {v}")
        rows_out.append({'paper': 'Communities', 'tabla': 'Bridges_ESP', 'nodo_u': u, 'nodo_v': v})

    # Louvain España
    louvain_esp, Q_louvain_esp = run_louvain(G_esp, 'España')
    rows_out.append({'paper': 'Communities', 'tabla': 'Louvain_ESP', 'k': len(louvain_esp), 'Q': round(Q_louvain_esp, 4)})

    # k-core España
    kmax_esp, kmax_count_esp, k1_count_esp = kcore_stats(G_esp, 'España')
    rows_out.append({'paper': 'Communities', 'tabla': 'Kcore_ESP', 'k_max': kmax_esp, 'k_max_count': kmax_count_esp, 'k1_count': k1_count_esp})

    # -----------------------------------------------------------------------
    # 3. Resumen para el paper
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESUMEN PARA EL PAPER")
    print("=" * 60)
    print(f"\nCyL:")
    print(f"  GN: k={len(gn_cyl)} comunidades, Q={Q_gn_cyl:.4f}")
    print(f"  Louvain: k={len(louvain_cyl)}, Q={Q_louvain_cyl:.4f}")
    print(f"  k_max = {kmax_cyl}")
    print(f"  Puentes inter-comunitarios: {len(inter_cyl)}")
    print(f"\nEspaña:")
    print(f"  GN: k={len(gn_esp)} comunidades, Q={Q_gn_esp:.4f}")
    print(f"  Louvain: k={len(louvain_esp)}, Q={Q_louvain_esp:.4f}")
    print(f"  k_max = {kmax_esp}")
    print(f"  Puentes inter-comunitarios: {len(inter_esp)}")

    # -----------------------------------------------------------------------
    # 4. Guardar CSV
    # -----------------------------------------------------------------------
    out_csv = os.path.join(RESULTS, 'tablas_communities.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if rows_out:
        all_keys = set()
        for r in rows_out:
            all_keys.update(r.keys())
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=sorted(all_keys))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\nResultados guardados en: {out_csv}")

    print("\nDone.")
