"""
Análisis estructural de la red ADIF de fibra oscura — grafo de junctions.

Metodología:
1. Cargar grafo físico completo desde adyacencia_red_adif.csv
2. Extraer la componente gigante conexa (LCC)
3. Construir el grafo de junctions contrayendo nodos de grado 2
4. Aplicar el protocolo de validación post-diseño (Cap. 4 de la tesis)
5. Simular resiliencia: fallos aleatorios y ataques dirigidos (grado e intermediación)

Umbral físico: Δ_eff = 50 km (los dos tramos de 47,4 km y 47,9 km de la línea
AVE 982 superan el umbral de referencia Δ = 45 km en menos del 7%, dentro del
margen de variabilidad del modelo físico simplificado).
"""

import os
import pandas as pd
import networkx as nx
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_ADIF = os.path.join(BASE, '..', 'datos', 'adif')

DELTA_REF   = 45.0   # km — umbral de referencia (modelo simplificado)
DELTA_EFF   = 50.0   # km — umbral operativo adoptado para esta red
SEED        = 42
R_RANDOM    = 1000   # realizaciones fallos aleatorios
P0          = 0.13   # fracción eliminada en fallos aleatorios
P_STAR_THR  = 0.50   # umbral S(p) para definir p*

random.seed(SEED)
np.random.seed(SEED)

# ── 1. Carga de datos ──────────────────────────────────────────────────────────

print("Cargando datos...")
nodes_df = pd.read_csv(
    os.path.join(DATA_ADIF, 'nodos_red_adif.csv'),
    quotechar='"', on_bad_lines='skip'
)
adj_df = pd.read_csv(
    os.path.join(DATA_ADIF, 'adyacencia_red_adif.csv'),
    quotechar='"', on_bad_lines='skip'
)

print(f"  Nodos cargados:  {len(nodes_df)}")
print(f"  Aristas cargadas: {len(adj_df)}")

# ── 2. Construir grafo completo ────────────────────────────────────────────────

print("\nConstruyendo grafo completo...")
G_full = nx.Graph()

# Nodos conectados únicamente
connected = nodes_df[nodes_df['conectado'] == 'SI'].copy()
for _, row in connected.iterrows():
    G_full.add_node(
        str(row['cod']),
        nombre=str(row['nombre']),
        categoria=str(row['categoria']),
        tipo_dep=str(row['tipo_dep']),
        lat=float(row['lat']),
        lon=float(row['lon'])
    )

# Aristas — la tabla es bidireccional, usar frozenset para deduplicar
seen = set()
long_edges = []
for _, row in adj_df.iterrows():
    u, v = str(row['cod']), str(row['vecino_cod'])
    key = frozenset([u, v])
    if key in seen:
        continue
    seen.add(key)
    if G_full.has_node(u) and G_full.has_node(v):
        try:
            d = float(row['dist_km'])
        except (ValueError, TypeError):
            continue
        G_full.add_edge(u, v, dist_km=d, tipo_red=str(row.get('tipo_red', '')))
        if d > DELTA_REF:
            long_edges.append((u, G_full.nodes[u]['nombre'],
                               v, G_full.nodes[v]['nombre'], d))

print(f"  Nodos en grafo: {G_full.number_of_nodes()}")
print(f"  Aristas en grafo: {G_full.number_of_edges()}")
print(f"  Aristas > {DELTA_REF} km: {len(long_edges)}")
for u, nu, v, nv, d in long_edges:
    print(f"    {nu} — {nv}: {d:.1f} km")

# ── 3. Componente gigante conexa (LCC) ────────────────────────────────────────

print("\nExtrayendo LCC...")
components = sorted(nx.connected_components(G_full), key=len, reverse=True)
print(f"  Componentes conexas: {len(components)}")
print(f"  Tamaños (top 5): {[len(c) for c in components[:5]]}")

lcc_nodes = components[0]
G_lcc = G_full.subgraph(lcc_nodes).copy()
print(f"  LCC: {G_lcc.number_of_nodes()} nodos, {G_lcc.number_of_edges()} aristas")

# ── 4. Grafo de junctions (contracción de nodos de grado 2) ───────────────────

print("\nConstruyendo grafo de junctions (grado ≥ 3 y grado = 1)...")

def build_junction_graph(G):
    """Contrae las cadenas de nodos de grado 2.
    Conserva nodos con grado ≥ 3 (bifurcaciones/junctions) y
    grado = 1 (extremos). El peso de cada arista resultante es la
    suma de dist_km a lo largo del camino contraído.
    """
    # Identificar nodos que se conservan
    keep = {n for n in G.nodes() if G.degree(n) != 2}

    J = nx.Graph()
    for n in keep:
        J.add_node(n, **G.nodes[n])

    visited_edges = set()

    for start in keep:
        for nbr in list(G.neighbors(start)):
            ek = frozenset([start, nbr])
            if ek in visited_edges:
                continue
            visited_edges.add(ek)

            # Recorrer la cadena desde start → nbr hasta el siguiente nodo keep
            acc_dist = G[start][nbr].get('dist_km', 0.0) or 0.0
            prev, cur = start, nbr

            while cur not in keep:
                neighbors = list(G.neighbors(cur))
                nxt = neighbors[0] if neighbors[1] == prev else neighbors[1]
                d = G[cur][nxt].get('dist_km', 0.0) or 0.0
                acc_dist += d
                ek2 = frozenset([cur, nxt])
                visited_edges.add(ek2)
                prev, cur = cur, nxt

            if cur == start:
                continue  # bucle, ignorar

            # cur es otro nodo keep
            if not J.has_edge(start, cur):
                J.add_edge(start, cur, dist_km=acc_dist)
            else:
                # mantener el camino más corto si hay rutas paralelas
                if acc_dist < J[start][cur]['dist_km']:
                    J[start][cur]['dist_km'] = acc_dist

    return J

J = build_junction_graph(G_lcc)
print(f"  Grafo de junctions: {J.number_of_nodes()} nodos, {J.number_of_edges()} aristas")

# Verificar componentes del grafo de junctions
junc_components = list(nx.connected_components(J))
print(f"  Componentes en grafo junctions: {len(junc_components)}")
if len(junc_components) > 1:
    print(f"  Tamaños: {sorted([len(c) for c in junc_components], reverse=True)[:10]}")
    # Usar la mayor componente
    J = J.subgraph(max(junc_components, key=len)).copy()
    print(f"  LCC del grafo junctions: {J.number_of_nodes()} nodos, {J.number_of_edges()} aristas")

# Distribución de grados en grafo junctions
degrees = [d for _, d in J.degree()]
print(f"  Grado: min={min(degrees)}, max={max(degrees)}, medio={np.mean(degrees):.2f}")

# Distribución de distancias de aristas en grafo junctions
dists = [d['dist_km'] for _, _, d in J.edges(data=True) if d.get('dist_km') is not None]
print(f"  Dist. aristas (km): min={min(dists):.1f}, max={max(dists):.1f}, "
      f"media={np.mean(dists):.1f}, p90={np.percentile(dists, 90):.1f}")
long_j = [(u, J.nodes[u]['nombre'], v, J.nodes[v]['nombre'], J[u][v]['dist_km'])
          for u, v in J.edges() if J[u][v].get('dist_km', 0) > DELTA_REF]
print(f"  Aristas > {DELTA_REF} km en junctions: {len(long_j)}")
for u, nu, v, nv, d in long_j:
    print(f"    {nu} — {nv}: {d:.1f} km")

# ── 5. Métricas estructurales ──────────────────────────────────────────────────

print("\n=== MÉTRICAS ESTRUCTURALES ===")

V = J.number_of_nodes()
E = J.number_of_edges()
rho = 2 * E / (V * (V - 1))
print(f"|V| = {V}")
print(f"|E| = {E}")
print(f"Densidad ρ = {rho:.6f}")
print(f"Grado medio k̄ = {2*E/V:.3f}")

# Componentes
kappa = nx.number_connected_components(J)
lcc_size = len(max(nx.connected_components(J), key=len))
S = lcc_size / V
print(f"κ(G) = {kappa}")
print(f"S(G) = {S:.4f}  (componente gigante: {lcc_size} nodos)")

# Diámetro y longitud media de camino (sin pesos para comparar con CyL/España)
print("\nCalculando diámetro y longitud media (sin pesos)...")
diam = nx.diameter(J)
ell = nx.average_shortest_path_length(J)
print(f"Diámetro = {diam}")
print(f"Longitud media de camino ℓ = {ell:.3f} saltos")

# Coeficiente de clustering
cc_mean = nx.average_clustering(J)
print(f"Clustering medio C̄△ = {cc_mean:.4f}")

# Puentes y puntos de articulación
print("\nCalculando puentes y puntos de articulación...")
bridges = list(nx.bridges(J))
articulations = list(nx.articulation_points(J))
print(f"Puentes: {len(bridges)}")
print(f"Puntos de articulación: {len(articulations)}")

# ── 6. Centralidades ──────────────────────────────────────────────────────────

print("\nCalculando centralidades...")
cd = nx.degree_centrality(J)
cc_cent = nx.closeness_centrality(J)
cb = nx.betweenness_centrality(J, normalized=True)

# Top-5 por intermediación
top5_cb = sorted(cb.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop-10 nodos por intermediación C_B:")
print(f"{'Nodo':<35} {'Grado':>6} {'C_D':>8} {'C_B':>10} {'C̄△':>8}")
print("-" * 72)
for nid, cb_val in top5_cb:
    name = J.nodes[nid].get('nombre', nid)[:33]
    deg = J.degree(nid)
    cd_val = cd[nid]
    cc_val = nx.clustering(J, nid)
    print(f"{name:<35} {deg:>6} {cd_val:>8.4f} {cb_val:>10.4f} {cc_val:>8.4f}")

cb_vals = list(cb.values())
print(f"\nC_B: media={np.mean(cb_vals):.4f}, máx={max(cb_vals):.4f}, "
      f"mediana={np.median(cb_vals):.4f}")

# ── 7. Detección de comunidades (Girvan-Newman, primeros niveles) ──────────────

print("\nDetectando comunidades (Girvan-Newman)...")
from networkx.algorithms.community import girvan_newman
gn = girvan_newman(J)
# Avanzar hasta que tengamos un número razonable de comunidades
n_target = 12  # objetivo inicial
communities = None
for comm in gn:
    communities = comm
    if len(communities) >= n_target:
        break

n_comm = len(communities)
sizes = sorted([len(c) for c in communities], reverse=True)
print(f"Comunidades detectadas: {n_comm}")
print(f"Tamaños: {sizes[:15]}")

# ── 8. Resiliencia — fallos aleatorios ────────────────────────────────────────

print(f"\nSimulando fallos aleatorios (p₀={P0}, R={R_RANDOM})...")
nodes_list = list(J.nodes())
n_remove = int(np.floor(P0 * V))
S_random = []
kappa_random = []
connected_count = 0

for _ in range(R_RANDOM):
    removed = random.sample(nodes_list, n_remove)
    G_r = J.copy()
    G_r.remove_nodes_from(removed)
    comps = list(nx.connected_components(G_r))
    if len(comps) == 0:
        S_val = 0.0
    else:
        lcc_r = max(comps, key=len)
        S_val = len(lcc_r) / V
    S_random.append(S_val)
    kappa_r = nx.number_connected_components(G_r)
    kappa_random.append(kappa_r)
    if kappa_r == 1:
        connected_count += 1

print(f"  S̄ = {np.mean(S_random):.4f}  σ = {np.std(S_random):.4f}  "
      f"min = {min(S_random):.4f}")
print(f"  κ=1 en {connected_count/R_RANDOM*100:.1f}% de las realizaciones")

# ── 9. Resiliencia — ataques dirigidos por grado ──────────────────────────────

print("\nSimulando ataques dirigidos por grado...")
p_values = [p/100 for p in range(0, 50)]
S_degree = []
nodes_sorted_degree = sorted(J.nodes(), key=lambda n: J.degree(n), reverse=True)

for p in p_values:
    n_rem = int(np.floor(p * V))
    removed_set = set(nodes_sorted_degree[:n_rem])
    G_a = J.copy()
    G_a.remove_nodes_from(removed_set)
    comps = list(nx.connected_components(G_a))
    if len(comps) == 0:
        S_val = 0.0
    else:
        lcc_a = max(comps, key=len)
        S_val = len(lcc_a) / V
    S_degree.append(S_val)

# p* por grado
p_star_degree = next((p_values[i] for i, s in enumerate(S_degree) if s < P_STAR_THR), None)
first_frag_degree = next((p_values[i] for i, s in enumerate(S_degree) if s < 1.0), None)
print(f"  Primera fragmentación (grado): p = {first_frag_degree*100:.0f}%")
print(f"  p* (grado): p* = {p_star_degree*100:.0f}%" if p_star_degree else "  p* > 49%")

# ── 10. Resiliencia — ataques dirigidos por intermediación ────────────────────

print("\nSimulando ataques dirigidos por intermediación...")
nodes_sorted_cb = sorted(J.nodes(), key=lambda n: cb[n], reverse=True)
S_cb = []

for p in p_values:
    n_rem = int(np.floor(p * V))
    removed_set = set(nodes_sorted_cb[:n_rem])
    G_a = J.copy()
    G_a.remove_nodes_from(removed_set)
    comps = list(nx.connected_components(G_a))
    if len(comps) == 0:
        S_val = 0.0
    else:
        lcc_a = max(comps, key=len)
        S_val = len(lcc_a) / V
    S_cb.append(S_val)

p_star_cb = next((p_values[i] for i, s in enumerate(S_cb) if s < P_STAR_THR), None)
first_frag_cb = next((p_values[i] for i, s in enumerate(S_cb) if s < 1.0), None)
print(f"  Primera fragmentación (C_B):   p = {first_frag_cb*100:.0f}%")
print(f"  p* (C_B): p* = {p_star_cb*100:.0f}%" if p_star_cb else "  p* > 49%")

# ── 11. Resumen final ─────────────────────────────────────────────────────────

print("\n" + "="*60)
print("RESUMEN — RED ADIF (GRAFO DE JUNCTIONS)")
print("="*60)
print(f"|V|                          {V}")
print(f"|E|                          {E}")
print(f"Densidad ρ                   {rho:.6f}")
print(f"Grado medio k̄               {2*E/V:.3f}")
print(f"κ(G)                         {kappa}")
print(f"S(G)                         {S:.4f}")
print(f"Diámetro                     {diam}")
print(f"Longitud media ℓ(G)          {ell:.3f} saltos")
print(f"Clustering medio C̄△          {cc_mean:.4f}")
print(f"Puentes                      {len(bridges)}")
print(f"Puntos de articulación       {len(articulations)}")
print(f"Comunidades (Girvan-Newman)  {n_comm}")
print(f"C_B máx                      {max(cb_vals):.4f}  ({J.nodes[top5_cb[0][0]]['nombre']})")
print(f"S̄ fallos aleatorios p₀=13%  {np.mean(S_random):.4f}  (σ={np.std(S_random):.4f})")
print(f"Primera fragmentación (grado) p = {first_frag_degree*100:.0f}%")
print(f"p* (grado)                   {p_star_degree*100:.0f}%" if p_star_degree else "p* (grado) > 49%")
print(f"Primera fragmentación (C_B)  p = {first_frag_cb*100:.0f}%")
print(f"p* (C_B)                     {p_star_cb*100:.0f}%" if p_star_cb else "p* (C_B) > 49%")

# Guardar curvas de resiliencia para la tesis
import json
results = {
    'metrics': {
        'V': V, 'E': E, 'density': rho, 'mean_degree': 2*E/V,
        'kappa': kappa, 'S': S, 'diameter': diam, 'mean_path': ell,
        'clustering': cc_mean, 'bridges': len(bridges),
        'articulations': len(articulations), 'communities': n_comm,
        'cb_max': max(cb_vals), 'cb_mean': np.mean(cb_vals),
    },
    'random_failures': {
        'p0': P0, 'R': R_RANDOM,
        'S_mean': float(np.mean(S_random)), 'S_std': float(np.std(S_random)),
        'S_min': float(min(S_random)),
        'pct_connected': connected_count / R_RANDOM
    },
    'attack_degree': {
        'p_values': p_values,
        'S_values': S_degree,
        'first_frag': first_frag_degree,
        'p_star': p_star_degree
    },
    'attack_cb': {
        'p_values': p_values,
        'S_values': S_cb,
        'first_frag': first_frag_cb,
        'p_star': p_star_cb
    },
    'top10_cb': [(J.nodes[n]['nombre'], float(cb[n]), J.degree(n),
                  float(cd[n]), float(nx.clustering(J, n)))
                 for n, _ in top5_cb]
}

with open(os.path.join(DATA_ADIF, 'resultados_adif_junctions.json'), 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\nResultados guardados en resultados_adif_junctions.json")
