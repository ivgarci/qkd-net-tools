"""
Genera adif_junctions_mapa.pdf — figura estática geográfica del grafo de
junctions de la red ADIF, apta para incluir en LaTeX.

Nodos: junctions (grado ≠ 2) del grafo completo ADIF
Aristas: tramos contraídos, coloreados por longitud respecto a Δ_eff = 50 km
"""

import os
import sys
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_ADIF = os.path.join(BASE, '..', 'datos', 'adif')
FIGS_ADIF = os.path.join(BASE, '..', 'figuras', 'adif')
os.makedirs(FIGS_ADIF, exist_ok=True)

DELTA_EFF = 50.0   # km — umbral operativo
SEED = 42

OUT_PDF = os.path.join(FIGS_ADIF, 'adif_junctions_mapa.pdf')
OUT_PNG = os.path.join(FIGS_ADIF, 'adif_junctions_mapa.png')

# ── 1. Cargar datos ────────────────────────────────────────────────────────────
print("Cargando datos...")
nodes_df = pd.read_csv(
    os.path.join(DATA_ADIF, 'nodos_red_adif.csv'),
    quotechar='"', on_bad_lines='skip'
)
adj_df = pd.read_csv(
    os.path.join(DATA_ADIF, 'adyacencia_red_adif.csv'),
    quotechar='"', on_bad_lines='skip'
)

# ── 2. Construir grafo completo ────────────────────────────────────────────────
print("Construyendo grafo completo...")
G_full = nx.Graph()

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

seen = set()
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

# LCC
components = sorted(nx.connected_components(G_full), key=len, reverse=True)
G_lcc = G_full.subgraph(components[0]).copy()
print(f"  LCC: {G_lcc.number_of_nodes()} nodos, {G_lcc.number_of_edges()} aristas")

# ── 3. Grafo de junctions ──────────────────────────────────────────────────────
print("Construyendo grafo de junctions...")

def build_junction_graph(G):
    # Orden determinista (no un `set`): evita dependencia de PYTHONHASHSEED
    # en J.nodes() (mismo arreglo que adif/analisis_adif_junctions.py; ver
    # pendientes.md §2).
    keep = sorted(n for n in G.nodes() if G.degree(n) != 2)
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
            acc_dist = G[start][nbr].get('dist_km', 0.0) or 0.0
            tipo_red = G[start][nbr].get('tipo_red', '')
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
                continue
            if not J.has_edge(start, cur):
                J.add_edge(start, cur, dist_km=acc_dist, tipo_red=tipo_red)
            else:
                if acc_dist < J[start][cur]['dist_km']:
                    J[start][cur]['dist_km'] = acc_dist
    return J

J = build_junction_graph(G_lcc)
junc_components = list(nx.connected_components(J))
if len(junc_components) > 1:
    J = J.subgraph(max(junc_components, key=len)).copy()
print(f"  Junctions: {J.number_of_nodes()} nodos, {J.number_of_edges()} aristas")

# ── 4. Posiciones geográficas ──────────────────────────────────────────────────
pos = {n: (J.nodes[n]['lon'], J.nodes[n]['lat']) for n in J.nodes()}

# ── 5. Figura ──────────────────────────────────────────────────────────────────
print("Generando figura...")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f8f8',
})

fig, ax = plt.subplots(figsize=(10, 8))

# Separar aristas cortas y largas
short_edges = [(u, v) for u, v, d in J.edges(data=True)
               if d.get('dist_km', 0) <= DELTA_EFF]
long_edges  = [(u, v) for u, v, d in J.edges(data=True)
               if d.get('dist_km', 0) > DELTA_EFF]

# Dibujar aristas cortas (≤ 50 km) — azul oscuro, más delgadas
nx.draw_networkx_edges(J, pos, edgelist=short_edges, ax=ax,
                       edge_color='#2166ac', width=0.5, alpha=0.65)

# Dibujar aristas largas (> 50 km) — naranja, más gruesas
nx.draw_networkx_edges(J, pos, edgelist=long_edges, ax=ax,
                       edge_color='#d95f02', width=1.5, alpha=0.90)

# Colorear nodos por grado
degrees = dict(J.degree())
degree_vals = np.array([degrees[n] for n in J.nodes()])

# Nodos de alto grado (≥ 5) en rojo, resto en azul claro
high_degree = [n for n in J.nodes() if degrees[n] >= 5]
low_degree  = [n for n in J.nodes() if degrees[n] < 5]

nx.draw_networkx_nodes(J, pos, nodelist=low_degree, ax=ax,
                       node_size=6, node_color='#6baed6', alpha=0.75)
nx.draw_networkx_nodes(J, pos, nodelist=high_degree, ax=ax,
                       node_size=25, node_color='#d73027', alpha=0.95)

# Etiquetas solo para nodos de grado más alto
top_nodes = sorted(high_degree, key=lambda n: degrees[n], reverse=True)[:8]
labels = {n: J.nodes[n]['nombre'].replace('BIF. ', '').replace('BIF.', '').title()
          for n in top_nodes}
nx.draw_networkx_labels(J, pos, labels=labels, ax=ax,
                        font_size=5.5, font_color='#222222',
                        verticalalignment='bottom',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6, ec='none'))

# ── Leyenda ────────────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], color='#2166ac', linewidth=1.2, alpha=0.8,
           label=f'Corridor $\\leq {DELTA_EFF:.0f}$ km ({len(short_edges)} links)'),
    Line2D([0], [0], color='#d95f02', linewidth=2.0, alpha=0.9,
           label=f'Corridor $> {DELTA_EFF:.0f}$ km — relay required ({len(long_edges)} links)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#6baed6',
           markersize=5, label=f'Junction (degree $<$ 5, {len(low_degree)} nodes)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d73027',
           markersize=8, label=f'Critical junction (degree $\\geq$ 5, {len(high_degree)} nodes)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
          framealpha=0.9, edgecolor='#cccccc')

ax.set_xlabel('Longitude (°)', fontsize=9)
ax.set_ylabel('Latitude (°)', fontsize=9)
ax.tick_params(labelsize=8)
ax.set_title(f'ADIF junction graph ($|V_J|={J.number_of_nodes()}$, $|E_J|={J.number_of_edges()}$)',
             fontsize=11, pad=8)

# Eliminar spines superiores y derechos
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()

fig.savefig(OUT_PDF, format='pdf', dpi=150, bbox_inches='tight')
fig.savefig(OUT_PNG, format='png', dpi=150, bbox_inches='tight')
print(f'Guardado: {OUT_PDF}')
print(f'Guardado: {OUT_PNG}')
