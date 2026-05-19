"""
Regenera esp_comunidades.png con visualización mejorada:
- Sin etiquetas (950 nodos → ilegible)
- Nodos pequeños con contorno
- Aristas finas y semitransparentes
- Sin whitespace (bbox_inches='tight')
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "../../datos/espana")
OUT_THESIS = os.path.join(BASE, "../../..", "697937f94a86c11bc36ad509", "Figures")

# --- Cargar datos ---
coords_df = pd.read_csv(os.path.join(DATA, "peninsula_1000.csv"), sep=";",
                        decimal=",")
coords_df.columns = [c.strip() for c in coords_df.columns]
# Algunas versiones tienen BOM
coords_df.columns = [c.lstrip('﻿') for c in coords_df.columns]

measures_df = pd.read_csv(os.path.join(DATA, "Node_Specific_Network_Measures.csv"),
                           index_col=0)

adj_df = pd.read_csv(os.path.join(DATA, "AdjacencyMatrixNamed45.csv"),
                     index_col=0)

# --- Grafo ---
G = nx.from_pandas_adjacency(adj_df)
print(f"Nodos: {G.number_of_nodes()}, Aristas: {G.number_of_edges()}")

# --- Posiciones geográficas ---
coord_map = {row["Población"]: (float(row["Longitud"]), float(row["Latitud"]))
             for _, row in coords_df.iterrows()}
# Solo nodos en el grafo
pos = {n: coord_map[n] for n in G.nodes() if n in coord_map}
missing = [n for n in G.nodes() if n not in coord_map]
if missing:
    print(f"  Nodos sin coordenadas: {len(missing)} — se excluyen del plot")
    G.remove_nodes_from(missing)

# --- Comunidades ---
cluster_map = measures_df["Cluster"].to_dict()
# Normalizar a enteros 0-based
unique_clusters = sorted(set(cluster_map.values()))
cluster_id = {c: i for i, c in enumerate(unique_clusters)}
n_communities = len(unique_clusters)
print(f"Comunidades: {n_communities}")

# Paleta cualitativa de 12 colores bien diferenciados
COLORS_12 = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#fabed4", "#469990", "#dcbeff", "#9A6324",
]
node_colors = [COLORS_12[cluster_id[cluster_map[n]] % len(COLORS_12)]
               for n in G.nodes()]

# --- Figura ---
fig, ax = plt.subplots(figsize=(14, 10))

# Aristas — muy finas y semitransparentes
nx.draw_networkx_edges(G, pos=pos, ax=ax,
                       edge_color="#888888", alpha=0.12,
                       width=0.35)

# Nodos — pequeños, con contorno fino
nx.draw_networkx_nodes(G, pos=pos, ax=ax,
                       node_color=node_colors,
                       node_size=18,
                       linewidths=0.3,
                       edgecolors="#333333",
                       alpha=0.90)

ax.axis("off")

# Leyenda compacta
legend_handles = [
    mpatches.Patch(facecolor=COLORS_12[cluster_id[c] % len(COLORS_12)],
                   edgecolor="#333333", linewidth=0.5,
                   label=f"Comunidad {c}")
    for c in unique_clusters
]
ax.legend(handles=legend_handles, loc="lower left",
          fontsize=7, ncol=2, framealpha=0.85,
          handlelength=1.2, handleheight=1.0,
          borderpad=0.5, labelspacing=0.3)

plt.tight_layout(pad=0.2)

# Guardar en figuras/espana/
out_local = os.path.join(BASE, "esp_comunidades_v2.png")
plt.savefig(out_local, dpi=300, bbox_inches="tight", pad_inches=0.05)
print(f"Guardado: {out_local}")

# Copiar directamente a Figures/ de la tesis
out_thesis = os.path.join(OUT_THESIS, "esp_comunidades.png")
if os.path.isdir(OUT_THESIS):
    plt.savefig(out_thesis, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Copiado a tesis: {out_thesis}")

plt.close()
