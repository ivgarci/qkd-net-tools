"""
Versión en español para la tesis, con paneles individuales por caso.
El fork en inglés para reutilización en otro artículo está en
k_core_decomposition_en.py (sin las adiciones específicas de la tesis).

Descomposición k-core para los tres casos de estudio QKD.

El k-core de un grafo es el subgrafo maximal donde todos los nodos tienen
al menos k vecinos dentro del subgrafo. La jerarquía de k-shells revela
la estructura de núcleos robustos — ortogonal al grado/intermediación.

Genera:
  datos/k_core_decomposition.csv   — tabla caso, nodo, k_core_index
  figuras/k_core_jerarquia_3casos.pdf/.png
"""

import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_CYL  = os.path.join(BASE, '..', 'datos', 'cyl')
DATA_ESP  = os.path.join(BASE, '..', 'datos', 'espana')
DATA_ADIF = os.path.join(BASE, '..', 'datos', 'adif')
FIGS_OUT  = os.path.join(BASE, '..', 'figuras')
os.makedirs(FIGS_OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Carga de grafos
# ---------------------------------------------------------------------------

def load_cyl():
    return nx.from_pandas_adjacency(
        pd.read_csv(os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv'), index_col=0)
    )


def load_espana():
    return nx.from_pandas_adjacency(
        pd.read_csv(os.path.join(DATA_ESP, 'AdjacencyMatrixNamed45.csv'), index_col=0)
    )


def load_adif_junction_graph():
    """Reconstruye el grafo de junctions desde los CSV de ADIF."""
    nodes_df = pd.read_csv(
        os.path.join(DATA_ADIF, 'nodos_red_adif.csv'), quotechar='"', on_bad_lines='skip'
    )
    adj_df = pd.read_csv(
        os.path.join(DATA_ADIF, 'adyacencia_red_adif.csv'), quotechar='"', on_bad_lines='skip'
    )

    G_full = nx.Graph()
    connected = nodes_df[nodes_df['conectado'] == 'SI'].copy()
    for _, row in connected.iterrows():
        G_full.add_node(str(row['cod']), lat=float(row['lat']), lon=float(row['lon']))

    seen = set()
    for _, row in adj_df.iterrows():
        u, v = str(row['cod']), str(row['vecino_cod'])
        key = frozenset([u, v])
        if key in seen:
            continue
        seen.add(key)
        if G_full.has_node(u) and G_full.has_node(v):
            try:
                G_full.add_edge(u, v, dist_km=float(row['dist_km']))
            except (ValueError, TypeError, KeyError):
                continue

    comps = sorted(nx.connected_components(G_full), key=len, reverse=True)
    G_lcc = G_full.subgraph(comps[0]).copy()

    # Contracción de nodos grado 2 → junctions.
    # Orden determinista (no un `set`): evita dependencia de PYTHONHASHSEED
    # en J.nodes() (mismo arreglo que adif/analisis_adif_junctions.py; ver
    # pendientes.md §2).
    keep = sorted(n for n in G_lcc.nodes() if G_lcc.degree(n) != 2)
    J = nx.Graph()
    for n in keep:
        J.add_node(n, **G_lcc.nodes[n])
    visited = set()
    for start in keep:
        for nbr in list(G_lcc.neighbors(start)):
            ek = frozenset([start, nbr])
            if ek in visited:
                continue
            visited.add(ek)
            acc = G_lcc[start][nbr].get('dist_km', 0.0) or 0.0
            prev, cur = start, nbr
            while cur not in keep:
                neighbors = list(G_lcc.neighbors(cur))
                nxt = neighbors[0] if neighbors[1] == prev else neighbors[1]
                acc += G_lcc[cur][nxt].get('dist_km', 0.0) or 0.0
                visited.add(frozenset([cur, nxt]))
                prev, cur = cur, nxt
            if cur != start and not J.has_edge(start, cur):
                J.add_edge(start, cur, dist_km=acc)

    jcomps = list(nx.connected_components(J))
    if len(jcomps) > 1:
        J = J.subgraph(max(jcomps, key=len)).copy()
    return J


# ---------------------------------------------------------------------------
# Análisis k-core
# ---------------------------------------------------------------------------

def kcore_stats(G, label):
    core_numbers = nx.core_number(G)
    max_k = max(core_numbers.values())
    shells = {}
    for node, k in core_numbers.items():
        shells.setdefault(k, []).append(node)

    print(f"\n{label}:")
    print(f"  |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    print(f"  k_max = {max_k}")
    for k in sorted(shells):
        print(f"  k={k:2d}: {len(shells[k]):4d} nodos")

    return core_numbers, max_k, shells


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------

def plot_kcore_panel(ax, G, core_numbers, label):
    max_k = max(core_numbers.values())
    if max_k == 0:
        max_k = 1

    pos = nx.spring_layout(G, seed=42, k=1.2)

    cmap = cm.get_cmap('plasma')
    node_colors = [cmap(core_numbers[n] / max_k) for n in G.nodes()]
    node_sizes  = [20 + 25 * core_numbers[n] for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, edge_color='gray', width=0.5)
    sc = nx.draw_networkx_nodes(G, pos, ax=ax,
                                node_color=node_colors,
                                node_size=node_sizes,
                                alpha=0.85)

    # Etiquetar solo los nodos del k-core máximo si hay pocos
    inner = [n for n, k in core_numbers.items() if k == max_k]
    if len(inner) <= 15:
        labels = {n: str(n)[:10] for n in inner}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=4.5)

    ax.set_title(f'{label}\n$k_{{max}}={max_k}$, núcleo interno: {len(inner)} nodos',
                 fontsize=9)
    ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=max_k))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label='k-core')


def plot_kcore_single(G, core_numbers, label, out_dir, filename_stem):
    """Panel individual, más grande, de un único caso (para la tesis)."""
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_kcore_panel(ax, G, core_numbers, label)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f'{filename_stem}.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Guardado (panel individual): {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Descomposición k-core — tres casos QKD")
    print("=" * 60)

    grafos = [
        ('CyL (|V|=100)',    load_cyl,               'cyl',    'k_core_jerarquia_cyl'),
        ('España (|V|=950)', load_espana,             'espana', 'k_core_jerarquia_esp'),
        ('ADIF (|V|≈485)',   load_adif_junction_graph, 'adif',  'k_core_jerarquia_adif'),
    ]

    all_rows = []
    results = []

    for label, loader, tag, stem in grafos:
        try:
            G = loader()
        except Exception as e:
            print(f"No se pudo cargar {label}: {e}")
            continue

        core_numbers, max_k, shells = kcore_stats(G, label)
        results.append((label, G, core_numbers, stem))

        for node, k in core_numbers.items():
            all_rows.append({'caso': label, 'nodo': str(node), 'k_core_index': k})

    # Guardar CSV
    df_kcore = pd.DataFrame(all_rows)
    out_csv = os.path.join(BASE, '..', 'datos', 'k_core_decomposition.csv')
    df_kcore.to_csv(out_csv, index=False)
    print(f"\nGuardado: {out_csv}")

    # Figura
    if results:
        fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6))
        if len(results) == 1:
            axes = [axes]

        for ax, (label, G, core_numbers, stem) in zip(axes, results):
            plot_kcore_panel(ax, G, core_numbers, label)

        fig.suptitle('Jerarquía k-core — casos de estudio QKD',
                     fontsize=12, y=1.01)
        fig.tight_layout()

        for ext in ('pdf', 'png'):
            path = os.path.join(FIGS_OUT, f'k_core_jerarquia_3casos.{ext}')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Guardado: {path}")
        plt.close(fig)

        print("\nGenerando paneles individuales (para la tesis)...")
        for label, G, core_numbers, stem in results:
            plot_kcore_single(G, core_numbers, label, FIGS_OUT, stem)

    print("\nDone.")
