"""
Ataques sobre aristas (enlaces) de redes QKD.

En QKD los enlaces físicos (fibras oscuras) también son atacables:
corte de fibra, intercepción o degradación. Este script simula la
eliminación incremental de aristas ordenadas por su betweenness de arista.

Métricas calculadas:
  - Edge betweenness centrality (C_B^e) para cada arista
  - Identificación de los top-10 enlaces más críticos
  - Curva S(p) bajo ataque incremental de aristas
  - Identificación de puentes (aristas cuya eliminación desconecta la red)

Genera:
  datos/cyl/edge_attack_results.csv
  datos/espana/edge_attack_results.csv
  figuras/ataques_aristas_3casos.pdf/.png
"""

import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_CYL  = os.path.join(BASE, '..', 'datos', 'cyl')
DATA_ESP  = os.path.join(BASE, '..', 'datos', 'espana')
DATA_ADIF = os.path.join(BASE, '..', 'datos', 'adif')
FIGS_OUT  = os.path.join(BASE, '..', 'figuras')
os.makedirs(FIGS_OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Simulación de ataque sobre aristas
# ---------------------------------------------------------------------------

def simulate_edge_attack(G_orig, p_steps):
    """
    Ataque estático sobre aristas: ordena por C_B^e (calculado en G original)
    y elimina en ese orden. Mide S(p) para cada fracción p de aristas eliminadas.
    """
    N = G_orig.number_of_nodes()
    E = G_orig.number_of_edges()
    if E == 0:
        return pd.DataFrame({'p_pct': p_steps, 'S_rel': [1.0] * len(p_steps),
                              'num_components': [1] * len(p_steps)})

    print(f"  Calculando edge betweenness centrality (|E|={E})...")
    ebc = nx.edge_betweenness_centrality(G_orig, normalized=True)
    sorted_edges = sorted(ebc, key=ebc.get, reverse=True)

    results = []
    G = G_orig.copy()
    prev_target = 0

    for p in p_steps:
        target_removed = int((p / 100) * E)
        edges_now = target_removed - prev_target

        for i in range(edges_now):
            idx = prev_target + i
            if idx < len(sorted_edges):
                e = sorted_edges[idx]
                if G.has_edge(*e):
                    G.remove_edge(*e)

        prev_target = target_removed

        if G.number_of_nodes() == 0 or not any(True for _ in G.nodes()):
            s_rel = 0.0
            n_comp = 0
        elif nx.is_connected(G):
            s_rel = 1.0
            n_comp = 1
        else:
            gcc = max(nx.connected_components(G), key=len)
            s_rel = len(gcc) / N
            n_comp = nx.number_connected_components(G)

        results.append({'p_pct': p, 'S_rel': s_rel, 'num_components': n_comp})

    return pd.DataFrame(results)


def top_critical_edges(G, top_n=10):
    """Devuelve los top_n enlaces con mayor edge betweenness."""
    ebc = nx.edge_betweenness_centrality(G, normalized=True)
    sorted_ebc = sorted(ebc.items(), key=lambda x: x[1], reverse=True)
    return sorted_ebc[:top_n]


def find_bridges(G):
    """Puentes: aristas cuya eliminación desconecta el grafo."""
    return list(nx.bridges(G))


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------

def plot_edge_attacks(casos, out_dir):
    n = len(casos)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, caso in zip(axes, casos):
        df = caso['df']
        ax.plot(df['p_pct'], df['S_rel'], '-',
                color='darkorange', lw=2.0, label='Ataque aristas ($C_B^e$)')
        ax.axhline(0.5, color='black', lw=0.8, ls=':', alpha=0.5)

        ps = next(
            (int(row.p_pct) for row in df.itertuples() if row.S_rel < 0.5),
            None
        )
        if ps is not None:
            ax.axvline(ps, color='darkorange', lw=0.8, ls=':', alpha=0.6)
            ax.text(ps + 0.5, 0.53, rf'$p^*={ps}\%$',
                    color='darkorange', fontsize=8)

        n_bridges = caso.get('n_bridges', 0)
        ax.set_title(f"{caso['label']}\n"
                     f"Puentes: {n_bridges}  |  Top arista: {caso.get('top_edge', '—')}")
        ax.set_xlabel(r'Fracción de aristas eliminadas $p$ (%)')
        ax.set_xlim(0, df['p_pct'].max())
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r'$S(p) = |GCC| / |V|$')
    fig.suptitle('Resiliencia bajo ataques a enlaces — redes QKD',
                 fontsize=12, y=1.01)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f'ataques_aristas_3casos.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Guardado: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Ataques sobre aristas — redes QKD")
    print("=" * 60)

    p_steps = list(range(0, 50, 1))

    casos_config = [
        {'label': r'CyL ($|V|=100$)',
         'adj': os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv'),
         'out': os.path.join(DATA_CYL, 'edge_attack_results.csv')},
        {'label': r'España ($|V|=950$)',
         'adj': os.path.join(DATA_ESP, 'AdjacencyMatrixNamed45.csv'),
         'out': os.path.join(DATA_ESP, 'edge_attack_results.csv')},
    ]

    casos_plot = []

    for cfg in casos_config:
        print(f"\n{cfg['label']}")
        G = nx.from_pandas_adjacency(pd.read_csv(cfg['adj'], index_col=0))
        print(f"  |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")

        bridges = find_bridges(G)
        print(f"  Puentes: {len(bridges)}")

        top_edges = top_critical_edges(G, top_n=10)
        print("  Top-10 aristas más críticas (C_B^e):")
        for edge, score in top_edges:
            print(f"    {edge[0][:20]:20s} — {edge[1][:20]:20s}: {score:.6f}")

        df = simulate_edge_attack(G, p_steps)
        df.to_csv(cfg['out'], index=False)
        print(f"  Guardado: {cfg['out']}")

        top_edge_str = f"{top_edges[0][0][0][:12]}—{top_edges[0][0][1][:12]}" if top_edges else '—'
        casos_plot.append({
            'label': cfg['label'],
            'df': df,
            'n_bridges': len(bridges),
            'top_edge': top_edge_str,
        })

    # Intentar añadir ADIF desde JSON pre-computado si existe
    adif_json = os.path.join(DATA_ADIF, 'resultados_adif_junctions.json')
    if os.path.exists(adif_json):
        with open(adif_json) as f:
            adif_data = json.load(f)
        # Usar curva de ataque de grado como proxy (no hay edge attack pre-computado)
        ae = adif_data.get('attack_degree', {})
        if ae:
            df_adif = pd.DataFrame({
                'p_pct': [v * 100 for v in ae['p_values']],
                'S_rel': ae['S_values'],
            })
            casos_plot.append({
                'label': r'ADIF ($|V|≈485$) — grado (proxy)',
                'df': df_adif,
                'n_bridges': 138,
                'top_edge': 'ver JSON',
            })

    print("\nGenerando figura...")
    plot_edge_attacks(casos_plot, FIGS_OUT)
    print("Done.")
