"""
Ataques DINÁMICOS sobre redes QKD — recálculo de centralidad tras cada eliminación.

Diferencia con ataques estáticos: en lugar de ordenar nodos UNA VEZ y eliminar
en orden fijo, aquí se recalcula la centralidad del grafo RESTANTE en cada paso.
Esto modela un adversario que observa la red tras cada ataque (más conservador).

Para la tesis: comparar p*_dinámico vs p*_estático — el primero es siempre ≤ al
segundo porque la reorganización centralidad favorece al atacante.

Genera:
  datos/cyl/dynamic_betweenness_attack_results.csv
  datos/espana/dynamic_betweenness_attack_results.csv
  figuras/dinamico_vs_estatico.pdf/.png

Nota: Para España (950 nodos), cada paso recalcula betweenness en O(V·E).
El bucle completo puede tardar 20-60 minutos. Se muestra progreso cada 5 pasos.
"""

import os
import time
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
FIGS_OUT  = os.path.join(BASE, '..', 'figuras')
os.makedirs(FIGS_OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Simulación de ataque dinámico
# ---------------------------------------------------------------------------

def simulate_dynamic_attack(G_orig, centrality_func, p_steps, label=''):
    """
    Ataque dinámico: eliminar el nodo más central del grafo ACTUAL en cada paso.

    Para cada p en p_steps (porcentaje de nodos a eliminar en total), mide:
      - S_rel = |GCC| / |V_orig|
      - Número de componentes

    Retorna DataFrame con columnas [p_pct, S_rel, num_components].
    """
    N = G_orig.number_of_nodes()
    G = G_orig.copy()
    removed = []
    results = []
    prev_target = 0

    t0 = time.time()
    for step, p in enumerate(p_steps):
        target_removed = int((p / 100) * N)
        nodes_to_remove_now = target_removed - prev_target

        for _ in range(nodes_to_remove_now):
            if G.number_of_nodes() == 0:
                break
            centrality = centrality_func(G)
            top_node = max(centrality, key=centrality.get)
            G.remove_node(top_node)
            removed.append(top_node)

        prev_target = target_removed

        if G.number_of_nodes() == 0:
            s_rel = 0.0
            n_comp = 0
        elif nx.is_connected(G):
            s_rel = G.number_of_nodes() / N
            n_comp = 1
        else:
            gcc = max(nx.connected_components(G), key=len)
            s_rel = len(gcc) / N
            n_comp = nx.number_connected_components(G)

        results.append({'p_pct': p, 'S_rel': s_rel, 'num_components': n_comp})

        if step % 5 == 0 and label:
            elapsed = time.time() - t0
            print(f"  [{label}] p={p}%  S={s_rel:.3f}  comp={n_comp}  "
                  f"({elapsed:.0f}s)")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Carga de grafos y ataque estático (para comparación)
# ---------------------------------------------------------------------------

def load_static_results(data_dir, N):
    path = os.path.join(data_dir, 'incremental_betweenness_attack_results.csv')
    if not os.path.exists(path):
        path = os.path.join(data_dir, 'incremental_targeted_attack_results.csv')
        if not os.path.exists(path):
            return None
    df = pd.read_csv(path)
    if 'S_rel' not in df.columns:
        df['S_rel'] = df['Largest Connected Component Size'] / N
    if 'p_pct' not in df.columns:
        df['p_pct'] = df['Removal Fraction (%)']
    return df[['p_pct', 'S_rel']]


# ---------------------------------------------------------------------------
# Figura comparativa dinámico vs estático
# ---------------------------------------------------------------------------

def plot_dynamic_vs_static(casos, out_dir):
    fig, axes = plt.subplots(1, len(casos), figsize=(7 * len(casos), 5), sharey=True)
    if len(casos) == 1:
        axes = [axes]

    for ax, caso in zip(axes, casos):
        df_din = caso['df_dynamic']
        ax.plot(df_din['p_pct'], df_din['S_rel'], '-',
                color='darkred', lw=2.0, label='Dinámico ($C_B$ recalculada)')

        df_est = caso.get('df_static')
        if df_est is not None:
            ax.plot(df_est['p_pct'], df_est['S_rel'], '--',
                    color='steelblue', lw=1.8, label='Estático ($C_B$ fija)')

        ax.axhline(0.5, color='black', lw=0.8, ls=':', alpha=0.5)
        ax.set_title(caso['label'])
        ax.set_xlabel(r'Fracción eliminada $p$ (%)')
        ax.set_xlim(0, df_din['p_pct'].max())
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r'$S(p) = |GCC| / |V|$')
    fig.suptitle('Ataque dinámico vs estático por intermediación — redes QKD',
                 fontsize=12, y=1.01)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f'dinamico_vs_estatico.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Guardado: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p_steps = list(range(0, 50, 1))

    casos = [
        {'label': r'CyL ($|V|=100$)',
         'adj': os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv'),
         'out': os.path.join(DATA_CYL, 'dynamic_betweenness_attack_results.csv'),
         'data_dir': DATA_CYL},
        {'label': r'España ($|V|=950$)',
         'adj': os.path.join(DATA_ESP, 'AdjacencyMatrixNamed45.csv'),
         'out': os.path.join(DATA_ESP, 'dynamic_betweenness_attack_results.csv'),
         'data_dir': DATA_ESP},
    ]

    results_plot = []

    for caso in casos:
        print(f"\n{'='*60}")
        print(f"Ataque dinámico — {caso['label']}")
        print('='*60)

        G = nx.from_pandas_adjacency(pd.read_csv(caso['adj'], index_col=0))
        N = G.number_of_nodes()
        print(f"  |V|={N}, |E|={G.number_of_edges()}")

        df_dynamic = simulate_dynamic_attack(
            G,
            centrality_func=nx.betweenness_centrality,
            p_steps=p_steps,
            label=caso['label'],
        )
        df_dynamic.to_csv(caso['out'], index=False)
        print(f"\n  Guardado: {caso['out']}")

        df_static = load_static_results(caso['data_dir'], N)

        results_plot.append({
            'label': caso['label'],
            'df_dynamic': df_dynamic,
            'df_static': df_static,
        })

        # Resumen p*
        for name, df in [('Dinámico', df_dynamic), ('Estático', df_static)]:
            if df is None:
                continue
            ps = next(
                (int(row.p_pct) for row in df.itertuples() if row.S_rel < 0.5),
                None
            )
            print(f"  p*_{name}: {ps}%")

    print("\nGenerando figura comparativa...")
    plot_dynamic_vs_static(results_plot, FIGS_OUT)
    print("Done.")
