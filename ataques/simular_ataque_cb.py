"""
Simulación de ataques dirigidos por centralidad de intermediación (C_B) — estática.
Protocolo: el ranking de C_B se calcula UNA sola vez sobre el grafo completo G original;
la secuencia de eliminación no se recalcula tras cada paso (adversario con información offline).

Genera:
  - CSV con S(p) por paso para C_B (estático) en CyL y España
  - Figura comparativa S(p) grado vs. intermediación para ambas redes
  - Tabla de índice de robustez R = sum(S_rel(p)) * 0.01 para todos los protocolos
"""

import sys
import os
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_CYL  = os.path.join(BASE, '..', 'datos', 'cyl')
DATA_ESP  = os.path.join(BASE, '..', 'datos', 'espana')
FIGS_CYL  = os.path.join(BASE, '..', 'figuras', 'cyl')
FIGS_ESP  = os.path.join(BASE, '..', 'figuras', 'espana')
FIGS_OUT  = os.path.join(BASE, '..', 'figuras')

os.makedirs(FIGS_CYL, exist_ok=True)
os.makedirs(FIGS_ESP, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_graph(adj_csv):
    adj = pd.read_csv(adj_csv, index_col=0)
    G = nx.from_pandas_adjacency(adj)
    return G


def relative_gcc(G_sub):
    """Tamaño relativo de la componente gigante respecto a N original."""
    if G_sub.number_of_nodes() == 0:
        return 0.0
    comps = list(nx.connected_components(G_sub))
    return max(len(c) for c in comps) / G_sub.number_of_nodes()


def simulate_static_attack(G, centrality_metric='betweenness', p_steps=range(0, 50)):
    """
    Ataque estático: ranking calculado una sola vez sobre G.
    Devuelve lista de (p_pct, S_rel, num_components).
    S_rel normalizado sobre |V(G)| original.
    """
    N = G.number_of_nodes()

    print(f"  Calculando centralidad {centrality_metric} sobre {N} nodos...", flush=True)
    if centrality_metric == 'betweenness':
        centrality = nx.betweenness_centrality(G, normalized=True)
    elif centrality_metric == 'degree':
        centrality = nx.degree_centrality(G)
    else:
        raise ValueError(f"Métrica desconocida: {centrality_metric}")

    sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)

    results = []
    for p in p_steps:
        n_remove = int((p / 100) * N)
        G_copy = G.copy()
        G_copy.remove_nodes_from(sorted_nodes[:n_remove])

        remaining = G_copy.number_of_nodes()
        if remaining == 0:
            s_rel = 0.0
            n_comp = 0
        else:
            comps = list(nx.connected_components(G_copy))
            gcc = max(len(c) for c in comps)
            s_rel = gcc / N
            n_comp = len(comps)

        results.append({'p_pct': p, 'S_rel': round(s_rel, 6), 'n_components': n_comp})
        if p % 5 == 0:
            print(f"    p={p:2d}% → S={s_rel:.3f}, componentes={n_comp}", flush=True)

    return pd.DataFrame(results)


def robustness_index(df_sp):
    """R ≈ integral de S(p) dp, aproximada por suma de Riemann con Δp=0.01."""
    return round(float(df_sp['S_rel'].sum() * 0.01), 4)


def p_star(df_sp, threshold=0.5):
    """Mínimo p tal que S_rel < threshold."""
    below = df_sp[df_sp['S_rel'] < threshold]
    if below.empty:
        return None
    return int(below.iloc[0]['p_pct'])


# ---------------------------------------------------------------------------
# Cargar grafos
# ---------------------------------------------------------------------------

print("=" * 60)
print("Cargando grafos...")
G_cyl = load_graph(os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv'))
G_esp = load_graph(os.path.join(DATA_ESP, 'AdjacencyMatrixNamed45.csv'))
print(f"  CyL:    |V|={G_cyl.number_of_nodes()}, |E|={G_cyl.number_of_edges()}")
print(f"  España: |V|={G_esp.number_of_nodes()}, |E|={G_esp.number_of_edges()}")


# ---------------------------------------------------------------------------
# Ejecutar ataques por C_B (estático)
# ---------------------------------------------------------------------------

P_STEPS = range(0, 50)

print("\n" + "=" * 60)
print("Simulando ataque C_B estático — CyL (100 nodos)")
df_cyl_cb = simulate_static_attack(G_cyl, 'betweenness', P_STEPS)
out_cb_cyl = os.path.join(DATA_CYL, 'incremental_betweenness_attack_results.csv')
df_cyl_cb.to_csv(out_cb_cyl, index=False)
print(f"  Guardado: {out_cb_cyl}")

print("\n" + "=" * 60)
print("Simulando ataque C_B estático — España (950 nodos)")
df_esp_cb = simulate_static_attack(G_esp, 'betweenness', P_STEPS)
out_cb_esp = os.path.join(DATA_ESP, 'incremental_betweenness_attack_results.csv')
df_esp_cb.to_csv(out_cb_esp, index=False)
print(f"  Guardado: {out_cb_esp}")


# ---------------------------------------------------------------------------
# Cargar ataques por grado existentes (estáticos, misma metodología)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Cargando resultados de ataque por grado (existentes)...")
df_cyl_cd = pd.read_csv(os.path.join(DATA_CYL, 'incremental_targeted_attack_results.csv'))
df_esp_cd = pd.read_csv(os.path.join(DATA_ESP, 'incremental_targeted_attack_results.csv'))

# Normalizar S_rel a partir de LCC absoluto
N_cyl = G_cyl.number_of_nodes()
N_esp = G_esp.number_of_nodes()
df_cyl_cd['S_rel'] = df_cyl_cd['Largest Connected Component Size'] / N_cyl
df_esp_cd['S_rel'] = df_esp_cd['Largest Connected Component Size'] / N_esp
df_cyl_cd['p_pct'] = df_cyl_cd['Removal Fraction (%)']
df_esp_cd['p_pct'] = df_esp_cd['Removal Fraction (%)']


# ---------------------------------------------------------------------------
# Calcular índice de robustez R
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Índice de robustez R = ∫ S(p) dp  [suma de Riemann, Δp=0.01, p∈[0%,49%]]")
print()

results_R = {
    'Caso': ['CyL (|V|=100)', 'España (|V|=950)'],
    'R_grado': [
        robustness_index(df_cyl_cd),
        robustness_index(df_esp_cd),
    ],
    'R_betweenness': [
        robustness_index(df_cyl_cb),
        robustness_index(df_esp_cb),
    ],
    'p*_grado': [
        p_star(df_cyl_cd),
        p_star(df_esp_cd),
    ],
    'p*_betweenness': [
        p_star(df_cyl_cb),
        p_star(df_esp_cb),
    ],
}

df_R = pd.DataFrame(results_R)
print(df_R.to_string(index=False))

out_R = os.path.join(os.path.join(BASE, '..', 'datos'), 'robustness_index_summary.csv')
df_R.to_csv(out_R, index=False)
print(f"\nGuardado: {out_R}")


# ---------------------------------------------------------------------------
# Figuras: CyL y España — grado vs. intermediación
# ---------------------------------------------------------------------------

THESIS_FIGS = os.path.join(
    BASE, '..', '..', '..', '697937f94a86c11bc36ad509', 'Figures'
)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'text.usetex': False,
})

def make_resilience_fig(df_cd, df_cb, case_label, N, p_star_cd, p_star_cb, filename_stem):
    fig, ax = plt.subplots(figsize=(6, 4))

    p_cd = df_cd['p_pct'].values
    s_cd = df_cd['S_rel'].values
    p_cb = df_cb['p_pct'].values
    s_cb = df_cb['S_rel'].values

    ax.plot(p_cd, s_cd, '-', color='steelblue',  lw=1.8, label=r'Ataque por $C_D$ (grado)')
    ax.plot(p_cb, s_cb, '--', color='firebrick', lw=1.8, label=r'Ataque por $C_B$ (intermediación)')

    ax.axhline(0.5, color='black', lw=0.8, ls=':', alpha=0.7)

    if p_star_cd is not None:
        ax.axvline(p_star_cd, color='steelblue',  lw=0.8, ls=':', alpha=0.7)
        ax.text(p_star_cd + 0.3, 0.52,
                rf'$p^\star_{{C_D}}={p_star_cd}\%$',
                color='steelblue', fontsize=8.5, va='bottom')

    if p_star_cb is not None:
        ax.axvline(p_star_cb, color='firebrick', lw=0.8, ls=':', alpha=0.7)
        ax.text(p_star_cb + 0.3, 0.42,
                rf'$p^\star_{{C_B}}={p_star_cb}\%$',
                color='firebrick', fontsize=8.5, va='bottom')

    ax.set_xlabel(r'Fracción de nodos eliminados $p$ (\%)')
    ax.set_ylabel(r'$S(G_p) = |GCC| / |V|$')
    ax.set_title(case_label)
    ax.set_xlim(0, 49)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.legend(loc='upper right')
    ax.grid(True, which='major', alpha=0.3)

    fig.tight_layout()

    for ext in ('pdf', 'png'):
        for dest in [FIGS_OUT, THESIS_FIGS]:
            path = os.path.join(dest, f'{filename_stem}.{ext}')
            try:
                fig.savefig(path, dpi=150, bbox_inches='tight')
                print(f"  Guardado: {path}")
            except Exception as e:
                print(f"  WARN: no se pudo guardar en {path}: {e}")

    plt.close(fig)


print("\n" + "=" * 60)
print("Generando figuras...")

psc_cyl = p_star(df_cyl_cd)
psb_cyl = p_star(df_cyl_cb)
psc_esp = p_star(df_esp_cd)
psb_esp = p_star(df_esp_cb)

make_resilience_fig(
    df_cyl_cd, df_cyl_cb,
    r'Castilla y León ($|V|=100$, $\Delta=45$ km)',
    N_cyl, psc_cyl, psb_cyl,
    'resiliencia_ataques_cyl'
)

make_resilience_fig(
    df_esp_cd, df_esp_cb,
    r'España peninsular ($|V|=950$, $\Delta=45$ km)',
    N_esp, psc_esp, psb_esp,
    'resiliencia_ataques_esp'
)

# Figura combinada 2-paneles (para sustituir resiliencia_ataques_dirigidos.pdf)
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for ax, df_cd, df_cb, label, N, psc, psb in [
    (axes[0], df_cyl_cd, df_cyl_cb,
     r'Castilla y León ($|V|=100$)', N_cyl, psc_cyl, psb_cyl),
    (axes[1], df_esp_cd, df_esp_cb,
     r'España peninsular ($|V|=950$)', N_esp, psc_esp, psb_esp),
]:
    ax.plot(df_cd['p_pct'], df_cd['S_rel'], '-',  color='steelblue',  lw=1.8,
            label=r'$C_D$ (grado)')
    ax.plot(df_cb['p_pct'], df_cb['S_rel'], '--', color='firebrick', lw=1.8,
            label=r'$C_B$ (intermediación)')
    ax.axhline(0.5, color='black', lw=0.8, ls=':', alpha=0.7)

    if psc is not None:
        ax.axvline(psc, color='steelblue',  lw=0.8, ls=':', alpha=0.5)
    if psb is not None:
        ax.axvline(psb, color='firebrick', lw=0.8, ls=':', alpha=0.5)

    ax.set_xlabel(r'Fracción eliminada $p$ (\%)')
    ax.set_title(label)
    ax.set_xlim(0, 49)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, which='major', alpha=0.3)

axes[0].set_ylabel(r'$S(G_p) = |GCC| / |V|$')
fig.tight_layout()

for ext in ('pdf', 'png'):
    for dest in [FIGS_OUT, THESIS_FIGS]:
        path = os.path.join(dest, f'resiliencia_ataques_dirigidos.{ext}')
        try:
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  Guardado (2-panel): {path}")
        except Exception as e:
            print(f"  WARN: {path}: {e}")

plt.close(fig)

print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"\nCyL  — p*(C_D)={psc_cyl}%,  p*(C_B)={psb_cyl}%")
print(f"       R(C_D)={results_R['R_grado'][0]},   R(C_B)={results_R['R_betweenness'][0]}")
print(f"\nEspa — p*(C_D)={psc_esp}%,  p*(C_B)={psb_esp}%")
print(f"       R(C_D)={results_R['R_grado'][1]},   R(C_B)={results_R['R_betweenness'][1]}")
print("\nDone.")
