"""
Figura comparativa de resiliencia S(p) para los tres casos de estudio:
  CyL (100 nodos) | España peninsular (950 nodos) | ADIF (485 junctions)

Estrategias: ataque por grado (C_D) y por intermediación (C_B).

Genera:
  - figuras/comparacion_resiliencia_3casos.pdf/.png
  - figuras/diametro_degradacion_random.pdf/.png
  - datos/robustness_index_summary_completo.csv
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_CYL  = os.path.join(BASE, '..', 'datos', 'cyl')
DATA_ESP  = os.path.join(BASE, '..', 'datos', 'espana')
DATA_ADIF = os.path.join(BASE, '..', 'datos', 'adif')
FIGS_OUT  = os.path.join(BASE, '..', 'figuras')

os.makedirs(FIGS_OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'text.usetex': False,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def robustness_index(s_values) -> float:
    """R ≈ integral de S(p) dp, suma de Riemann con Δp=0.01."""
    return round(float(np.sum(s_values) * 0.01), 4)


def p_star(p_values, s_values, threshold: float = 0.5):
    """Mínimo p (%) tal que S_rel < threshold."""
    for p, s in zip(p_values, s_values):
        if s < threshold:
            return int(p)
    return None


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_degree_attack(data_dir: str, N: int) -> pd.DataFrame:
    path = os.path.join(data_dir, 'incremental_targeted_attack_results.csv')
    df = pd.read_csv(path)
    df['p_pct'] = df['Removal Fraction (%)']
    df['S_rel'] = df['Largest Connected Component Size'] / N
    return df[['p_pct', 'S_rel']]


def load_betweenness_attack(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, 'incremental_betweenness_attack_results.csv')
    return pd.read_csv(path)[['p_pct', 'S_rel']]


def load_adif_curves(json_path: str):
    """Devuelve (df_degree, df_betweenness) desde el JSON pre-computado de ADIF."""
    with open(json_path) as f:
        data = json.load(f)
    N = data['metrics']['V']

    def to_df(attack_dict):
        p_vals = attack_dict['p_values']
        s_vals = attack_dict['S_values']
        return pd.DataFrame({'p_pct': p_vals, 'S_rel': s_vals})

    return to_df(data['attack_degree']), to_df(data['attack_cb']), N


def load_random_failures(data_dir: str, N: int) -> pd.DataFrame:
    """Carga resultados de fallos aleatorios y devuelve estadísticas por trial."""
    path = os.path.join(data_dir, 'random_failure_results.csv')
    df = pd.read_csv(path)
    df['S_rel'] = df['Largest Connected Component Size'] / N
    # Diámetro: puede ser NaN si la red queda desconectada
    df['diameter'] = df['Diameter of Largest Component']
    return df


# ---------------------------------------------------------------------------
# Figura 1: Comparación resiliencia 3 casos
# ---------------------------------------------------------------------------

def make_comparison_figure(casos: list, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, caso in zip(axes, casos):
        df_cd = caso['df_cd']
        df_cb = caso['df_cb']

        ax.plot(df_cd['p_pct'], df_cd['S_rel'], '-',
                color='steelblue', lw=1.8, label=r'$C_D$ (grado)')
        ax.plot(df_cb['p_pct'], df_cb['S_rel'], '--',
                color='firebrick', lw=1.8, label=r'$C_B$ (intermediación)')
        ax.axhline(0.5, color='black', lw=0.8, ls=':', alpha=0.6)

        ps_cd = p_star(df_cd['p_pct'], df_cd['S_rel'])
        ps_cb = p_star(df_cb['p_pct'], df_cb['S_rel'])

        if ps_cd is not None:
            ax.axvline(ps_cd, color='steelblue', lw=0.8, ls=':', alpha=0.6)
            ax.text(ps_cd + 0.3, 0.53,
                    rf'$p^*_{{C_D}}={ps_cd}\%$',
                    color='steelblue', fontsize=7.5, va='bottom')
        if ps_cb is not None:
            ax.axvline(ps_cb, color='firebrick', lw=0.8, ls=':', alpha=0.6)
            ax.text(ps_cb + 0.3, 0.40,
                    rf'$p^*_{{C_B}}={ps_cb}\%$',
                    color='firebrick', fontsize=7.5, va='bottom')

        ax.set_title(caso['label'])
        ax.set_xlabel(r'Fracción eliminada $p$ (%)')
        ax.set_xlim(0, caso.get('x_max', 49))
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.legend(loc='upper right')
        ax.grid(True, which='major', alpha=0.3)

    axes[0].set_ylabel(r'$S(G_p) = |GCC| / |V|$')
    fig.suptitle('Comparación de resiliencia — tres casos de estudio QKD',
                 fontsize=12, y=1.01)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f'comparacion_resiliencia_3casos.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Guardado: {path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Figura 2: Degradación del diámetro (fallos aleatorios)
# ---------------------------------------------------------------------------

def plot_diameter_degradation(casos_rf: list, out_dir: str) -> None:
    """Curva de diámetro medio vs. fracción eliminada (fallos aleatorios)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = ['steelblue', 'firebrick']
    for caso, color in zip(casos_rf, colors):
        df = caso['df']
        # Agrupar por índice de trial y calcular media del diámetro
        # Los CSVs tienen R filas (una por trial), sin columna p_pct explícita
        # Se usa el valor único de p_0 como referencia
        p0 = caso.get('p0', 13)
        diameters = df['diameter'].replace(0, np.nan)
        mean_d = float(diameters.mean())
        std_d = float(diameters.std())

        # Representar distribución del diámetro como boxplot en p=p0
        ax.errorbar(p0, mean_d, yerr=std_d,
                    fmt='o', color=color, capsize=4, markersize=7,
                    label=f"{caso['label']} (media={mean_d:.1f}±{std_d:.1f})")

    ax.set_xlabel(r'Fracción eliminada $p_0$ (%)')
    ax.set_ylabel('Diámetro de la GCC')
    ax.set_title('Degradación del diámetro bajo fallos aleatorios')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f'diametro_degradacion_random.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Guardado: {path}")

    plt.close(fig)


def plot_diameter_vs_removal(casos_cd: list, out_dir: str) -> None:
    """Diámetro vs. fracción eliminada para ataques dirigidos por grado."""
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = ['steelblue', 'firebrick']
    for caso, color in zip(casos_cd, colors):
        df = caso['df_cd_raw']
        ax.plot(df['Removal Fraction (%)'],
                df['Diameter of Largest Component'],
                '-', color=color, lw=1.6, label=caso['label'])

    ax.set_xlabel(r'Fracción de nodos eliminados $p$ (%)')
    ax.set_ylabel('Diámetro de la GCC')
    ax.set_title('Evolución del diámetro bajo ataques dirigidos por grado')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))

    fig.tight_layout()

    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f'diametro_ataques_dirigidos.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Guardado: {path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Tabla unificada de robustez
# ---------------------------------------------------------------------------

def build_unified_robustness_table(casos: list, adif_data: dict) -> pd.DataFrame:
    rows = []
    for caso in casos:
        df_cd = caso['df_cd']
        df_cb = caso['df_cb']
        rows.append({
            'Caso': caso['label'],
            'R_grado': robustness_index(df_cd['S_rel'].values),
            'R_betweenness': robustness_index(df_cb['S_rel'].values),
            'p*_grado (%)': p_star(df_cd['p_pct'], df_cd['S_rel']),
            'p*_betweenness (%)': p_star(df_cb['p_pct'], df_cb['S_rel']),
        })

    # ADIF desde JSON — p_values en fracción [0..0.49], p_star también en fracción
    adif_cd = adif_data['attack_degree']
    adif_cb = adif_data['attack_cb']
    rows.append({
        'Caso': 'ADIF (|V|=485)',
        'R_grado': robustness_index(adif_cd['S_values']),
        'R_betweenness': robustness_index(adif_cb['S_values']),
        'p*_grado (%)': int(round(adif_cd['p_star'] * 100)) if adif_cd.get('p_star') is not None else None,
        'p*_betweenness (%)': int(round(adif_cb['p_star'] * 100)) if adif_cb.get('p_star') is not None else None,
    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 65)
    print("Comparación de resiliencia — tres casos QKD")
    print("=" * 65)

    # Cargar grafos (solo para N)
    import networkx as nx
    adj_cyl = pd.read_csv(os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv'), index_col=0)
    adj_esp = pd.read_csv(os.path.join(DATA_ESP, 'AdjacencyMatrixNamed45.csv'), index_col=0)
    N_cyl = len(adj_cyl)
    N_esp = len(adj_esp)

    # Cargar curvas de ataque
    df_cyl_cd = load_degree_attack(DATA_CYL, N_cyl)
    df_cyl_cb = load_betweenness_attack(DATA_CYL)
    df_esp_cd = load_degree_attack(DATA_ESP, N_esp)
    df_esp_cb = load_betweenness_attack(DATA_ESP)
    df_adif_cd, df_adif_cb, N_adif = load_adif_curves(
        os.path.join(DATA_ADIF, 'resultados_adif_junctions.json')
    )

    # Cargar raw para diámetro
    df_cyl_cd_raw = pd.read_csv(os.path.join(DATA_CYL, 'incremental_targeted_attack_results.csv'))
    df_esp_cd_raw = pd.read_csv(os.path.join(DATA_ESP, 'incremental_targeted_attack_results.csv'))

    casos = [
        {'label': r'CyL ($|V|=100$, $\Delta=45$ km)',
         'df_cd': df_cyl_cd, 'df_cb': df_cyl_cb,
         'df_cd_raw': df_cyl_cd_raw},
        {'label': r'España ($|V|=950$, $\Delta=45$ km)',
         'df_cd': df_esp_cd, 'df_cb': df_esp_cb,
         'df_cd_raw': df_esp_cd_raw},
        {'label': r'ADIF ($|V|=485$, $\Delta_{\rm eff}=50$ km)',
         'df_cd': df_adif_cd, 'df_cb': df_adif_cb,
         'x_max': max(df_adif_cd['p_pct'].max(), df_adif_cb['p_pct'].max())},
    ]

    print("\nGenerando figura comparativa...")
    make_comparison_figure(casos, FIGS_OUT)

    # Diámetro bajo ataques dirigidos
    print("\nGenerando figura de diámetro bajo ataques...")
    plot_diameter_vs_removal(casos[:2], FIGS_OUT)

    # Fallos aleatorios — diámetro
    df_rf_cyl = load_random_failures(DATA_CYL, N_cyl)
    df_rf_esp = load_random_failures(DATA_ESP, N_esp)
    casos_rf = [
        {'label': 'CyL', 'df': df_rf_cyl, 'p0': 13},
        {'label': 'España', 'df': df_rf_esp, 'p0': 13},
    ]
    print("\nGenerando figura de degradación del diámetro (fallos aleatorios)...")
    plot_diameter_degradation(casos_rf, FIGS_OUT)

    # Tabla unificada
    with open(os.path.join(DATA_ADIF, 'resultados_adif_junctions.json')) as f:
        adif_json = json.load(f)

    df_R = build_unified_robustness_table(casos[:2], adif_json)

    print("\n" + "=" * 65)
    print("TABLA UNIFICADA DE ROBUSTEZ")
    print("=" * 65)
    print(df_R.to_string(index=False))

    out_csv = os.path.join(BASE, '..', 'datos', 'robustness_index_summary_completo.csv')
    df_R.to_csv(out_csv, index=False)
    print(f"\nGuardado: {out_csv}")
    print("\nDone.")
