"""
Figures for IEEE TDSC submission: Static vs. Adaptive Adversary Models for QKD Networks.

Generates five publication figures in English with a colour palette distinct
from companion papers (which use steelblue/firebrick):

  Figures/resiliencia_ataques_cyl.pdf    — CyL: three attack protocols
  Figures/resiliencia_ataques_esp.pdf    — Spain: three attack protocols
  Figures/dinamico_vs_estatico.pdf       — CyL+Spain static vs adaptive panel
  Figures/resiliencia_ataques_dirigidos.pdf — ADIF quasi-tree inversion
  Figures/centralidades_intermediacion.pdf  — betweenness distributions (all 3)

Colour scheme (adversary paper, TDSC):
  Adaptive degree   A_D^a  #E76F51  orange   solid
  Static betweenness A_B^s #1B998B  teal     dashed
  Adaptive betweenness A_B^a #264653 dark     solid, bolder

Output directory: ../../articulos/QKD-Adversary/Figures/
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

# ── Paths ────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_CYL  = os.path.join(BASE, '..', 'datos', 'cyl')
DATA_ESP  = os.path.join(BASE, '..', 'datos', 'espana')
DATA_ADIF = os.path.join(BASE, '..', 'datos', 'adif')
FIGS_OUT  = os.path.normpath(
    os.path.join(BASE, '..', '..', '..', 'articulos',
                 'QKD-Adversary', 'Figures')
)
os.makedirs(FIGS_OUT, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
# Palette distinct from steelblue/firebrick used in ADIF/TNSM companion papers
COL_DEGREE   = '#E76F51'   # warm orange  — A_D^a
COL_STATIC   = '#1B998B'   # teal         — A_B^s
COL_ADAPTIVE = '#264653'   # dark navy    — A_B^a
COL_RANDOM   = '#A8DADC'   # pale blue    — random failures

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         10,
    'axes.labelsize':    10,
    'axes.titlesize':    10,
    'legend.fontsize':    8,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'figure.dpi':        200,
    'text.usetex':       False,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


# ── Helpers ──────────────────────────────────────────────────────────────────

def savefig(fig, stem):
    for ext in ('pdf', 'png'):
        path = os.path.join(FIGS_OUT, f'{stem}.{ext}')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close(fig)


def p_star(p_vals, s_vals, threshold=0.5):
    for p, s in zip(p_vals, s_vals):
        if s < threshold:
            return p
    return None


def annotate_pstar(ax, p, color, y_text=0.53):
    if p is None:
        return
    pct = p * 100 if p <= 1 else p
    ax.axvline(pct, color=color, lw=0.9, ls=':', alpha=0.65)
    # % must live outside $...$ in matplotlib mathtext
    ax.text(pct + 0.4, y_text,
            rf"$p^\star\!=\!{pct:.0f}$%",
            color=color, fontsize=7.5, va='bottom')


def load_three_curves(data_dir, N):
    """Return (df_degree_adaptive, df_betweenness_static, df_betweenness_adaptive)."""
    # Adaptive degree (recalculated after each removal — standard degree attack)
    df_cd = pd.read_csv(os.path.join(data_dir, 'incremental_targeted_attack_results.csv'))
    df_cd['p_pct'] = df_cd['Removal Fraction (%)']
    df_cd['S_rel'] = df_cd['Largest Connected Component Size'] / N

    # Static betweenness (ranking frozen at t=0)
    df_bs = pd.read_csv(os.path.join(data_dir, 'incremental_betweenness_attack_results.csv'))

    # Adaptive betweenness (recomputed each step)
    df_ba = pd.read_csv(os.path.join(data_dir, 'dynamic_betweenness_attack_results.csv'))

    return df_cd, df_bs, df_ba


# ── Figure 1 & 2: CyL and Spain — three protocols ────────────────────────────

def plot_three_protocols(df_cd, df_bs, df_ba, title, stem,
                         x_max=49, pstar_y=0.53):
    fig, ax = plt.subplots(figsize=(6.5, 4))

    ax.plot(df_cd['p_pct'], df_cd['S_rel'],
            color=COL_DEGREE, lw=2.0, ls='-',
            label=r'Adaptive degree ($\mathcal{A}_D^a$)')
    ax.plot(df_bs['p_pct'], df_bs['S_rel'],
            color=COL_STATIC, lw=1.8, ls='--',
            label=r'Static betweenness ($\mathcal{A}_B^s$)')
    ax.plot(df_ba['p_pct'], df_ba['S_rel'],
            color=COL_ADAPTIVE, lw=2.2, ls='-',
            label=r'Adaptive betweenness ($\mathcal{A}_B^a$)')

    ax.axhline(0.5, color='grey', lw=0.8, ls=':', alpha=0.7)

    ps_cd = p_star(df_cd['p_pct'], df_cd['S_rel'])
    ps_bs = p_star(df_bs['p_pct'], df_bs['S_rel'])
    ps_ba = p_star(df_ba['p_pct'], df_ba['S_rel'])

    for ps, col, y in [(ps_cd, COL_DEGREE, pstar_y + 0.12),
                       (ps_bs, COL_STATIC, pstar_y + 0.04),
                       (ps_ba, COL_ADAPTIVE, pstar_y - 0.04)]:
        annotate_pstar(ax, ps, col, y_text=y)

    ax.set_xlabel(r'Fraction of nodes removed $p$ (%)')
    ax.set_ylabel(r'Relative LCC size $S(p) = |\mathrm{LCC}| / |V|$')
    ax.set_title(title)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.legend(loc='upper right')
    ax.grid(True, which='major', alpha=0.25)

    fig.tight_layout()
    savefig(fig, stem)


# ── Figure 3: static vs adaptive panel comparison ────────────────────────────

def plot_static_vs_adaptive(cases, stem):
    fig, axes = plt.subplots(1, len(cases), figsize=(6.5 * len(cases), 4),
                             sharey=True)
    if len(cases) == 1:
        axes = [axes]

    for ax, case in zip(axes, cases):
        df_bs = case['df_bs']
        df_ba = case['df_ba']

        ax.plot(df_bs['p_pct'], df_bs['S_rel'],
                color=COL_STATIC, lw=1.8, ls='--',
                label=r'Static ($\mathcal{A}_B^s$)')
        ax.plot(df_ba['p_pct'], df_ba['S_rel'],
                color=COL_ADAPTIVE, lw=2.2, ls='-',
                label=r'Adaptive ($\mathcal{A}_B^a$)')

        ax.axhline(0.5, color='grey', lw=0.8, ls=':', alpha=0.7)

        ps_bs = p_star(df_bs['p_pct'], df_bs['S_rel'])
        ps_ba = p_star(df_ba['p_pct'], df_ba['S_rel'])
        for ps, col, y in [(ps_bs, COL_STATIC, 0.55),
                           (ps_ba, COL_ADAPTIVE, 0.46)]:
            annotate_pstar(ax, ps, col, y_text=y)

        D = (ps_bs / ps_ba) if ps_bs and ps_ba and ps_ba > 0 else None
        if D:
            ax.text(0.62, 0.80,
                    rf'$D = {D:.2f}$',
                    transform=ax.transAxes,
                    fontsize=9, color=COL_ADAPTIVE,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white',
                              ec=COL_ADAPTIVE, alpha=0.85))

        ax.set_title(case['title'])
        ax.set_xlabel(r'Fraction of nodes removed $p$ (%)')
        ax.set_xlim(0, case.get('x_max', 49))
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.legend(loc='upper right')
        ax.grid(True, which='major', alpha=0.25)

    axes[0].set_ylabel(r'Relative LCC size $S(p)$')
    fig.tight_layout()
    savefig(fig, stem)


# ── Figure 4: ADIF quasi-tree inversion ──────────────────────────────────────

def plot_adif(adif_json_path, stem):
    with open(adif_json_path) as f:
        data = json.load(f)
    N = data['metrics']['V']

    p_cd  = [v * 100 for v in data['attack_degree']['p_values']]
    s_cd  = data['attack_degree']['S_values']
    p_cb  = [v * 100 for v in data['attack_cb']['p_values']]
    s_cb  = data['attack_cb']['S_values']
    ps_cd = data['attack_degree']['p_star']
    ps_cb = data['attack_cb']['p_star']

    fig, ax = plt.subplots(figsize=(6.5, 4))

    ax.plot(p_cd, s_cd, color=COL_DEGREE, lw=2.0, ls='-',
            label=r'Adaptive degree ($\mathcal{A}_D^a$)')
    ax.plot(p_cb, s_cb, color=COL_STATIC, lw=1.8, ls='--',
            label=r'Static betweenness ($\mathcal{A}_B^s$)')

    ax.axhline(0.5, color='grey', lw=0.8, ls=':', alpha=0.7)
    for ps, col, y in [(ps_cd, COL_DEGREE, 0.55),
                       (ps_cb, COL_STATIC, 0.44)]:
        annotate_pstar(ax, ps, col, y_text=y)

    ax.text(0.55, 0.72,
            'Degree attack more destructive\n'
            r'($p^\star_{C_D} < p^\star_{C_B}$): quasi-tree inversion',
            transform=ax.transAxes,
            fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFF3E0', ec=COL_DEGREE,
                      alpha=0.9))

    ax.set_xlabel(r'Fraction of nodes removed $p$ (%)')
    ax.set_ylabel(r'Relative LCC size $S(p) = |\mathrm{LCC}| / |V|$')
    ax.set_title(r'ADIF railway network ($|V|\!=\!485$, $|E|\!=\!633$)')
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.legend(loc='upper right')
    ax.grid(True, which='major', alpha=0.25)

    fig.tight_layout()
    savefig(fig, stem)


# ── Figure 5: betweenness distributions ──────────────────────────────────────

def plot_betweenness_distributions(graphs, stem):
    """
    graphs: list of dicts with keys 'G', 'label', 'color'
    """
    fig, axes = plt.subplots(len(graphs), 1,
                             figsize=(6.5, 2.8 * len(graphs)),
                             sharex=False)
    if len(graphs) == 1:
        axes = [axes]

    for ax, info in zip(axes, graphs):
        G = info['G']
        cb = nx.betweenness_centrality(G, normalized=True)
        vals = list(cb.values())

        ax.hist(vals, bins=40, color=info['color'], alpha=0.75,
                edgecolor='white', linewidth=0.5)

        cb_max = max(cb, key=cb.get)
        v_max  = cb[cb_max]
        ax.axvline(v_max, color='black', lw=1.0, ls='--', alpha=0.6)
        ax.text(v_max + 0.002, ax.get_ylim()[1] * 0.7 if ax.get_ylim()[1] > 0 else 1,
                f'max={v_max:.3f}\n({cb_max})',
                fontsize=7, va='top')

        ax.set_ylabel('Node count')
        ax.set_title(info['label'])
        ax.grid(True, which='major', alpha=0.25)

    axes[-1].set_xlabel('Normalised betweenness centrality $C_B(v)$')
    fig.tight_layout()
    savefig(fig, stem)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 65)
    print('Adversary paper figures — QKD networks (TDSC submission)')
    print('=' * 65)
    print(f'Output: {FIGS_OUT}\n')

    # Load graphs
    adj_cyl = pd.read_csv(os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv'),
                          index_col=0)
    adj_esp = pd.read_csv(os.path.join(DATA_ESP, 'AdjacencyMatrixNamed45.csv'),
                          index_col=0)
    N_cyl = len(adj_cyl)
    N_esp = len(adj_esp)
    G_cyl = nx.from_pandas_adjacency(adj_cyl)
    G_esp = nx.from_pandas_adjacency(adj_esp)

    # Load attack curves
    print('Loading attack curves...')
    df_cyl_cd, df_cyl_bs, df_cyl_ba = load_three_curves(DATA_CYL, N_cyl)
    df_esp_cd, df_esp_bs, df_esp_ba = load_three_curves(DATA_ESP, N_esp)

    # Figure 1: CyL — three attack protocols
    print('\nFig 1: CyL — three attack protocols')
    plot_three_protocols(
        df_cyl_cd, df_cyl_bs, df_cyl_ba,
        title=r'CyL regional network ($|V|\!=\!100$, $|E|\!=\!254$, $\Delta\!=\!45$ km)',
        stem='resiliencia_ataques_cyl',
        x_max=49,
    )

    # Figure 2: Spain — three attack protocols
    print('\nFig 2: Spain — three attack protocols')
    plot_three_protocols(
        df_esp_cd, df_esp_bs, df_esp_ba,
        title=r'Spain national network ($|V|\!=\!950$, $|E|\!=\!5681$, $\Delta\!=\!45$ km)',
        stem='resiliencia_ataques_esp',
        x_max=49,
        pstar_y=0.55,
    )

    # Figure 3: static vs adaptive panel (CyL and Spain)
    print('\nFig 3: Static vs adaptive betweenness comparison')
    plot_static_vs_adaptive(
        cases=[
            {'title': r'CyL ($|V|\!=\!100$)',
             'df_bs': df_cyl_bs, 'df_ba': df_cyl_ba, 'x_max': 49},
            {'title': r'Spain ($|V|\!=\!950$)',
             'df_bs': df_esp_bs, 'df_ba': df_esp_ba, 'x_max': 49},
        ],
        stem='dinamico_vs_estatico',
    )

    # Figure 4: ADIF quasi-tree inversion
    print('\nFig 4: ADIF quasi-tree inversion')
    plot_adif(
        adif_json_path=os.path.join(DATA_ADIF, 'resultados_adif_junctions.json'),
        stem='resiliencia_ataques_dirigidos',
    )

    # Figure 5: betweenness centrality distributions (slow — computes betweenness)
    print('\nFig 5: Betweenness distributions (computing centrality...)')
    print('  CyL betweenness...')

    # Build ADIF junction graph from raw edge-list CSV
    print('  Building ADIF junction graph from raw edge-list...')
    adif_edge_path = os.path.join(DATA_ADIF, 'adyacencia_red_adif.csv')
    G_adif = None
    if os.path.exists(adif_edge_path):
        df_adif_raw = pd.read_csv(adif_edge_path)
        G_full = nx.Graph()
        for _, row in df_adif_raw.iterrows():
            u = str(row['cod'])
            v = str(row['vecino_cod'])
            if u != v:
                G_full.add_edge(u, v)
        # Keep LCC
        lcc_nodes = max(nx.connected_components(G_full), key=len)
        G_lcc = G_full.subgraph(lcc_nodes).copy()

        # Contract degree-2 chains → junction graph (degree 1 or ≥ 3)
        def build_junction_graph(G):
            junctions = {n for n in G.nodes() if G.degree(n) != 2}
            J = nx.Graph()
            J.add_nodes_from(junctions)
            for start in junctions:
                for nbr in G.neighbors(start):
                    if nbr in junctions:
                        J.add_edge(start, nbr)
                        continue
                    # Walk chain
                    prev, cur = start, nbr
                    while cur not in junctions:
                        nxt = [n for n in G.neighbors(cur) if n != prev]
                        if not nxt:
                            break
                        prev, cur = cur, nxt[0]
                    if cur in junctions and cur != start:
                        J.add_edge(start, cur)
            return J

        G_adif = build_junction_graph(G_lcc)
        print(f'  ADIF junction graph: |V|={G_adif.number_of_nodes()}, '
              f'|E|={G_adif.number_of_edges()}')
    else:
        print('  WARNING: adyacencia_red_adif.csv not found; ADIF panel skipped.')

    graph_infos = [
        {'G': G_cyl, 'label': r'CyL ($|V|\!=\!100$, $|E|\!=\!254$)',
         'color': COL_DEGREE},
        {'G': G_esp, 'label': r'Spain ($|V|\!=\!950$, $|E|\!=\!5681$)',
         'color': COL_STATIC},
    ]
    if G_adif is not None:
        graph_infos.append(
            {'G': G_adif,
             'label': r'ADIF junction graph ($|V|\!=\!485$, $|E|\!=\!633$)',
             'color': COL_ADAPTIVE})

    plot_betweenness_distributions(graph_infos, 'centralidades_intermediacion')

    print('\nDone — all figures written to', FIGS_OUT)
