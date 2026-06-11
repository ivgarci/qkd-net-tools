"""
Figures for the QKD-Service-Resilience paper (IEEE JSAC Quantum Series).

Generates (PDF + PNG) into articulos/QKD-Service-Resilience/Figures/:

  fig_cp_sp.{pdf,png}        (Fig 1, double column, 7.16 in)
      3 horizontal panels (CyL, Spain, ADIF). Per panel: S(p) (dashed) and
      normalised C(p)/C(0) (solid) for two representative protocols
      (random failures with 95% CI band, static betweenness attack).
      Horizontal dotted reference at 0.5. The visual gap between the C
      and S curves of the same protocol is the headline message of the
      paper: service capacity collapses before connectivity.

  fig_weighted_rank.{pdf,png}   (Fig 2, single column, 3.5 in)
      Scatter of unweighted betweenness rank (x) vs SKR-weighted
      betweenness rank (y), log-log, colour per network, y=x reference.
      Annotated outliers: Bif. Torrejon de Velasco (1 -> 155) and
      Alcazar de San Juan (62 -> 1) in ADIF; Toro (1 -> 5) and
      Villalpando (17 -> 1) in CyL.

  fig_null_model.{pdf,png}      (Fig 3, double column, 7.16 in)
      2 panels (Spain, ADIF), adaptive betweenness attack: real S(p)
      vs density-matched Erdos-Renyi null mean +/- std band (50
      realisations); normalised C(p)/C(0) as thin lines for both.

Inputs (datos/resultados_papers/):
  capacidad_{cyl,espana,adif}.csv, null_model_er.csv,
  betweenness_ponderada.csv

Run with:
    /Users/igarcia/my_env/bin/python analisis/generar_figuras_service_resilience.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE, '..', 'datos', 'resultados_papers')
PAPER_DIR = os.path.join(BASE, '..', '..', '..', 'articulos',
                         'QKD-Service-Resilience', 'Figures')
os.makedirs(PAPER_DIR, exist_ok=True)

# ── Style (consistent with generar_figuras_skr_routing.py) ───────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        8,
    'axes.labelsize':   8,
    'axes.linewidth':   0.7,
    'xtick.labelsize':  7,
    'ytick.labelsize':  7,
    'legend.fontsize':  7,
    'lines.linewidth':  1.4,
    'grid.alpha':       0.30,
    'grid.linewidth':   0.4,
})

COL_RANDOM = '#2166ac'   # blue   (portfolio palette)
COL_BETW   = '#d95f02'   # orange (portfolio palette)
COL_REAL   = '#b2182b'   # dark red  (real network, Fig 3)
COL_NULL   = '#4393c3'   # light blue (E-R null, Fig 3)

NET_LABELS = {'cyl': 'CyL', 'espana': 'Spain', 'adif': 'ADIF'}
NET_ORDER  = ['cyl', 'espana', 'adif']
NET_COLORS = {'cyl': '#1b9e77', 'espana': '#2166ac', 'adif': '#d95f02'}
NET_MARKER = {'cyl': 'o', 'espana': 's', 'adif': '^'}

ONE_COL = 3.5    # IEEE single column (in)
TWO_COL = 7.16   # IEEE double column (in)


def save(fig, stem):
    for ext, dpi in (('pdf', None), ('png', 300)):
        out = os.path.join(PAPER_DIR, f'{stem}.{ext}')
        fig.savefig(out, dpi=dpi, bbox_inches='tight')
        print(f'  -> {out}')
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# FIG 1 — C(p) vs S(p), 3 panels, random + static betweenness
# ═════════════════════════════════════════════════════════════════════════════

def fig_cp_sp():
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 2.45), sharey=True)

    for ax, red in zip(axes, NET_ORDER):
        df = pd.read_csv(os.path.join(DATA_DIR, f'capacidad_{red}.csv'))
        df = df[df.p <= 0.5]

        for proto, col, lbl in (('random', COL_RANDOM, 'random'),
                                ('betweenness_static', COL_BETW,
                                 'betweenness (static)')):
            d = df[df.protocolo == proto].sort_values('p')
            c0 = d.loc[d.p == 0.0, 'C_median'].iloc[0]

            # S(p): dashed
            ax.plot(d.p, d.S, ls='--', color=col, lw=1.2, alpha=0.85,
                    label=f'$S(p)$ — {lbl}')

            # C(p)/C(0): solid (random only sampled at multiples of 0.05)
            dc = d[d.C_median.notna()]
            ax.plot(dc.p, dc.C_median / c0, ls='-', color=col, lw=1.7,
                    marker='o' if proto == 'random' else None,
                    ms=2.6, label=f'$C(p)/C(0)$ — {lbl}')

            # 95% CI band for random C
            if proto == 'random' and dc.C_ci_low.notna().any():
                ax.fill_between(dc.p, dc.C_ci_low / c0, dc.C_ci_high / c0,
                                color=col, alpha=0.18, lw=0)

        ax.axhline(0.5, color='0.35', ls=':', lw=0.9)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel('Fraction of nodes removed $p$')
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.grid(True)
        ax.text(0.97, 0.95, NET_LABELS[red], transform=ax.transAxes,
                ha='right', va='top', fontsize=8.5, fontweight='bold')

    # avoid tick-label collision between adjacent panels
    for ax in axes[:2]:
        labels = [f'{t:g}' for t in ax.get_xticks()]
        labels[-1] = ''
        ax.set_xticklabels(labels)

    axes[0].set_ylabel('Normalised metric')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9,
               handlelength=1.9, borderpad=0.4, columnspacing=1.2)

    fig.subplots_adjust(wspace=0.08)
    save(fig, 'fig_cp_sp')


# ═════════════════════════════════════════════════════════════════════════════
# FIG 2 — weighted vs unweighted betweenness rank, log-log scatter
# ═════════════════════════════════════════════════════════════════════════════

def fig_weighted_rank():
    bw = pd.read_csv(os.path.join(DATA_DIR, 'betweenness_ponderada.csv'))

    fig, ax = plt.subplots(figsize=(ONE_COL, 3.1))

    for red in NET_ORDER:
        d = bw[bw.red == red]
        ax.scatter(d.rank_unweighted, d.rank_weighted,
                   s=7, alpha=0.45, lw=0,
                   color=NET_COLORS[red], marker=NET_MARKER[red],
                   label=NET_LABELS[red], rasterized=True)

    # y = x reference
    lims = [0.7, 1500]
    ax.plot(lims, lims, color='0.25', ls='--', lw=0.9, zorder=1)
    ax.text(500, 290, '$y=x$', fontsize=7, color='0.25', rotation=38)

    # Annotated outliers
    annot = [
        ('adif', 'TORREJON', 'Bif. Torrejón de\nVelasco (1→155)',
         (3.2, 420), 'left'),
        ('adif', 'ALCAZAR',  'Alcázar de San\nJuan (62→1)',
         (170, 3.4), 'left'),
        ('cyl',  'Toro',        'Toro (1→5)',        (1.05, 22), 'left'),
        ('cyl',  'Villalpando', 'Villalpando (17→1)', (2.4, 1.9), 'left'),
    ]
    for red, key, text, xytext, ha in annot:
        d = bw[(bw.red == red) & bw.nombre.str.contains(key, na=False)]
        row = d.iloc[0]
        ax.annotate(text,
                    xy=(row.rank_unweighted, row.rank_weighted),
                    xytext=xytext, fontsize=6.2, ha=ha, va='center',
                    arrowprops=dict(arrowstyle='->', lw=0.7, color='0.15',
                                    shrinkA=1, shrinkB=2))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel('Topological betweenness rank (unweighted)')
    ax.set_ylabel('Service betweenness rank ($\\ell_e = 1/\\mathrm{SKR}_e$)')
    ax.grid(True, which='both')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3,
              framealpha=0.9, handletextpad=0.2, borderpad=0.4,
              columnspacing=1.0, markerscale=1.6)

    save(fig, 'fig_weighted_rank')


# ═════════════════════════════════════════════════════════════════════════════
# FIG 3 — real network vs density-matched E-R null, adaptive betweenness
# ═════════════════════════════════════════════════════════════════════════════

def fig_null_model():
    nm = pd.read_csv(os.path.join(DATA_DIR, 'null_model_er.csv'))
    proto = 'betweenness_adaptive'

    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.6), sharey=True)

    for ax, red in zip(axes, ['espana', 'adif']):
        # Real network
        df = pd.read_csv(os.path.join(DATA_DIR, f'capacidad_{red}.csv'))
        d = df[(df.protocolo == proto) & (df.p <= 0.5)].sort_values('p')
        c0 = d.loc[d.p == 0.0, 'C_median'].iloc[0]

        # E-R null
        n = nm[(nm.red == red) & (nm.protocolo == proto)
               & (nm.p <= 0.5)].sort_values('p')
        nc = n[n.C_median_mean.notna()]
        nc0 = nc.loc[nc.p == 0.0, 'C_median_mean'].iloc[0]

        # S(p): real (thick) vs null mean +/- std (band)
        ax.plot(d.p, d.S, color=COL_REAL, lw=1.8,
                label='$S(p)$ — real network')
        ax.plot(n.p, n.S_mean, color=COL_NULL, lw=1.6, ls='--',
                label='$\\bar{S}(p)$ — E--R null ($\\pm 1$ std)')
        ax.fill_between(n.p, n.S_mean - n.S_std, n.S_mean + n.S_std,
                        color=COL_NULL, alpha=0.25, lw=0)

        # C(p)/C(0): thin lines
        dc = d[d.C_median.notna()]
        ax.plot(dc.p, dc.C_median / c0, color=COL_REAL, lw=0.9, ls='-',
                marker='o', ms=2.2, alpha=0.65,
                label='$C(p)/C(0)$ — real network')
        ax.plot(nc.p, nc.C_median_mean / nc0, color=COL_NULL, lw=0.9,
                ls='--', marker='s', ms=2.2, alpha=0.65,
                label='$\\bar{C}(p)/\\bar{C}(0)$ — E--R null')

        ax.axhline(0.5, color='0.35', ls=':', lw=0.9)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel('Fraction of nodes removed $p$')
        ax.grid(True)
        ax.text(0.97, 0.95, NET_LABELS[red], transform=ax.transAxes,
                ha='right', va='top', fontsize=8.5, fontweight='bold')

    # tick-label collision between adjacent panels
    for ax in axes:
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    labels = [f'{t:g}' for t in axes[0].get_xticks()]
    labels[-1] = ''
    axes[0].set_xticklabels(labels)

    axes[0].set_ylabel('Normalised metric')
    axes[1].legend(*axes[0].get_legend_handles_labels(),
                   loc='center right', framealpha=0.92, handlelength=2.0,
                   borderpad=0.4, labelspacing=0.35)

    fig.subplots_adjust(wspace=0.07)
    save(fig, 'fig_null_model')


if __name__ == '__main__':
    print('Generating service-resilience figures ...')
    fig_cp_sp()
    fig_weighted_rank()
    fig_null_model()
    print('Done.')
