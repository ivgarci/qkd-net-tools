"""
Figures for the Fault-Aware QKD paper (P8, Journal of Systems Architecture).

Generates (PDF + PNG) into articulos/QKD-Fault-Aware-JSA/Figures/:

  fig_s1_s3.{pdf,png}        (Fig 1, double column, ~7 in)
      Two panels.
      Left  (S1): C(delta)/C(0) vs delta for the three networks (one solid
          line per network), with a marker at the fleet-ageing threshold
          delta* = 0.5 and a horizontal reference at 0.5.
      Right (S3): C/C(0) vs k (number of blinded relays) on ADIF, the
          network where the three targeting strategies separate the most:
          cb_weighted, cb_unweighted, and random (+/- std band). Annotated
          that the topology stays intact (S = 1) across the whole axis.
      Headline: device faults collapse service capacity while the network
      remains connected.

  fig_cm.{pdf,png}           (Fig 2, single column, ~3.5 in)
      CM2 recovery curve C(m)/C(0) vs m on ADIF, for three restoration
      orderings (cb_weighted, cb_unweighted, random +/- std), with a
      horizontal reference at 0.9 and markers at m = 8, 9, 10.
      Message: restoring detectors recovers service, and the restoration
      order matters; the attack-optimal ranking is not the
      restoration-optimal one.

Inputs (datos/resultados_papers/):
  fallos_s1.csv         (red, delta, C_median, C_rel, frac_pares_cero)
  fallos_s3.csv         (red, estrategia, k, C_rel, frac_pares_cero, C_rel_std)
  contramedidas_cm2.csv (red, estrategia, m, C_rel, C_rel_std)

Run with:
    /Users/igarcia/my_env/bin/python analisis/generar_figuras_fault_aware.py
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
                         'QKD-Fault-Aware-JSA', 'Figures')
os.makedirs(PAPER_DIR, exist_ok=True)

# -- Style (consistent with generar_figuras_service_resilience.py) ------------
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

# Portfolio palette
NET_LABELS = {'cyl': 'CyL', 'espana': 'Spain', 'adif': 'ADIF'}
NET_ORDER  = ['cyl', 'espana', 'adif']
NET_COLORS = {'cyl': '#1b9e77', 'espana': '#2166ac', 'adif': '#d95f02'}
NET_MARKER = {'cyl': 'o', 'espana': 's', 'adif': '^'}

# Targeting / restoration strategies
STRAT_LABELS = {
    'cb_weighted':   'SKR-weighted betweenness ($C_B^w$)',
    'cb_unweighted': 'topological betweenness ($C_B$)',
    'random':        'random',
}
STRAT_COLORS = {
    'cb_weighted':   '#b2182b',   # dark red
    'cb_unweighted': '#2166ac',   # blue
    'random':        '#7f7f7f',   # grey
}
STRAT_MARKER = {'cb_weighted': 'o', 'cb_unweighted': 's', 'random': '^'}
STRAT_ORDER  = ['cb_weighted', 'cb_unweighted', 'random']

ONE_COL = 3.5    # single column (in)
TWO_COL = 7.0    # double column (in)


def save(fig, stem):
    for ext, dpi in (('pdf', None), ('png', 300)):
        out = os.path.join(PAPER_DIR, f'{stem}.{ext}')
        fig.savefig(out, dpi=dpi, bbox_inches='tight')
        print(f'  -> {out}')
    plt.close(fig)


# =============================================================================
# FIG 1 — S1 fleet ageing (left) + S3 adversarial blinding on ADIF (right)
# =============================================================================

def fig_s1_s3():
    s1 = pd.read_csv(os.path.join(DATA_DIR, 'fallos_s1.csv'))
    s3 = pd.read_csv(os.path.join(DATA_DIR, 'fallos_s3.csv'))

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(TWO_COL, 2.7))

    # -- Left panel: S1, C(delta)/C(0) vs delta, one line per network --------
    for red in NET_ORDER:
        d = s1[s1.red == red].sort_values('delta')
        axl.plot(d.delta, d.C_rel, ls='-', lw=1.7,
                 color=NET_COLORS[red], marker=NET_MARKER[red], ms=3.2,
                 label=NET_LABELS[red])
        # marker at the fleet-ageing threshold delta* = 0.5
        row = d[np.isclose(d.delta, 0.5)]
        if not row.empty:
            axl.plot(0.5, row.C_rel.iloc[0], marker='*', ms=9,
                     color=NET_COLORS[red], mec='black', mew=0.5, zorder=5)

    axl.axhline(0.5, color='0.35', ls=':', lw=0.9)
    axl.axvline(0.5, color='0.55', ls=(0, (1, 1)), lw=0.8)
    axl.text(0.5, 1.02, r'$\delta^{*}=0.5$', ha='center', va='bottom',
             fontsize=7, color='0.30')
    axl.set_xlim(0, 0.9)
    axl.set_ylim(0, 1.05)
    axl.set_xlabel(r'Fleet efficiency-loss fraction $\delta$')
    axl.set_ylabel(r'Normalised capacity $C(\delta)/C(0)$')
    axl.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    axl.grid(True)
    axl.legend(loc='lower left', framealpha=0.9, handlelength=1.8,
               borderpad=0.4)
    axl.text(0.03, 0.96, '(a) S1: fleet ageing', transform=axl.transAxes,
             ha='left', va='top', fontsize=8, fontweight='bold')

    # -- Right panel: S3, C/C(0) vs k on ADIF, three strategies --------------
    d3 = s3[s3.red == 'adif']
    for strat in STRAT_ORDER:
        ds = d3[d3.estrategia == strat].sort_values('k')
        axr.plot(ds.k, ds.C_rel, ls='-', lw=1.7,
                 color=STRAT_COLORS[strat], marker=STRAT_MARKER[strat],
                 ms=3.4, label=STRAT_LABELS[strat])
        if strat == 'random' and ds.C_rel_std.notna().any():
            axr.fill_between(ds.k,
                             (ds.C_rel - ds.C_rel_std).clip(lower=0),
                             (ds.C_rel + ds.C_rel_std).clip(upper=1.0),
                             color=STRAT_COLORS[strat], alpha=0.18, lw=0)

    axr.set_xscale('log')
    axr.set_xlim(0.9, 60)
    axr.set_ylim(0, 1.08)
    axr.set_xlabel(r'Number of blinded relays $k$')
    axr.set_ylabel(r'Normalised capacity $C/C(0)$')
    axr.set_xticks([1, 2, 5, 10, 20, 50])
    axr.get_xaxis().set_major_formatter(
        matplotlib.ticker.ScalarFormatter())
    axr.minorticks_off()
    axr.grid(True, which='major')
    axr.legend(loc='lower left', framealpha=0.9, handlelength=1.8,
               borderpad=0.4)
    # topology stays intact across the whole axis
    axr.text(0.97, 0.80, 'topology intact ($S=1$)',
             transform=axr.transAxes, ha='right', va='top', fontsize=7,
             style='italic', color='0.20',
             bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.6',
                       lw=0.5, alpha=0.85))
    axr.text(0.97, 0.965, '(b) S3: targeted blinding (ADIF)',
             transform=axr.transAxes, ha='right', va='top', fontsize=8,
             fontweight='bold')

    fig.subplots_adjust(wspace=0.28)
    save(fig, 'fig_s1_s3')


# =============================================================================
# FIG 2 — CM2 recovery curve on ADIF, three restoration orderings
# =============================================================================

def fig_cm():
    cm = pd.read_csv(os.path.join(DATA_DIR, 'contramedidas_cm2.csv'))
    d = cm[cm.red == 'adif']

    fig, ax = plt.subplots(figsize=(ONE_COL, 3.0))

    for strat in STRAT_ORDER:
        ds = d[d.estrategia == strat].sort_values('m')
        ax.plot(ds.m, ds.C_rel, ls='-', lw=1.7,
                color=STRAT_COLORS[strat], marker=STRAT_MARKER[strat],
                ms=3.6, label=STRAT_LABELS[strat])
        if strat == 'random' and ds.C_rel_std.notna().any():
            ax.fill_between(ds.m,
                            (ds.C_rel - ds.C_rel_std).clip(lower=0),
                            (ds.C_rel + ds.C_rel_std).clip(upper=1.0),
                            color=STRAT_COLORS[strat], alpha=0.18, lw=0)

    ax.axhline(0.9, color='0.35', ls=':', lw=0.9)
    ax.text(1.1, 0.905, '0.9', fontsize=6.5, color='0.30', va='bottom')

    # markers at m = 8, 9, 10
    for mm in (8, 9, 10):
        ax.axvline(mm, color='0.70', ls=(0, (1, 1)), lw=0.7, zorder=0)

    ax.set_xlim(0.7, 10.4)
    ax.set_ylim(0.40, 1.05)
    ax.set_xlabel(r'Number of restored relays $m$')
    ax.set_ylabel(r'Recovered capacity $C(m)/C(0)$')
    ax.set_xticks(range(1, 11))
    ax.grid(True, axis='y')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=1,
              framealpha=0.9, handlelength=1.8, borderpad=0.4,
              labelspacing=0.3)

    save(fig, 'fig_cm')


if __name__ == '__main__':
    print('Generating fault-aware figures ...')
    fig_s1_s3()
    fig_cm()
    print('Done.')
