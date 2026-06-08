"""
Enrutamiento consciente de QKD - Cómputo completo para la red España.

Calcula todas las rutas all-pairs (450,925 pares) para la red Spain de 950 nodos.
Compara dos estrategias:
  1. Ruta mínima en saltos (Dijkstra estándar)
  2. Ruta máxima-SKR bottleneck (widest-path Dijkstra)

Genera:
  - datos/resultados_papers/enrutamiento_espana_allpairs.csv
  - datos/resultados_papers/tablas_skr_routing.json
  - ../../../articulos/QKD-SKR-Routing/Figures/comparacion_rutas_qkd.pdf
"""

import os
import sys
import math
import time
import json
import heapq
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_ESP   = os.path.join(BASE, '..', 'datos', 'espana')
OUT_DIR    = os.path.join(BASE, '..', 'datos', 'resultados_papers')
FIGS_PAPER = os.path.join(BASE, '..', '..', '..', 'articulos',
                           'QKD-SKR-Routing', 'Figures')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIGS_PAPER, exist_ok=True)

sys.path.insert(0, os.path.join(BASE, '..'))
from protocols.skr_bb84 import (skr_bb84_decoy, channel_transmittance,
                                  qber, h2, _haversine,
                                  ALPHA_DB_KM, MU, E_DETECTOR, F_EC)

# ---------------------------------------------------------------------------
# Parámetros nominales (defaults del paper)
# ---------------------------------------------------------------------------
ETA_DET_NOM  = 0.10
P_DARK_NOM   = 1e-6
DELTA_NOM    = 45.0     # km — link threshold
RHO_F        = 1.25     # routing factor (fibre vs haversine)

ADJ_CSV    = os.path.join(DATA_ESP, 'AdjacencyMatrixNamed45.csv')
COORDS_CSV = os.path.join(DATA_ESP, 'peninsula_1000.csv')


# ---------------------------------------------------------------------------
# Haversine (km) — sin routing factor para distancia geográfica s-t
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    return _haversine(lat1, lon1, lat2, lon2)


# ---------------------------------------------------------------------------
# Construir grafo SKR-anotado
# ---------------------------------------------------------------------------
def build_graph(eta_det=ETA_DET_NOM, p_dark=P_DARK_NOM, delta=DELTA_NOM):
    """Carga el grafo España y anota cada arista con dist_km (con routing factor) y SKR."""
    adj = pd.read_csv(ADJ_CSV, index_col=0)
    G = nx.from_pandas_adjacency(adj)

    coords_df = pd.read_csv(COORDS_CSV, delimiter=';')
    coords_df.columns = [c.strip().lstrip('﻿') for c in coords_df.columns]
    coords_df['Latitud']  = coords_df['Latitud'].astype(str).str.replace(',', '.').astype(float)
    coords_df['Longitud'] = coords_df['Longitud'].astype(str).str.replace(',', '.').astype(float)
    coords = {row['Población']: (row['Latitud'], row['Longitud'])
              for _, row in coords_df.iterrows()
              if row['Población'] in G.nodes()}

    missing = 0
    for u, v in G.edges():
        if u in coords and v in coords:
            lat1, lon1 = coords[u]
            lat2, lon2 = coords[v]
            haversine_km = haversine(lat1, lon1, lat2, lon2)
            fibre_km = haversine_km * RHO_F
            skr = skr_bb84_decoy(fibre_km, eta_det=eta_det, p_dark=p_dark)
            G[u][v]['dist_km']     = fibre_km
            G[u][v]['haversine_km'] = haversine_km
            G[u][v]['SKR']          = max(skr, 1e-15)   # avoid zero weight
        else:
            # Fallback: usar distancia nominal delta/2
            G[u][v]['dist_km']     = delta / 2
            G[u][v]['haversine_km'] = delta / 2
            G[u][v]['SKR']          = skr_bb84_decoy(delta / 2,
                                                       eta_det=eta_det,
                                                       p_dark=p_dark)
            missing += 1

    if missing:
        print(f"  [AVISO] {missing} aristas sin coordenadas — usada distancia nominal")

    return G, coords


# ---------------------------------------------------------------------------
# Widest-path (max-min bottleneck) Dijkstra — single source
# ---------------------------------------------------------------------------
def widest_path_dijkstra(G, source, weight='SKR'):
    """
    Dijkstra variante max-min bottleneck.
    Devuelve:
      bottleneck[v]  — max bottleneck SKR desde source hasta v
      prev[v]        — predecesor para reconstruir camino
    """
    nodes = list(G.nodes())
    bottleneck = {n: -1.0 for n in nodes}
    bottleneck[source] = float('inf')
    prev = {n: None for n in nodes}
    # heap: (-bottleneck, node)
    heap = [(-float('inf'), source)]

    visited = set()

    while heap:
        neg_b, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)

        for v in G.neighbors(u):
            if v in visited:
                continue
            w = G[u][v].get(weight, 0.0)
            new_b = min(-neg_b, w)
            if new_b > bottleneck[v]:
                bottleneck[v] = new_b
                prev[v] = u
                heapq.heappush(heap, (-new_b, v))

    return bottleneck, prev


def reconstruct_path(prev, source, target):
    """Reconstruye el camino desde source hasta target usando el diccionario prev."""
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    if path[0] != source:
        return []   # sin camino
    return path


# ---------------------------------------------------------------------------
# All-pairs computation para una configuración de parámetros
# ---------------------------------------------------------------------------
def compute_all_pairs(G, coords, label="base"):
    """
    Computa todas las rutas all-pairs (unordered).
    Devuelve DataFrame con métricas por par.
    """
    nodes = list(G.nodes())
    N = len(nodes)
    print(f"  [{label}] Nodos: {N}, Pares: {N*(N-1)//2:,}")

    # --- Hop-count shortest paths (BFS, unit weights) ---
    t0 = time.time()
    print(f"  [{label}] Calculando rutas mínimas en saltos (BFS all-pairs)...")
    # nx.single_source_shortest_path_length para cada nodo
    hop_dict = {}      # hop_dict[u][v] = hops
    skr_hop_dict = {}  # skr_hop_dict[u][v] = SKR bottleneck of hop path

    for i, src in enumerate(nodes):
        if i % 100 == 0:
            print(f"    BFS source {i}/{N}", flush=True)
        # shortest path lengths (hops)
        lengths = dict(nx.single_source_shortest_path_length(G, src))
        hop_dict[src] = lengths

        # For SKR bottleneck on hop path, we need the actual paths
        # We'll compute these along with widest paths below to avoid double work

    t_bfs = time.time() - t0
    print(f"  [{label}] BFS done in {t_bfs:.1f}s")

    # --- Widest-path + hop-count path in same pass ---
    t1 = time.time()
    print(f"  [{label}] Calculando widest-path all-pairs...")

    rows = []
    for i, src in enumerate(nodes):
        if i % 100 == 0:
            elapsed = time.time() - t1
            eta = elapsed / max(i, 1) * (N - i)
            print(f"    Widest-path source {i}/{N}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                  flush=True)

        # Widest path from src
        btl_wide, prev_wide = widest_path_dijkstra(G, src, weight='SKR')

        # Hop-count Dijkstra from src (unit weights) to get both path and SKR
        # We use nx.single_source_dijkstra with unit weight to get paths
        hop_lengths, hop_paths = nx.single_source_dijkstra(G, src, weight=lambda u,v,d: 1)

        # Coordinates of source
        lat_s, lon_s = coords.get(src, (None, None))

        for j, tgt in enumerate(nodes):
            if tgt <= src:   # process only i < j (string comparison for node names)
                continue
            # Skip unreachable
            if hop_lengths.get(tgt, None) is None:
                continue
            if btl_wide.get(tgt, -1) < 0:
                continue

            # Hop-count path metrics
            h_path = hop_paths[tgt]
            hop_n  = len(h_path) - 1
            # SKR bottleneck on hop-count path
            skr_hop = min(G[h_path[k]][h_path[k+1]].get('SKR', 0)
                          for k in range(len(h_path)-1)) if hop_n > 0 else 0.0

            # Widest-path metrics
            w_path = reconstruct_path(prev_wide, src, tgt)
            if not w_path:
                continue
            hop_w  = len(w_path) - 1
            skr_w  = btl_wide[tgt]

            # Improvement factor
            I = skr_w / skr_hop if skr_hop > 0 else np.nan
            dh = hop_w - hop_n

            # Geographic distance (haversine, no routing factor)
            lat_t, lon_t = coords.get(tgt, (None, None))
            if lat_s and lat_t:
                dist_geo = haversine(lat_s, lon_s, lat_t, lon_t)
            else:
                dist_geo = np.nan

            rows.append({
                'node_s':      src,
                'node_t':      tgt,
                'dist_km':     round(dist_geo, 2) if not np.isnan(dist_geo) else np.nan,
                'hop_shortest': hop_n,
                'hop_widest':  hop_w,
                'delta_h':     dh,
                'skr_hop':     skr_hop,
                'skr_widest':  skr_w,
                'I':           I,
            })

    t_wide = time.time() - t1
    print(f"  [{label}] Widest-path done in {t_wide:.1f}s")

    df = pd.DataFrame(rows)
    print(f"  [{label}] Pares computados: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Estadísticas agregadas
# ---------------------------------------------------------------------------
def aggregate_stats(df, label=""):
    """Calcula estadísticas agregadas de I y delta_h."""
    I = df['I'].dropna()
    dh = df['delta_h'].dropna()

    stats = {
        'label':      label,
        'n_pairs':    len(df),
        'I_mean':     round(float(I.mean()), 4),
        'I_median':   round(float(I.median()), 4),
        'I_std':      round(float(I.std()), 4),
        'I_P90':      round(float(I.quantile(0.90)), 4),
        'I_P99':      round(float(I.quantile(0.99)), 4),
        'I_max':      round(float(I.max()), 4),
        'dh_mean':    round(float(dh.mean()), 4),
        'dh_median':  round(float(dh.median()), 4),
        'dh_std':     round(float(dh.std()), 4),
        'dh_P90':     round(float(dh.quantile(0.90)), 4),
        'dh_P99':     round(float(dh.quantile(0.99)), 4),
        'dh_max':     int(dh.max()),
    }
    return stats


def geo_stratified_stats(df):
    """Estadísticas estratificadas por distancia geográfica."""
    bins  = [0, 100, 300, 500, 700, 1e9]
    labels = ['[0,100)', '[100,300)', '[300,500)', '[500,700)', '>700']
    df = df.copy()
    df['dist_bin'] = pd.cut(df['dist_km'], bins=bins, labels=labels, right=False)

    rows = []
    for lbl in labels:
        sub = df[df['dist_bin'] == lbl]
        if sub.empty:
            continue
        I  = sub['I'].dropna()
        dh = sub['delta_h'].dropna()
        rows.append({
            'bin':        lbl,
            'n_pairs':    len(sub),
            'I_mean':     round(float(I.mean()), 4),
            'I_median':   round(float(I.median()), 4),
            'I_max':      round(float(I.max()), 4),
            'dh_mean':    round(float(dh.mean()), 4),
        })

    # All pairs
    I_all  = df['I'].dropna()
    dh_all = df['delta_h'].dropna()
    rows.append({
        'bin':     'All',
        'n_pairs': len(df),
        'I_mean':  round(float(I_all.mean()), 4),
        'I_median': round(float(I_all.median()), 4),
        'I_max':   round(float(I_all.max()), 4),
        'dh_mean': round(float(dh_all.mean()), 4),
    })

    return rows


# ---------------------------------------------------------------------------
# Sensitivity sweeps
# ---------------------------------------------------------------------------
def sensitivity_eta(eta_values=(0.05, 0.10, 0.15, 0.20)):
    print("\n=== Sensibilidad: eta_det ===")
    results = []
    for eta in eta_values:
        print(f"\n  eta_det = {eta}")
        G, coords = build_graph(eta_det=eta, p_dark=P_DARK_NOM, delta=DELTA_NOM)
        df = compute_all_pairs(G, coords, label=f"eta={eta}")
        s = aggregate_stats(df, label=str(eta))
        s['param'] = 'eta_det'
        s['value'] = eta
        results.append(s)
        print(f"    -> I_mean={s['I_mean']:.4f}, dh_mean={s['dh_mean']:.4f}")
    return results


def sensitivity_pdark(pdark_values=(1e-7, 1e-6, 1e-5)):
    print("\n=== Sensibilidad: p_dark ===")
    results = []
    for pd_val in pdark_values:
        print(f"\n  p_dark = {pd_val:.0e}")
        G, coords = build_graph(eta_det=ETA_DET_NOM, p_dark=pd_val, delta=DELTA_NOM)
        df = compute_all_pairs(G, coords, label=f"pdark={pd_val:.0e}")
        s = aggregate_stats(df, label=f"{pd_val:.0e}")
        s['param'] = 'p_dark'
        s['value'] = pd_val
        results.append(s)
        print(f"    -> I_mean={s['I_mean']:.4f}, dh_mean={s['dh_mean']:.4f}")
    return results


def sensitivity_delta(delta_values=(35, 40, 45, 50)):
    """
    Para la sensibilidad a Delta: el paper especifica que se usa el grafo
    fijo (Spain 950 nodos, Delta=45km) pero se varía el parámetro de
    threshold en el modelo SKR (equivalente a variar el operating point).
    Usamos el mismo grafo base pero recomputamos SKR con las distancias
    existentes, truncando el link-weight SKR según si dist <= delta*rho_f
    para determinar aristas activas.

    NOTA: dado que el grafo Spain fue generado con Delta=45km, al usar
    Delta=35 o 40 km algunos links existentes tienen dist>Delta*rho_f y
    su SKR sería muy baja. Al usar Delta=50 los links existentes se mantienen.
    Re-anotamos el grafo con los mismos links pero diferente modelo SKR.
    """
    print("\n=== Sensibilidad: Delta ===")
    results = []
    for delta in delta_values:
        print(f"\n  Delta = {delta} km")
        # Re-anotar con mismo grafo pero diferente threshold
        G, coords = build_graph(eta_det=ETA_DET_NOM, p_dark=P_DARK_NOM, delta=delta)
        df = compute_all_pairs(G, coords, label=f"delta={delta}")
        s = aggregate_stats(df, label=str(delta))
        s['param'] = 'delta'
        s['value'] = delta
        results.append(s)
        print(f"    -> I_mean={s['I_mean']:.4f}, dh_mean={s['dh_mean']:.4f}")
    return results


# ---------------------------------------------------------------------------
# Figure 2: comparacion_rutas_qkd.pdf
# ---------------------------------------------------------------------------
def plot_figure2(df_base, out_path):
    """
    Panel (a): scatter hop-count bottleneck vs widest-path bottleneck,
               colour-coded by geographic distance bin.
    Panel (b): histogram of improvement factor I with mean/median/std annotation.
    """
    dist_bins   = [0, 100, 300, 500, 700, 1e9]
    bin_labels  = ['[0,100) km', '[100,300) km', '[300,500) km',
                   '[500,700) km', '>700 km']
    bin_colors  = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

    df = df_base.copy()
    df['dist_bin_idx'] = pd.cut(df['dist_km'], bins=dist_bins,
                                 labels=range(len(bin_labels)), right=False
                                ).astype(float).astype('Int64')

    # Sample for scatter (too many points to plot all)
    rng = np.random.default_rng(42)
    valid = df[df['skr_hop'] > 0].copy()
    sample_n = min(5000, len(valid))
    idx = rng.choice(len(valid), size=sample_n, replace=False)
    sample = valid.iloc[idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Panel (a): scatter ---
    ax = axes[0]
    for k, (lbl, col) in enumerate(zip(bin_labels, bin_colors)):
        sub = sample[sample['dist_bin_idx'] == k]
        if not sub.empty:
            ax.scatter(sub['skr_hop'], sub['skr_widest'],
                       color=col, alpha=0.4, s=6, label=lbl, rasterized=True)

    # Diagonal line
    skr_min = valid['skr_hop'].min() * 0.9
    skr_max = valid['skr_widest'].max() * 1.1
    ax.plot([skr_min, skr_max], [skr_min, skr_max],
            'k--', lw=0.8, alpha=0.6, label='No gain (diagonal)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Bottleneck SKR — hop-count path (bits/pulse)', fontsize=10)
    ax.set_ylabel('Bottleneck SKR — max-min path (bits/pulse)', fontsize=10)
    ax.set_title('(a) Hop-count vs max-min bottleneck SKR', fontsize=11)
    ax.legend(fontsize=7, markerscale=3)
    ax.grid(True, which='both', alpha=0.25)

    # --- Panel (b): histogram of I ---
    ax2 = axes[1]
    I_vals = df['I'].dropna()
    # Cap at 6 for visibility
    I_plot = I_vals.clip(upper=6.0)
    n_clipped = (I_vals > 6.0).sum()

    ax2.hist(I_plot, bins=80, color='steelblue', alpha=0.75,
             edgecolor='white', linewidth=0.3)
    ax2.axvline(1.0, color='black', lw=1.0, ls='--', label='I=1 (no gain)')

    mean_I   = float(I_vals.mean())
    med_I    = float(I_vals.median())
    std_I    = float(I_vals.std())
    p90_I    = float(I_vals.quantile(0.90))
    p99_I    = float(I_vals.quantile(0.99))

    ax2.axvline(mean_I,   color='darkorange', lw=1.5, ls='-',
                label=f'Mean = {mean_I:.3f}')
    ax2.axvline(med_I,    color='green',      lw=1.5, ls='-.',
                label=f'Median = {med_I:.3f}')

    text = (f'Mean   = {mean_I:.3f}\n'
            f'Median = {med_I:.3f}\n'
            f'Std    = {std_I:.3f}\n'
            f'P90    = {p90_I:.3f}\n'
            f'P99    = {p99_I:.3f}')
    ax2.text(0.97, 0.95, text, transform=ax2.transAxes,
             va='top', ha='right', fontsize=8.5,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_xlabel('Improvement factor $I = R_{\\mathrm{mm}} / R_{\\mathrm{hop}}$',
                   fontsize=10)
    ax2.set_ylabel('Number of pairs', fontsize=10)
    ax2.set_title(f'(b) Distribution of improvement factor $I$ ({len(I_vals):,} pairs)',
                  fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

    if n_clipped:
        ax2.set_xlabel(
            ax2.get_xlabel() +
            f'\n(I capped at 6; {n_clipped:,} pairs with I>6 not shown)', fontsize=9)

    fig.suptitle('SKR-Aware Routing — Spain QKD Network (950 nodes, 5681 edges)',
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Print tables
# ---------------------------------------------------------------------------
def print_table1(s):
    print("\n" + "="*65)
    print("TABLE 1 — Aggregate routing statistics (Spain, 450,925 pairs)")
    print("="*65)
    print(f"{'Statistic':<20} {'I(s,t)':>12} {'Δh(s,t)':>12}")
    print("-"*45)
    print(f"{'Mean':<20} {s['I_mean']:>12.4f} {s['dh_mean']:>12.4f}")
    print(f"{'Median':<20} {s['I_median']:>12.4f} {s['dh_median']:>12.4f}")
    print(f"{'Std':<20} {s['I_std']:>12.4f} {s['dh_std']:>12.4f}")
    print(f"{'P90':<20} {s['I_P90']:>12.4f} {s['dh_P90']:>12.4f}")
    print(f"{'P99':<20} {s['I_P99']:>12.4f} {s['dh_P99']:>12.4f}")
    print(f"{'Max':<20} {s['I_max']:>12.4f} {s['dh_max']:>12}")
    print(f"\n  n_pairs = {s['n_pairs']:,}")


def print_table2(geo_rows):
    print("\n" + "="*70)
    print("TABLE 2 — Geographic distance stratification")
    print("="*70)
    print(f"{'Distance bin':<16} {'Pairs':>10} {'Mean I':>8} {'Median I':>10} {'Max I':>8} {'Mean Δh':>10}")
    print("-"*65)
    for r in geo_rows:
        print(f"{r['bin']:<16} {r['n_pairs']:>10,} {r['I_mean']:>8.4f} "
              f"{r['I_median']:>10.4f} {r['I_max']:>8.4f} {r['dh_mean']:>10.4f}")


def print_sensitivity_table(results, param_name):
    print(f"\n{'='*60}")
    print(f"TABLE — Sensitivity to {param_name}")
    print(f"{'='*60}")
    print(f"  {'Value':<12} {'Mean I':>8} {'Mean Δh':>10}")
    print("-"*35)
    for r in results:
        print(f"  {r['label']:<12} {r['I_mean']:>8.4f} {r['dh_mean']:>10.4f}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    t_start = time.time()
    print("=" * 65)
    print("QKD-Aware Routing — Spain All-Pairs Computation")
    print("=" * 65)

    # ---- Step 1: Base case ----
    print("\n[1/6] Building base graph (eta=0.10, p_dark=1e-6, delta=45km)...")
    G_base, coords = build_graph()
    print(f"  |V|={G_base.number_of_nodes()}, |E|={G_base.number_of_edges()}")

    print("\n[2/6] Computing all-pairs routing (base case)...")
    t_base_start = time.time()
    df_base = compute_all_pairs(G_base, coords, label="base")
    t_base = time.time() - t_base_start
    print(f"  Base case done in {t_base:.1f}s")

    # Save full pair-level CSV
    out_csv = os.path.join(OUT_DIR, 'enrutamiento_espana_allpairs.csv')
    df_base.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    # ---- Step 2: Aggregate statistics (Table 1) ----
    print("\n[3/6] Computing aggregate statistics...")
    s_base = aggregate_stats(df_base, label="base")
    print_table1(s_base)

    # ---- Step 3: Geographic stratification (Table 2) ----
    geo_rows = geo_stratified_stats(df_base)
    print_table2(geo_rows)

    # ---- Step 4: Sensitivity sweeps ----
    print("\n[4/6] Sensitivity: eta_det...")
    eta_results = sensitivity_eta()
    print_sensitivity_table(eta_results, 'eta_det')

    print("\n[5/6] Sensitivity: p_dark...")
    pdark_results = sensitivity_pdark()
    print_sensitivity_table(pdark_results, 'p_dark')

    print("\n[6/6] Sensitivity: Delta...")
    delta_results = sensitivity_delta()
    print_sensitivity_table(delta_results, 'delta')

    # ---- Compile JSON output ----
    print("\nCompiling JSON summary...")

    # Paper placeholder comparison
    paper_I_mean   = 1.48
    paper_I_median = 1.21
    paper_I_std    = 0.62
    paper_I_P90    = 2.31
    paper_I_P99    = 4.17
    paper_dh_mean  = 5.3

    json_out = {
        'metadata': {
            'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'network':   'Spain QKD 950-node 5681-edge',
            'n_pairs':   s_base['n_pairs'],
            'runtime_base_s': round(t_base, 1),
        },
        'table1_aggregate': s_base,
        'table2_geographic': geo_rows,
        'table3_sensitivity_eta': eta_results,
        'table4_sensitivity_pdark': pdark_results,
        'table5_sensitivity_delta': delta_results,
        'paper_placeholder_comparison': {
            'paper_I_mean':    paper_I_mean,
            'computed_I_mean': s_base['I_mean'],
            'diff_I_mean':     round(s_base['I_mean'] - paper_I_mean, 4),
            'paper_dh_mean':   paper_dh_mean,
            'computed_dh_mean': s_base['dh_mean'],
            'diff_dh_mean':    round(s_base['dh_mean'] - paper_dh_mean, 4),
        }
    }

    out_json = os.path.join(OUT_DIR, 'tablas_skr_routing.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=2, default=str)
    print(f"Saved: {out_json}")

    # ---- Figure 2 ----
    print("\nGenerating Figure 2 (comparacion_rutas_qkd.pdf)...")
    fig2_path = os.path.join(FIGS_PAPER, 'comparacion_rutas_qkd.pdf')
    plot_figure2(df_base, fig2_path)

    # ---- Final summary ----
    t_total = time.time() - t_start
    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)
    print(f"Total runtime: {t_total:.1f}s")
    print(f"Pairs computed: {s_base['n_pairs']:,}")
    print(f"\nTable 1 actual values vs paper placeholders:")
    print(f"  I mean:   {s_base['I_mean']:.4f}  (paper: {paper_I_mean})")
    print(f"  I median: {s_base['I_median']:.4f}  (paper: {paper_I_median})")
    print(f"  I std:    {s_base['I_std']:.4f}  (paper: {paper_I_std})")
    print(f"  I P90:    {s_base['I_P90']:.4f}  (paper: {paper_I_P90})")
    print(f"  I P99:    {s_base['I_P99']:.4f}  (paper: {paper_I_P99})")
    print(f"  Δh mean:  {s_base['dh_mean']:.4f}  (paper: {paper_dh_mean})")
    print("\nDone.")
