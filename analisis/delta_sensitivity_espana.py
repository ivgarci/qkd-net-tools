"""
Δ (link-distance threshold) sensitivity analysis — Spain QKD network.

For each Δ ∈ {35, 40, 45, 50} km, the Spain graph is rebuilt from scratch
using raw municipality coordinates (peninsula_1000.csv).  An edge (u,v) exists
iff haversine(u,v) ≤ Δ km.  Each edge is annotated with the BB84 decoy SKR
using fibre distance d_fibre = haversine_km × ρ_f.

This fixes the previous Table 5 computation in enrutamiento_espana_completo.py
which incorrectly re-used the same Δ=45 km adjacency matrix for all four
threshold values, producing identical results.

Outputs:
  - Prints Table 5 to stdout
  - Updates datos/resultados_papers/tablas_skr_routing.json
    (replaces the 'table5_sensitivity_delta' key with correct values)
"""

import os
import sys
import math
import time
import json
import heapq
import numpy as np
import pandas as pd
import networkx as nx

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_ESP = os.path.join(BASE, '..', 'datos', 'espana')
OUT_DIR  = os.path.join(BASE, '..', 'datos', 'resultados_papers')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE, '..'))
from protocols.skr_bb84 import skr_bb84_decoy, _haversine

COORDS_CSV = os.path.join(DATA_ESP, 'peninsula_1000.csv')
JSON_OUT   = os.path.join(OUT_DIR, 'tablas_skr_routing.json')

# Physical / routing constants (must match enrutamiento_espana_completo.py)
RHO_F       = 1.25   # routing factor: fibre_km = haversine_km * RHO_F
ETA_DET_NOM = 0.10
P_DARK_NOM  = 1e-6


# ---------------------------------------------------------------------------
# Haversine wrapper
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    return _haversine(lat1, lon1, lat2, lon2)


# ---------------------------------------------------------------------------
# Load coordinates
# ---------------------------------------------------------------------------
def load_coords(csv_path=COORDS_CSV):
    """Returns dict: poblacion -> (lat, lon)."""
    df = pd.read_csv(csv_path, delimiter=';')
    df.columns = [c.strip().lstrip('﻿').lstrip('﻿') for c in df.columns]
    df['Latitud']  = df['Latitud'].astype(str).str.replace(',', '.').astype(float)
    df['Longitud'] = df['Longitud'].astype(str).str.replace(',', '.').astype(float)
    coords = {row['Población']: (row['Latitud'], row['Longitud'])
              for _, row in df.iterrows()}
    return coords


# ---------------------------------------------------------------------------
# Build graph for a given Δ
# ---------------------------------------------------------------------------
def build_graph_for_delta(coords, delta_km,
                           eta_det=ETA_DET_NOM, p_dark=P_DARK_NOM):
    """
    Builds an undirected graph from municipality coordinates.
    Edge (u,v) exists iff haversine(u,v) <= delta_km.
    Edge weight 'SKR' uses fibre distance = haversine_km * RHO_F.
    """
    pops = list(coords.keys())
    n    = len(pops)

    G = nx.Graph()
    G.add_nodes_from(pops)

    n_edges = 0
    for i in range(n):
        lat1, lon1 = coords[pops[i]]
        for j in range(i + 1, n):
            lat2, lon2 = coords[pops[j]]
            hav_km = haversine(lat1, lon1, lat2, lon2)
            if hav_km <= delta_km:
                fibre_km = hav_km * RHO_F
                skr = skr_bb84_decoy(fibre_km, eta_det=eta_det, p_dark=p_dark)
                G.add_edge(pops[i], pops[j],
                           haversine_km=hav_km,
                           dist_km=fibre_km,
                           SKR=max(skr, 1e-15))
                n_edges += 1

    return G


# ---------------------------------------------------------------------------
# Widest-path (max-min bottleneck) Dijkstra — single source
# (identical to enrutamiento_espana_completo.py)
# ---------------------------------------------------------------------------
def widest_path_dijkstra(G, source, weight='SKR'):
    nodes = list(G.nodes())
    bottleneck = {n: -1.0 for n in nodes}
    bottleneck[source] = float('inf')
    prev = {n: None for n in nodes}
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
    path = []
    cur  = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    if not path or path[0] != source:
        return []
    return path


# ---------------------------------------------------------------------------
# All-pairs computation on a (possibly disconnected) graph
# ---------------------------------------------------------------------------
def compute_all_pairs(G, label=""):
    """
    Runs all-pairs widest-path + hop-shortest Dijkstra.
    If the graph is disconnected the computation is restricted to pairs
    within the same connected component.
    Returns a DataFrame of per-pair metrics.
    """
    nodes = list(G.nodes())
    N = len(nodes)
    print(f"  [{label}] |V|={N}, |E|={G.number_of_edges()}", flush=True)

    # Connectivity check
    comps = list(nx.connected_components(G))
    n_comps = len(comps)
    if n_comps > 1:
        sizes = sorted([len(c) for c in comps], reverse=True)
        print(f"  [{label}] DISCONNECTED — {n_comps} components, "
              f"sizes: {sizes[:5]}{'...' if len(sizes) > 5 else ''}", flush=True)
        # Work on largest connected component
        lcc_nodes = max(comps, key=len)
        G = G.subgraph(lcc_nodes).copy()
        nodes = list(G.nodes())
        N = len(nodes)
        print(f"  [{label}] Using largest component: |V|={N}, |E|={G.number_of_edges()}", flush=True)
    else:
        print(f"  [{label}] Connected: yes", flush=True)

    n_pairs_expected = N * (N - 1) // 2
    print(f"  [{label}] Expected pairs: {n_pairs_expected:,}", flush=True)

    t1 = time.time()
    rows = []

    for i, src in enumerate(nodes):
        if i % 100 == 0:
            elapsed = time.time() - t1
            eta = elapsed / max(i, 1) * (N - i) if i > 0 else 0
            print(f"    [{label}] source {i}/{N}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                  flush=True)

        # Widest-path from src
        btl_wide, prev_wide = widest_path_dijkstra(G, src, weight='SKR')

        # Hop-count Dijkstra from src
        hop_lengths, hop_paths = nx.single_source_dijkstra(
            G, src, weight=lambda u, v, d: 1)

        for tgt in nodes:
            if tgt <= src:
                continue
            if hop_lengths.get(tgt) is None:
                continue
            if btl_wide.get(tgt, -1) < 0:
                continue

            # Hop-count path
            h_path = hop_paths[tgt]
            hop_n  = len(h_path) - 1
            skr_hop = (min(G[h_path[k]][h_path[k + 1]].get('SKR', 0)
                           for k in range(len(h_path) - 1))
                       if hop_n > 0 else 0.0)

            # Widest path
            w_path = reconstruct_path(prev_wide, src, tgt)
            if not w_path:
                continue
            hop_w  = len(w_path) - 1
            skr_w  = btl_wide[tgt]

            I  = skr_w / skr_hop if skr_hop > 0 else np.nan
            dh = hop_w - hop_n

            rows.append({
                'node_s':       src,
                'node_t':       tgt,
                'hop_shortest': hop_n,
                'hop_widest':   hop_w,
                'delta_h':      dh,
                'skr_hop':      skr_hop,
                'skr_widest':   skr_w,
                'I':            I,
            })

    elapsed_total = time.time() - t1
    df = pd.DataFrame(rows)
    print(f"  [{label}] Done in {elapsed_total:.1f}s — {len(df):,} pairs", flush=True)
    return df, G   # return the (possibly trimmed) graph


# ---------------------------------------------------------------------------
# Aggregate statistics (same as enrutamiento_espana_completo.py)
# ---------------------------------------------------------------------------
def aggregate_stats(df, label=""):
    I  = df['I'].dropna()
    dh = df['delta_h'].dropna()
    return {
        'label':     label,
        'n_pairs':   len(df),
        'I_mean':    round(float(I.mean()), 4),
        'I_median':  round(float(I.median()), 4),
        'I_std':     round(float(I.std()), 4),
        'I_P90':     round(float(I.quantile(0.90)), 4),
        'I_P99':     round(float(I.quantile(0.99)), 4),
        'I_max':     round(float(I.max()), 4),
        'dh_mean':   round(float(dh.mean()), 4),
        'dh_median': round(float(dh.median()), 4),
        'dh_std':    round(float(dh.std()), 4),
        'dh_P90':    round(float(dh.quantile(0.90)), 4),
        'dh_P99':    round(float(dh.quantile(0.99)), 4),
        'dh_max':    int(dh.max()),
    }


# ---------------------------------------------------------------------------
# Print Table 5
# ---------------------------------------------------------------------------
def print_table5(results):
    print()
    print("=" * 90)
    print("TABLE 5 — Sensitivity to Δ (link-distance threshold)")
    print("  Graph rebuilt from raw coordinates for each Δ")
    print("=" * 90)
    hdr = (f"{'Δ (km)':<8} {'|V|':>6} {'|E|':>8} {'Connected?':>12} "
           f"{'n_pairs':>10} {'Mean I':>8} {'Median I':>10} "
           f"{'Std I':>8} {'P90 I':>8} {'P99 I':>8} {'Mean Δh':>10}")
    print(hdr)
    print("-" * 90)
    for r in results:
        conn = "yes" if r['connected'] else f"LCC={r['lcc_size']}"
        print(f"{r['delta']:<8} {r['n_nodes']:>6} {r['n_edges']:>8} {conn:>12} "
              f"{r['n_pairs']:>10,} {r['I_mean']:>8.4f} {r['I_median']:>10.4f} "
              f"{r['I_std']:>8.4f} {r['I_P90']:>8.4f} {r['I_P99']:>8.4f} "
              f"{r['dh_mean']:>10.4f}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    t_global = time.time()

    print("=" * 65)
    print("Δ-Sensitivity Analysis — Spain QKD Network")
    print("Rebuilding graph from coordinates for each threshold")
    print("=" * 65)

    # Load coordinates once
    print(f"\nLoading coordinates from: {COORDS_CSV}")
    coords = load_coords()
    print(f"  Loaded {len(coords)} municipalities")

    delta_values = [35, 40, 45, 50]
    sensitivity_results = []   # for JSON (matches table5_sensitivity_delta format)
    table5_rows = []           # for pretty-print

    for delta in delta_values:
        print(f"\n{'='*50}")
        print(f"  Δ = {delta} km")
        print(f"{'='*50}")

        # Build graph
        t0 = time.time()
        G = build_graph_for_delta(coords, delta_km=delta)
        t_build = time.time() - t0
        n_nodes_full = G.number_of_nodes()
        n_edges_full = G.number_of_edges()
        print(f"  Graph built in {t_build:.1f}s — |V|={n_nodes_full}, |E|={n_edges_full}",
              flush=True)

        # Remove isolated nodes (nodes with no edges — they don't participate)
        isolated = [v for v in G.nodes() if G.degree(v) == 0]
        if isolated:
            print(f"  Removing {len(isolated)} isolated nodes (degree=0)", flush=True)
            G.remove_nodes_from(isolated)

        # Check connectivity
        comps = list(nx.connected_components(G))
        is_connected = (len(comps) == 1)
        n_nodes_active = G.number_of_nodes()

        # Compute all pairs
        df, G_used = compute_all_pairs(G, label=f"Δ={delta}")

        # After compute_all_pairs G_used may be restricted to LCC
        n_nodes_used = G_used.number_of_nodes()
        n_edges_used = G_used.number_of_edges()

        # Aggregate stats
        s = aggregate_stats(df, label=str(delta))
        s['param'] = 'delta'
        s['value'] = delta

        # Build table5 display row
        row = {
            'delta':      delta,
            'n_nodes':    n_nodes_used,
            'n_edges':    n_edges_used,
            'connected':  is_connected,
            'lcc_size':   n_nodes_used,
            'n_pairs':    s['n_pairs'],
            'I_mean':     s['I_mean'],
            'I_median':   s['I_median'],
            'I_std':      s['I_std'],
            'I_P90':      s['I_P90'],
            'I_P99':      s['I_P99'],
            'I_max':      s['I_max'],
            'dh_mean':    s['dh_mean'],
            'dh_median':  s['dh_median'],
            'dh_std':     s['dh_std'],
            'dh_max':     s['dh_max'],
        }
        table5_rows.append(row)

        # Enrich stats for JSON output
        s['n_nodes_full']   = n_nodes_full    # before removing isolated
        s['n_edges_full']   = n_edges_full
        s['n_nodes_active'] = n_nodes_used    # after LCC selection
        s['n_edges_active'] = n_edges_used
        s['connected']      = is_connected
        sensitivity_results.append(s)

        print(f"\n  -> I_mean={s['I_mean']:.4f}, dh_mean={s['dh_mean']:.4f}")

    # Print Table 5
    print_table5(table5_rows)

    # ---- Update JSON ----
    print(f"\nUpdating JSON: {JSON_OUT}")
    if os.path.exists(JSON_OUT):
        with open(JSON_OUT, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    else:
        json_data = {}

    json_data['table5_sensitivity_delta'] = sensitivity_results
    json_data['table5_sensitivity_delta_meta'] = {
        'generated':   time.strftime('%Y-%m-%dT%H:%M:%S'),
        'description': (
            'Graph rebuilt from peninsula_1000.csv coordinates for each Δ. '
            'Edge exists iff haversine(u,v) <= Δ km. '
            'SKR uses fibre_km = haversine_km * 1.25 with BB84 decoy model. '
            'If disconnected, computation uses the largest connected component.'
        ),
    }

    with open(JSON_OUT, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  Saved: {JSON_OUT}")

    t_total = time.time() - t_global
    print(f"\nTotal runtime: {t_total:.1f}s")
    print("Done.")
