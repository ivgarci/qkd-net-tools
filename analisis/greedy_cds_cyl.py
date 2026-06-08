#!/usr/bin/env python3
"""
greedy_cds_cyl.py — Greedy MCDS baseline for CyL (P2, QKD-PAM-Generation paper)

Implements a greedy connected dominating set heuristic on G_Δ (CyL, n=267, Δ=45km):
  1. Greedy MDS (set cover): iteratively pick the node covering most uncovered nodes.
  2. Steiner connection: BFS-based path repair to connect the dominating set.

Run with:
    python greedy_cds_cyl.py
"""

import os, time, math
import pandas as pd
import networkx as nx
from haversine import haversine, Unit

BASE    = os.path.dirname(os.path.abspath(__file__))
CSV_CYL = os.path.join(BASE, '..', 'datos', 'cyl', 'cyl_1000.csv')
DELTA   = 45.0
PAM_K   = 100

# ---------------------------------------------------------------------------
def load_cyl():
    df = pd.read_csv(CSV_CYL, sep=';', decimal='.', encoding='utf-8-sig')
    names  = list(df['Población'])
    coords = list(zip(df['Latitud'], df['Longitud']))
    return names, coords

def build_graph(names, coords, delta):
    n = len(names)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            d = haversine(coords[i], coords[j], unit=Unit.KILOMETERS)
            if d <= delta:
                G.add_edge(i, j, weight=d)
    return G

# ---------------------------------------------------------------------------
# Step 1: Greedy minimum dominating set (set-cover greedy)
# ---------------------------------------------------------------------------
def greedy_mds(G):
    n  = G.number_of_nodes()
    covered = set()
    D = set()
    closed_nbrs = {v: set(G.neighbors(v)) | {v} for v in range(n)}

    while len(covered) < n:
        best, best_score = -1, -1
        for v in range(n):
            score = len(closed_nbrs[v] - covered)
            if score > best_score:
                best_score, best = score, v
        if best_score == 0:
            break
        D.add(best)
        covered |= closed_nbrs[best]

    return D

# ---------------------------------------------------------------------------
# Step 2: Connect D via BFS shortest paths (Steiner connection)
# ---------------------------------------------------------------------------
def connect_ds(G, D):
    """Add minimum-length paths between connected components of G[D]."""
    D2 = set(D)
    while True:
        sub   = G.subgraph(list(D2))
        comps = list(nx.connected_components(sub))
        if len(comps) == 1:
            break
        best_path, best_len = None, math.inf
        c0 = list(comps[0])
        for ci in comps[1:]:
            for u in c0:
                for v in ci:
                    try:
                        p = nx.shortest_path(G, u, v)
                        if len(p) < best_len:
                            best_len, best_path = len(p), p
                    except nx.NetworkXNoPath:
                        pass
        if best_path is None:
            print("  WARNING: graph may be disconnected; cannot connect all components.")
            break
        for node in best_path:
            D2.add(node)
    return D2

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify(G, D):
    n        = G.number_of_nodes()
    sol      = set(D)
    dom_ok   = all(sol & (set(G.neighbors(v)) | {v}) for v in range(n))
    conn_ok  = nx.is_connected(G.subgraph(list(sol)))
    return dom_ok, conn_ok

# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Greedy MCDS Baseline — CyL, Δ=45 km")
    print("=" * 60)

    names, coords = load_cyl()
    n = len(names)
    print(f"\n[1] n = {n} municipalities")

    t0 = time.time()
    G  = build_graph(names, coords, DELTA)
    print(f"[2] G_Δ: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}, "
          f"built in {time.time()-t0:.1f}s")

    t0   = time.time()
    mds  = greedy_mds(G)
    t_mds = time.time() - t0
    d_ok, _ = verify(G, mds)
    print(f"\n[3] Greedy MDS: k_MDS = {len(mds)}, dominated={d_ok}, {t_mds:.2f}s")

    t0    = time.time()
    mcds  = connect_ds(G, mds)
    t_cds = time.time() - t0
    d_ok2, c_ok2 = verify(G, mcds)
    print(f"[4] Steiner connection: k_MCDS = {len(mcds)}, "
          f"dominated={d_ok2}, connected={c_ok2}, {t_cds:.2f}s")

    gap = (PAM_K - len(mcds)) / len(mcds) * 100
    print("\n" + "=" * 60)
    print(f"  k_greedy_MDS  = {len(mds)}")
    print(f"  k_greedy_MCDS = {len(mcds)}")
    print(f"  k*_PAM        = {PAM_K}")
    print(f"  PAM gap vs greedy MCDS: {gap:+.1f}%")
    print("=" * 60)
    print(f"\n  FOR PAPER:")
    print(f"    Greedy MCDS heuristic: k = {len(mcds)} "
          f"(dominated={d_ok2}, connected={c_ok2})")
    print(f"    Runtime: {t_mds+t_cds:.2f}s total")
    print(f"    PAM excess: {PAM_K - len(mcds)} relays (+{gap:.1f}%)")

if __name__ == "__main__":
    main()
