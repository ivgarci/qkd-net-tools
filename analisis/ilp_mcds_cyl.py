#!/usr/bin/env python3
"""
ilp_mcds_cyl.py — MCDS ILP lower bound for CyL (S4, QKD-PAM-Generation paper)

Computes:
  1. LP relaxation of the domination-only problem  → valid lower bound k_LP
  2. ILP minimum dominating set (no connectivity)  → k_DS  (intermediate)
  3. ILP minimum *connected* dominating set (MCDS) → k_MCDS (exact k*)
  4. Gap analysis vs PAM result k*_PAM = 100

Reference network: CyL, n=267 municipalities, Δ=45 km (haversine).

Requires: pulp  (pip install pulp)
Run with: python ilp_mcds_cyl.py
"""

import os
import sys
import time
import math
import pandas as pd
import numpy as np
import networkx as nx
from haversine import haversine, Unit
import pulp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE     = os.path.dirname(os.path.abspath(__file__))
CSV_CYL  = os.path.join(BASE, '..', 'datos', 'cyl', 'cyl_1000.csv')
DELTA    = 45.0   # km
ETA      = 1.25   # routing factor mid-point (reference value, not used in graph)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_cyl():
    df = pd.read_csv(CSV_CYL, sep=';', decimal='.', encoding='utf-8-sig')
    names  = list(df['Población'])
    coords = list(zip(df['Latitud'], df['Longitud']))
    return names, coords

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph(names, coords, delta):
    """G_Δ: undirected, edge iff haversine(u,v) ≤ delta km."""
    n = len(names)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(coords[i], coords[j], unit=Unit.KILOMETERS)
            if d <= delta:
                G.add_edge(i, j, weight=d)
    return G

# ---------------------------------------------------------------------------
# Problem 1: LP relaxation of domination constraints
#   min  Σ x_v
#   s.t. Σ_{u ∈ N[v]} x_u ≥ 1   ∀v
#        0 ≤ x_v ≤ 1             ∀v
# ---------------------------------------------------------------------------
def solve_lp_relaxation(G):
    n = G.number_of_nodes()
    prob = pulp.LpProblem("MCDS_LP_relaxation", pulp.LpMinimize)

    x = [pulp.LpVariable(f"x_{v}", lowBound=0, upBound=1) for v in range(n)]

    prob += pulp.lpSum(x)

    for v in range(n):
        nbrs = list(G.neighbors(v)) + [v]
        prob += pulp.lpSum(x[u] for u in nbrs) >= 1, f"dom_{v}"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return pulp.value(prob.objective)

# ---------------------------------------------------------------------------
# Problem 2: ILP minimum dominating set (no connectivity)
#   same as LP but x_v ∈ {0,1}
# ---------------------------------------------------------------------------
def solve_mds_ilp(G):
    n = G.number_of_nodes()
    prob = pulp.LpProblem("MDS_ILP", pulp.LpMinimize)

    x = [pulp.LpVariable(f"x_{v}", cat='Binary') for v in range(n)]

    prob += pulp.lpSum(x)

    for v in range(n):
        nbrs = list(G.neighbors(v)) + [v]
        prob += pulp.lpSum(x[u] for u in nbrs) >= 1, f"dom_{v}"

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))
    status = pulp.LpStatus[prob.status]
    obj    = pulp.value(prob.objective)
    sol    = [v for v in range(n) if pulp.value(x[v]) and pulp.value(x[v]) > 0.5]
    return obj, status, sol

# ---------------------------------------------------------------------------
# Problem 3: ILP MCDS — domination + single-commodity flow connectivity
#
#   Fix root r (highest-degree node in G).
#   For each directed arc (u→v), d(u,v) ≤ Δ:
#     f[u,v] ≥ 0
#
#   (C1) domination:  Σ_{u∈N[v]} x_u ≥ 1          ∀v
#   (C2) root forced: x_r = 1
#   (C3) flow balance at root:
#           Σ_v f[r,v] - Σ_v f[v,r] = Σ_{v≠r} x_v
#   (C4) flow balance at non-root v:
#           Σ_u f[u,v] - Σ_w f[v,w] = x_v          ∀v ≠ r
#   (C5) capacity:  f[u,v] ≤ (n-1)·x_u             ∀(u,v)
#        capacity:  f[u,v] ≤ (n-1)·x_v             ∀(u,v)
# ---------------------------------------------------------------------------
def solve_mcds_ilp(G, time_limit=300):
    n  = G.number_of_nodes()
    r  = max(range(n), key=lambda v: G.degree(v))

    edges_undir = list(G.edges())
    arcs = [(u, v) for (u, v) in edges_undir] + [(v, u) for (u, v) in edges_undir]

    prob = pulp.LpProblem("MCDS_ILP", pulp.LpMinimize)

    x = [pulp.LpVariable(f"x_{v}", cat='Binary') for v in range(n)]
    f = {(u, v): pulp.LpVariable(f"f_{u}_{v}", lowBound=0) for (u, v) in arcs}

    prob += pulp.lpSum(x)

    for v in range(n):
        nbrs = list(G.neighbors(v)) + [v]
        prob += pulp.lpSum(x[u] for u in nbrs) >= 1, f"dom_{v}"

    prob += x[r] == 1, "root_selected"

    out_r = [v for (u, v) in arcs if u == r]
    in_r  = [u for (u, v) in arcs if v == r]
    prob += (pulp.lpSum(f[(r, v)] for v in out_r)
             - pulp.lpSum(f[(u, r)] for u in in_r)
             == pulp.lpSum(x[v] for v in range(n) if v != r)), "flow_root"

    for v in range(n):
        if v == r:
            continue
        out_v = [w for (u, w) in arcs if u == v]
        in_v  = [u for (u, w) in arcs if w == v]
        prob += (pulp.lpSum(f[(u, v)] for u in in_v)
                 - pulp.lpSum(f[(v, w)] for w in out_v)
                 == x[v]), f"flow_bal_{v}"

    for (u, v) in arcs:
        prob += f[(u, v)] <= (n - 1) * x[u], f"cap_src_{u}_{v}"
        prob += f[(u, v)] <= (n - 1) * x[v], f"cap_dst_{u}_{v}"

    solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=time_limit)
    prob.solve(solver)

    pulp_status = pulp.LpStatus[prob.status]
    obj  = pulp.value(prob.objective)
    sol  = [v for v in range(n) if pulp.value(x[v]) and pulp.value(x[v]) > 0.5]
    status = pulp_status

    return obj, status, sol, r

# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------
def verify_domination(G, sol, delta=None):
    sol_set = set(sol)
    n = G.number_of_nodes()
    uncovered = []
    for v in range(n):
        nbrs = set(G.neighbors(v)) | {v}
        if not nbrs & sol_set:
            uncovered.append(v)
    return len(uncovered) == 0, uncovered

def verify_connected(G, sol):
    sub = G.subgraph(sol)
    return nx.is_connected(sub)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    PAM_K_STAR = 100

    print("=" * 60)
    print("  MCDS ILP Lower Bound — CyL, Δ=45 km  (S4)")
    print("=" * 60)

    print("\n[1] Loading CyL data ...")
    names, coords = load_cyl()
    n = len(names)
    print(f"    n = {n} municipalities")

    print(f"\n[2] Building G_Δ (Δ={DELTA} km) ...")
    t0 = time.time()
    G  = build_graph(names, coords, DELTA)
    t1 = time.time()
    print(f"    |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}, "
          f"connected={nx.is_connected(G)}, built in {t1-t0:.1f}s")
    degs = [d for _, d in G.degree()]
    print(f"    degree: min={min(degs)}, avg={sum(degs)/n:.1f}, max={max(degs)}")

    print("\n[3] Solving LP relaxation (domination only) ...")
    t0 = time.time()
    k_lp = solve_lp_relaxation(G)
    t1 = time.time()
    print(f"    k_LP = {k_lp:.3f}  (lower bound, {t1-t0:.1f}s)")

    print("\n[4] Solving MDS ILP (domination, no connectivity) ...")
    t0 = time.time()
    k_ds, st_ds, sol_ds = solve_mds_ilp(G)
    t1 = time.time()
    dom_ok, _ = verify_domination(G, sol_ds)
    print(f"    k_DS = {int(round(k_ds))}  status={st_ds}, "
          f"dominated={dom_ok}, {t1-t0:.1f}s")

    print("\n[5] Solving MCDS ILP (domination + flow connectivity) ...")
    print("    (time limit: 300 s)")
    t0 = time.time()
    k_mcds, st_mcds, sol_mcds, root = solve_mcds_ilp(G, time_limit=300)
    t1 = time.time()

    if sol_mcds:
        dom_ok2, _ = verify_domination(G, sol_mcds)
        con_ok2    = verify_connected(G, sol_mcds)
    else:
        dom_ok2 = con_ok2 = False

    print(f"    k_MCDS = {int(round(k_mcds)) if k_mcds else '?'}  "
          f"status={st_mcds}, dominated={dom_ok2}, connected={con_ok2}, "
          f"root='{names[root]}', {t1-t0:.1f}s")

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  LP relaxation lower bound  k_LP   = {k_lp:.2f}")
    print(f"  MDS ILP (no connectivity)  k_DS   = {int(round(k_ds))}")
    if k_mcds:
        k_mcds_int = int(round(k_mcds))
        print(f"  MCDS ILP (exact, status={st_mcds}) k_MCDS = {k_mcds_int}")
        gap_pam  = (PAM_K_STAR - k_mcds_int) / k_mcds_int * 100
        gap_lp   = (PAM_K_STAR - k_lp)       / k_lp       * 100
        print(f"  PAM result                 k*_PAM = {PAM_K_STAR}")
        print(f"\n  PAM suboptimality gap (vs MCDS): {gap_pam:+.1f}%")
        print(f"  PAM suboptimality gap (vs LP):   {gap_lp:+.1f}%")
        if gap_pam <= 0:
            print("    PAM IS optimal (k*_PAM = k_MCDS).")
        elif gap_pam < 5:
            print(f"    PAM is within {gap_pam:.1f}% of the exact optimum — near-optimal.")
        else:
            print(f"    PAM uses {gap_pam:.1f}% more relays than strictly necessary.")
    else:
        print(f"  MCDS ILP: no feasible solution found within time limit")
        print(f"  PAM result k*_PAM = {PAM_K_STAR}")
        print(f"  LP lower bound: gap ≥ {(PAM_K_STAR - k_lp)/k_lp*100:.1f}%")

    print("\n" + "-" * 60)
    print("  FOR PAPER (Table / Section wording):")
    print(f"    LP relaxation: k_LP = {k_lp:.1f}")
    print(f"    MDS lower bound (no conn.): k_DS = {int(round(k_ds))}")
    if k_mcds and st_mcds in ('Optimal', 'Integer optimal'):
        k_mcds_int = int(round(k_mcds))
        gap = (PAM_K_STAR - k_mcds_int) / k_mcds_int * 100
        print(f"    Exact MCDS: k* = {k_mcds_int} (CBC, status=Optimal)")
        print(f"    PAM gap:    {gap:.1f}%  (= {PAM_K_STAR - k_mcds_int} extra relays)")
    print("-" * 60)


if __name__ == "__main__":
    main()
