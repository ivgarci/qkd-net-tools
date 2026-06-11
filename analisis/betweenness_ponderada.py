"""
EXP-9 — Criticidad (betweenness) ponderada por SKR — Tarea A.

Para cada red (cyl, espana, adif), con los grafos del script canónico
analisis/capacidad_servicio_ataques.py (atributo 'skr' por arista):

  1. C_B no ponderada: nx.betweenness_centrality (normalizada).
  2. C_B ponderada por SKR: misma función con weight = ℓ_e, donde la
     longitud QKD de cada arista es ℓ_e = 1/SKR_e. Para aristas con
     SKR_e = 0 (existen 7 en ADIF, corredores >50 km fuera de rango
     BB84+decoy) se usa ℓ_e = 1/max(SKR_e, 1e-12), de modo que esas
     aristas son "casi infinitamente largas" sin romper el cálculo.
  3. Comparación de rankings: overlap top-10 y top-25 (|∩|/k) y
     Kendall-τ (scipy.stats.kendalltau) sobre los valores de
     centralidad de todos los nodos.

Salidas:
  datos/resultados_papers/betweenness_ponderada.csv
      red, nodo, nombre, cb_unweighted, cb_weighted,
      rank_unweighted, rank_weighted
  datos/resultados_papers/betweenness_ponderada_resumen.csv
      red, n, m, n_aristas_skr0, overlap_top10, overlap_top25,
      kendall_tau, p_value, top1_unweighted, top1_weighted

Log: /Users/igarcia/doctorado/2025_2026/experimentos/exp9_betweenness_ponderada.log

Uso:
    cd /Users/igarcia/doctorado/2025_2026/codigo/qkd-net-tools
    python analisis/betweenness_ponderada.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import kendalltau

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# Reutilizamos carga de grafos (con 'skr' por arista) y logging del
# script canónico EXP-8 — no se duplica lógica de construcción.
import capacidad_servicio_ataques as csa  # noqa: E402

OUT_DIR = csa.OUT_DIR
LOG_PATH = os.path.join(csa.LOG_DIR, 'exp9_betweenness_ponderada.log')

SKR_MIN = 1e-12     # suelo para ℓ_e = 1/max(SKR_e, SKR_MIN) si SKR_e = 0
REDES = ('cyl', 'espana', 'adif')


def top_k(cent, k):
    """Conjunto de los k nodos más centrales (desempate por id, determinista)."""
    return set(csa._ranking_estatico(cent)[:k])


def main():
    csa.LOG_FH = open(LOG_PATH, 'w')
    log = csa.log
    t_ini = time.time()

    log("=" * 70)
    log("EXP-9 — Betweenness ponderada por SKR (ℓ_e = 1/max(SKR_e, 1e-12))")
    log("=" * 70)

    rows_nodos, rows_resumen = [], []

    for red in REDES:
        t0 = time.time()
        G = csa.CARGADORES[red]()
        n, m = G.number_of_nodes(), G.number_of_edges()

        # Longitud QKD por arista; suelo 1e-12 para las aristas con SKR = 0
        n_skr0 = 0
        for u, v, d in G.edges(data=True):
            if d['skr'] <= 0.0:
                n_skr0 += 1
            G[u][v]['ell'] = 1.0 / max(d['skr'], SKR_MIN)
        log(f"--- {red}: |V|={n}, |E|={m}, aristas con SKR=0: {n_skr0} "
            f"(ℓ_e = 1/max(SKR_e, {SKR_MIN:g}))")

        t1 = time.time()
        cb_u = nx.betweenness_centrality(G, normalized=True)
        t2 = time.time()
        cb_w = nx.betweenness_centrality(G, normalized=True, weight='ell')
        t3 = time.time()
        log(f"    C_B no ponderada: {t2 - t1:.1f} s — "
            f"C_B ponderada: {t3 - t2:.1f} s")

        nodos = sorted(G.nodes(), key=str)
        vals_u = np.array([cb_u[x] for x in nodos])
        vals_w = np.array([cb_w[x] for x in nodos])

        # rangos (1 = más central; empates → rango mínimo)
        rank_u = pd.Series(vals_u).rank(ascending=False, method='min')
        rank_w = pd.Series(vals_w).rank(ascending=False, method='min')

        for i, x in enumerate(nodos):
            rows_nodos.append({
                'red': red,
                'nodo': x,
                'nombre': G.nodes[x].get('nombre', x),
                'cb_unweighted': vals_u[i],
                'cb_weighted': vals_w[i],
                'rank_unweighted': int(rank_u.iloc[i]),
                'rank_weighted': int(rank_w.iloc[i]),
            })

        ov10 = len(top_k(cb_u, 10) & top_k(cb_w, 10)) / 10.0
        ov25 = len(top_k(cb_u, 25) & top_k(cb_w, 25)) / 25.0
        tau, pval = kendalltau(vals_u, vals_w)

        top1_u = csa._ranking_estatico(cb_u)[0]
        top1_w = csa._ranking_estatico(cb_w)[0]
        nom_u = G.nodes[top1_u].get('nombre', top1_u)
        nom_w = G.nodes[top1_w].get('nombre', top1_w)

        rows_resumen.append({
            'red': red, 'n': n, 'm': m, 'n_aristas_skr0': n_skr0,
            'overlap_top10': ov10, 'overlap_top25': ov25,
            'kendall_tau': tau, 'p_value': pval,
            'top1_unweighted': nom_u, 'top1_weighted': nom_w,
        })

        log(f"    overlap top-10 = {ov10:.2f}, top-25 = {ov25:.2f}, "
            f"Kendall-τ = {tau:.4f} (p = {pval:.3e})")
        log(f"    top-1 no ponderado: {nom_u} (C_B = {cb_u[top1_u]:.4f}) — "
            f"top-1 ponderado: {nom_w} (C_B = {cb_w[top1_w]:.4f})")
        log(f"    {red} completado en {time.time() - t0:.1f} s")

    out1 = os.path.join(OUT_DIR, 'betweenness_ponderada.csv')
    out2 = os.path.join(OUT_DIR, 'betweenness_ponderada_resumen.csv')
    pd.DataFrame(rows_nodos).to_csv(out1, index=False)
    pd.DataFrame(rows_resumen).to_csv(out2, index=False)
    log(f"Guardado: {os.path.abspath(out1)} ({len(rows_nodos)} filas)")
    log(f"Guardado: {os.path.abspath(out2)} ({len(rows_resumen)} filas)")
    log(f"EXP-9 terminado en {time.time() - t_ini:.1f} s")
    csa.LOG_FH.close()


if __name__ == '__main__':
    main()
