"""
EXP-10 — Null model Erdős–Rényi G(n,m) con control de densidad — Tarea B.

Para cada red real (cyl, espana, adif) con n nodos y m aristas:
  1. 50 realizaciones E-R G(n, m) (nx.gnm_random_graph, semillas 0..49).
  2. SKR por arista de cada realización: re-muestreo con reemplazo de la
     distribución empírica de SKR de la red real (np.random.default_rng
     con semilla = la de la realización).
  3. Dos protocolos de ataque del script canónico
     analisis/capacidad_servicio_ataques.py (se REUTILIZAN sus funciones
     orden_adaptativo, medir_curva, bottlenecks_pares, calcular_umbrales;
     eliminación de un nodo cada vez recalculando la centralidad, idéntico
     al experimento sobre las redes reales): degree_adaptive y
     betweenness_adaptive, p = 0 … 0.50.
       - S(p) = LCC/n en pasos del 1 % (P_GRID canónico).
       - C(p) = mediana del bottleneck widest-path (SKR) sobre 500 pares
         fijos (semilla 42, mismo muestreo que muestrear_pares canónico);
         par desconectado o con extremo eliminado → 0. Solo en
         p ∈ {0, 0.05, …, 0.50}.
  4. Media ± std sobre realizaciones: S̄(p), C̄_median(p), y umbrales
     p̄*_topo (primer p con S < 0.5) y p̄*_svc (primer p con
     C_median < 0.5·C_median(0)). Realizaciones sin umbral en [0, 0.5]
     (NaN) se excluyen de media/std (se reporta cuántas).
  5. Se cronometra la realización 0 de España antes del bucle: si la
     proyección total de España con MAX_WORKERS procesos supera 4 h se
     reduce España a 20 realizaciones (cyl/adif mantienen 50).

Salidas:
  datos/resultados_papers/null_model_er.csv
      red, protocolo, p, S_mean, S_std, C_median_mean, C_median_std
  datos/resultados_papers/null_model_er_umbrales.csv
      red, protocolo, n_realizaciones, p_star_topo_mean, p_star_topo_std,
      p_star_svc_mean, p_star_svc_std, n_sin_p_topo, n_sin_p_svc

Log: ``logs/exp10_null_model_er.log`` por defecto.

Uso:
    python analisis/null_model_er.py
"""

import os
import sys
import time
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd
import networkx as nx

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# Funciones canónicas EXP-8: carga de redes reales, protocolos de ataque,
# motor widest-path (bosque de máxima expansión) y umbrales.
import capacidad_servicio_ataques as csa  # noqa: E402

OUT_DIR = csa.OUT_DIR
LOG_PATH = os.path.join(csa.LOG_DIR, 'exp10_null_model_er.log')

REDES = ('cyl', 'espana', 'adif')
N_REAL = 50                  # realizaciones por red (España puede bajar a 20)
N_REAL_ESP_REDUCIDO = 20
N_PARES = 500                # pares fijos para C(p)
SEED_PARES = 42
P_C = set(round(0.05 * j, 2) for j in range(11))   # C(p): 0, 0.05, ..., 0.50
MAX_WORKERS = 8
BUDGET_ESP_S = 4 * 3600      # presupuesto España: 4 h

log = csa.log


def muestrear_pares_er(n, n_pares=N_PARES, seed=SEED_PARES):
    """
    500 pares fijos sobre los nodos 0..n-1 de las realizaciones E-R.
    Mismo procedimiento que csa.muestrear_pares (combinations ordenadas +
    choice sin reemplazo, semilla 42), sin escribir CSV.
    """
    todos = list(combinations(range(n), 2))
    rng = np.random.default_rng(seed)
    if n_pares >= len(todos):
        return todos
    idx = rng.choice(len(todos), size=n_pares, replace=False)
    return [todos[i] for i in sorted(idx)]


def run_realization(args):
    """
    Una realización E-R G(n,m) con SKR re-muestreado: dos protocolos
    adaptativos canónicos. Devuelve curvas S/C y umbrales por protocolo.
    """
    red, seed, n, m, skr_emp, pares = args
    t0 = time.time()

    G0 = nx.gnm_random_graph(n, m, seed=seed)
    rng = np.random.default_rng(seed)
    skr_vals = rng.choice(skr_emp, size=m, replace=True)
    for (u, v), s in zip(G0.edges(), skr_vals):
        G0[u][v]['skr'] = float(s)
        G0[u][v]['SKR'] = float(s)

    k_max = csa.k_de_p(csa.P_GRID[-1], n)
    out = {'red': red, 'seed': seed}
    for proto, metrica in (('degree_adaptive', 'degree'),
                           ('betweenness_adaptive', 'betweenness')):
        orden = csa.orden_adaptativo(G0, metrica, k_max,
                                     etiqueta=f'{red} seed={seed}')
        filas = csa.medir_curva(G0, orden, pares, p_medir_c=P_C)
        df = pd.DataFrame(filas)
        p_topo, p_svc, c0 = csa.calcular_umbrales(df)
        out[proto] = {'S': df['S'].tolist(),
                      'C': {f['p']: f['C_median'] for f in filas
                            if f['p'] in P_C},
                      'p_star_topo': p_topo, 'p_star_svc': p_svc, 'c0': c0}
    out['t_s'] = time.time() - t0
    return out


def main():
    csa.LOG_FH = open(LOG_PATH, 'w')
    t_ini = time.time()

    log("=" * 70)
    log("EXP-10 — Null model Erdős–Rényi G(n,m) con SKR re-muestreado")
    log("=" * 70)
    log(f"Protocolos canónicos (orden_adaptativo: 1 nodo/paso con recálculo); "
        f"S(p) en P_GRID (paso 1%); C(p) mediana sobre {N_PARES} pares fijos "
        f"(semilla {SEED_PARES}) en p ∈ {{0, 0.05, ..., 0.50}}; "
        f"workers = {MAX_WORKERS}")

    rows_curvas, rows_umbrales = [], []

    for red in REDES:
        G_real = csa.CARGADORES[red]()
        n, m = G_real.number_of_nodes(), G_real.number_of_edges()
        skr_emp = np.array([d['skr'] for _, _, d in G_real.edges(data=True)])
        pares = muestrear_pares_er(n)
        log(f"--- {red}: E-R G({n},{m}); distribución empírica SKR: "
            f"{len(skr_emp)} valores en [{skr_emp.min():.3e}, "
            f"{skr_emp.max():.3e}] bits/pulso "
            f"(aristas SKR=0: {int((skr_emp <= 0).sum())})")

        n_real = N_REAL
        t_red = time.time()

        # Realización 0 en el proceso principal (cronometraje para España)
        r0 = run_realization((red, 0, n, m, skr_emp, pares))
        results = [r0]
        log(f"    realización 0: {r0['t_s']:.1f} s")
        if red == 'espana':
            proy = r0['t_s'] * N_REAL / MAX_WORKERS
            log(f"    proyección España: {N_REAL} realiz. × {r0['t_s']:.1f} s "
                f"/ {MAX_WORKERS} workers ≈ {proy / 3600:.2f} h")
            if proy > BUDGET_ESP_S:
                n_real = N_REAL_ESP_REDUCIDO
                log(f"    proyección > 4 h → España se reduce a {n_real} "
                    f"realizaciones (cyl/adif mantienen {N_REAL})")
            else:
                log(f"    proyección ≤ 4 h → se mantienen {N_REAL} realizaciones")

        tasks = [(red, s, n, m, skr_emp, pares) for s in range(1, n_real)]
        with Pool(processes=MAX_WORKERS) as pool:
            results.extend(pool.map(run_realization, tasks))
        results.sort(key=lambda r: r['seed'])
        log(f"    {len(results)} realizaciones completadas en "
            f"{time.time() - t_red:.1f} s (media por realización "
            f"{np.mean([r['t_s'] for r in results]):.1f} s)")

        for proto in ('degree_adaptive', 'betweenness_adaptive'):
            S_mat = np.array([r[proto]['S'] for r in results])
            for i, p in enumerate(csa.P_GRID):
                row = {'red': red, 'protocolo': proto, 'p': p,
                       'S_mean': S_mat[:, i].mean(),
                       'S_std': S_mat[:, i].std(ddof=0),
                       'C_median_mean': np.nan, 'C_median_std': np.nan}
                if p in P_C:
                    C_vals = np.array([r[proto]['C'][p] for r in results])
                    row['C_median_mean'] = C_vals.mean()
                    row['C_median_std'] = C_vals.std(ddof=0)
                rows_curvas.append(row)

            pt = np.array([r[proto]['p_star_topo'] for r in results], float)
            ps = np.array([r[proto]['p_star_svc'] for r in results], float)
            n_nan_t = int(np.isnan(pt).sum())
            n_nan_s = int(np.isnan(ps).sum())
            rows_umbrales.append({
                'red': red, 'protocolo': proto,
                'n_realizaciones': len(results),
                'p_star_topo_mean': np.nanmean(pt),
                'p_star_topo_std': np.nanstd(pt, ddof=0),
                'p_star_svc_mean': np.nanmean(ps),
                'p_star_svc_std': np.nanstd(ps, ddof=0),
                'n_sin_p_topo': n_nan_t, 'n_sin_p_svc': n_nan_s,
            })
            log(f"    {proto}: p̄*_topo = {np.nanmean(pt):.4f} ± "
                f"{np.nanstd(pt):.4f} (sin umbral: {n_nan_t}) — "
                f"p̄*_svc = {np.nanmean(ps):.4f} ± {np.nanstd(ps):.4f} "
                f"(sin umbral: {n_nan_s}); "
                f"C̄(0) = {np.mean([r[proto]['c0'] for r in results]):.4e}")

    out1 = os.path.join(OUT_DIR, 'null_model_er.csv')
    out2 = os.path.join(OUT_DIR, 'null_model_er_umbrales.csv')
    pd.DataFrame(rows_curvas).to_csv(out1, index=False)
    pd.DataFrame(rows_umbrales).to_csv(out2, index=False)
    log(f"Guardado: {os.path.abspath(out1)} ({len(rows_curvas)} filas)")
    log(f"Guardado: {os.path.abspath(out2)} ({len(rows_umbrales)} filas)")
    log(f"EXP-10 terminado en {(time.time() - t_ini) / 60:.1f} min")
    csa.LOG_FH.close()


if __name__ == '__main__':
    main()
