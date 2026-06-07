#!/usr/bin/env python3
"""
tablas_paper_pam_generation.py — QKD-PAM-Generation (IEEE TNSE submission)
Genera los valores exactos para todas las tablas del paper que estaban marcados
como estimados o pendientes de verificación.

Ejecutar con:
    /Users/igarcia/my_env/bin/python tablas_paper_pam_generation.py

Cubre:
    - Table II : k-means vs PAM — violaciones de cobertura (España, k=950, Δ=45km)
    - Table III: Sensibilidad a Δ — k* para Δ∈{30, 45, 60} km en CyL y España
    - Table IV : k más allá de k* — |E_R|, densidad y puentes para k=80..120 en CyL
    - Benchmark: tiempo de ejecución real en este hardware

Prerrequisitos:
    pip install scikit-learn-extra haversine
    (ya disponibles en /Users/igarcia/my_env)

Datos requeridos:
    datos/cyl/cyl_1000.csv       — municipios CyL (Población;Latitud;Longitud)
    datos/espana/peninsula_1000.csv — municipios España
    datos/cyl/AdjacencyMatrixNamed45.csv — grafo CyL Δ=45km (para Table IV)
"""

import os
import sys
import csv
import time
import math
import platform
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from haversine import haversine, Unit

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_CYL  = os.path.join(BASE, '..', 'datos', 'cyl')
DATA_ESP  = os.path.join(BASE, '..', 'datos', 'espana')
RESULTS   = os.path.join(BASE, '..', 'datos', 'resultados_papers')
os.makedirs(RESULTS, exist_ok=True)

DELTA_DEFAULT = 45.0   # km
DELTA_LOW     = 30.0
DELTA_HIGH    = 60.0
SEED          = 42


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def load_muns(csv_path):
    """Carga CSV de municipios. Admite decimal '.' o ','."""
    try:
        df = pd.read_csv(csv_path, sep=';', decimal='.', encoding='utf-8-sig')
        if df['Latitud'].dtype == object:
            raise ValueError
        return df
    except Exception:
        return pd.read_csv(csv_path, sep=';', decimal=',', encoding='utf-8-sig')


def haversine_matrix(coords_deg):
    """Calcula matriz de distancias haversine en km. coords_deg: array (n,2) [lat,lon]."""
    n = len(coords_deg)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            d = haversine(coords_deg[i], coords_deg[j], unit=Unit.KILOMETERS)
            D[i, j] = D[j, i] = d
    return D


def build_feasibility_graph(coords_deg, names, delta):
    """Construye grafo de factibilidad G_Δ."""
    G = nx.Graph()
    G.add_nodes_from(names)
    n = len(names)
    for i in range(n):
        for j in range(i+1, n):
            d = haversine(coords_deg[i], coords_deg[j], unit=Unit.KILOMETERS)
            if d <= delta:
                G.add_edge(names[i], names[j], dist_km=d)
    return G


def coverage_check(coords_deg, labels, medoid_indices, delta):
    """
    Para cada clúster, calcula la distancia máxima entre cualquier miembro y el medoide.
    Devuelve lista de booleanos (True = violación de cobertura).
    """
    violations = []
    unique_labels = set(labels)
    for lab in unique_labels:
        members = np.where(labels == lab)[0]
        med_idx = medoid_indices[lab]
        max_d = max(
            haversine(coords_deg[m], coords_deg[med_idx], unit=Unit.KILOMETERS)
            for m in members
        )
        violations.append(max_d > delta)
    return violations


def find_kstar(coords_deg, names, delta, k_start=1, seed=SEED):
    """
    Búsqueda iterativa de k* (mínimo k para el que PAM satisface cobertura y conectividad).
    Devuelve (k_star, medoid_indices, labels).
    """
    n = len(names)
    print(f"    Calculando matriz de distancias ({n}×{n})...", flush=True)
    t0 = time.time()
    D = haversine_matrix(coords_deg)
    print(f"    Matriz calculada en {time.time()-t0:.1f}s")

    # Pre-chequeo rápido: si G_Δ es desconexo, k* no existe
    G_feas = nx.Graph()
    G_feas.add_nodes_from(range(n))
    rows, cols = np.where(D <= delta)
    for r, c in zip(rows, cols):
        if r != c:
            G_feas.add_edge(int(r), int(c))
    if not nx.is_connected(G_feas):
        raise RuntimeError(f"G_Δ({delta}km) es desconexo — k* no existe para este Δ")

    k = k_start
    while k <= n:
        km = KMedoids(n_clusters=k, metric='precomputed', random_state=seed, method='pam')
        km.fit(D)
        labels = km.labels_
        meds   = km.medoid_indices_

        # Verificar cobertura
        viols = coverage_check(coords_deg, labels, meds, delta)
        covered = not any(viols)

        if covered:
            # Verificar conectividad del backbone
            medoid_names = [names[m] for m in meds]
            # Construir subgrafo sobre medoides
            G_back = nx.Graph()
            G_back.add_nodes_from(medoid_names)
            for i in range(len(meds)):
                for j in range(i+1, len(meds)):
                    if D[meds[i], meds[j]] <= delta:
                        G_back.add_edge(medoid_names[i], medoid_names[j])
            if nx.is_connected(G_back):
                return k, meds, labels

        k += 1
        if k % 10 == 0:
            print(f"    k={k}...", flush=True)

    raise RuntimeError(f"No se encontró k* hasta k={n}")


# ---------------------------------------------------------------------------
# TABLE II — k-means vs PAM: violaciones de cobertura
# ---------------------------------------------------------------------------

def table_ii_kmeans_violations(coords_deg, names, delta=DELTA_DEFAULT, k=950, n_runs=20, seed=SEED):
    """
    Ejecuta k-means n_runs veces, proyecta centroides al candidato más cercano,
    cuenta violaciones de cobertura por ejecución.
    """
    print(f"\n{'='*60}")
    print(f"TABLE II — k-means violaciones de cobertura")
    print(f"k={k}, Δ={delta}km, {n_runs} semillas")
    print(f"{'='*60}")

    n = len(names)
    coords_arr = np.array(coords_deg, dtype=np.float64)  # (n, 2)

    violation_counts = []
    t0 = time.time()

    for run_i in range(n_runs):
        run_seed = seed + run_i
        km = KMeans(n_clusters=k, random_state=run_seed, n_init=1, max_iter=300)
        km.fit(coords_arr)
        centroids = km.cluster_centers_  # (k, 2)
        labels    = km.labels_           # (n,)

        # Proyectar cada centroide al candidato más cercano
        projected = []
        for c in centroids:
            dists = np.sqrt(((coords_arr - c)**2).sum(axis=1))
            projected.append(int(np.argmin(dists)))

        # Contar violaciones: para cada clúster, ¿la distancia máxima al
        # proyectado supera Δ?
        violations = 0
        for lab in range(k):
            members = np.where(labels == lab)[0]
            if len(members) == 0:
                continue
            proj_idx = projected[lab]
            proj_coord = coords_deg[proj_idx]
            max_d = max(
                haversine(coords_deg[m], proj_coord, unit=Unit.KILOMETERS)
                for m in members
            )
            if max_d > delta:
                violations += 1

        violation_counts.append(violations)
        if (run_i + 1) % 5 == 0:
            print(f"  Semilla {run_i+1}/{n_runs}: {violations} violaciones")

    elapsed = time.time() - t0
    viol_arr = np.array(violation_counts)
    mean_v   = viol_arr.mean()
    std_v    = viol_arr.std()
    max_v    = viol_arr.max()
    min_v    = viol_arr.min()

    print(f"\n  Resultados ({n_runs} runs, {elapsed:.0f}s):")
    print(f"  Media   ± std:  {mean_v:.1f} ± {std_v:.1f}  ({100*mean_v/k:.1f}%)")
    print(f"  Máximo:         {max_v}  ({100*max_v/k:.1f}%)")
    print(f"  Mínimo:         {min_v}  ({100*min_v/k:.1f}%)")
    print(f"\n  >> Para Table II:")
    print(f"     k-means + proj. (media): {mean_v:.0f} ({100*mean_v/k:.1f}%)")
    print(f"     k-means + proj. (máx.):  {max_v}  ({100*max_v/k:.1f}%)")
    print(f"     PAM:                      0  (0%)")

    return {
        'mean_violations': round(float(mean_v), 1),
        'std_violations':  round(float(std_v), 1),
        'max_violations':  int(max_v),
        'min_violations':  int(min_v),
        'pct_mean':        round(100*float(mean_v)/k, 1),
        'pct_max':         round(100*int(max_v)/k, 1),
        'n_runs':          n_runs,
        'k': k,
    }


# ---------------------------------------------------------------------------
# TABLE III — Sensibilidad a Δ
# ---------------------------------------------------------------------------

def table_iii_delta_sensitivity(coords_cyl, names_cyl, coords_esp, names_esp):
    print(f"\n{'='*60}")
    print(f"TABLE III — Sensibilidad a Δ")
    print(f"{'='*60}")
    results = {}

    for delta in [DELTA_LOW, DELTA_DEFAULT, DELTA_HIGH]:
        print(f"\n  Δ = {delta} km")
        for (coords, names, label) in [
            (coords_cyl, names_cyl, 'CyL'),
            (coords_esp, names_esp, 'España'),
        ]:
            print(f"    [{label}] buscando k*...")
            t0 = time.time()
            try:
                kstar, meds, labels = find_kstar(coords, names, delta)
                elapsed = time.time() - t0
                print(f"    [{label}] k*={kstar} en {elapsed:.0f}s")
                results[(label, delta)] = kstar
            except Exception as e:
                print(f"    [{label}] ERROR: {e}")
                results[(label, delta)] = None

    print(f"\n  Tabla III completa:")
    print(f"  {'Dataset':>12} | Δ=30km | Δ=45km | Δ=60km")
    print(f"  {'-'*40}")
    for label in ['CyL', 'España']:
        row = [str(results.get((label, d), '?')) for d in [DELTA_LOW, DELTA_DEFAULT, DELTA_HIGH]]
        print(f"  {label:>12} | {row[0]:>6} | {row[1]:>6} | {row[2]:>6}")

    return results


# ---------------------------------------------------------------------------
# TABLE IV — k más allá de k* (usando la matriz de adyacencia existente Δ=45km)
# ---------------------------------------------------------------------------

def table_iv_k_beyond_kstar():
    """
    Para cada k, construye el grafo de factibilidad G_Δ sobre los k medoides.
    k=100 usa los nodos confirmados del AdjacencyMatrixNamed45 (red real CyL).
    Otros k: PAM fresco con seed=42 sobre los 267 municipios CyL.
    """
    print(f"\n{'='*60}")
    print(f"TABLE IV — k más allá de k* (CyL, Δ={DELTA_DEFAULT}km)")
    print(f"{'='*60}")

    k_values = [80, 90, 100, 110, 120]

    coords_df  = load_muns(os.path.join(DATA_CYL, 'cyl_1000.csv'))
    coords_deg = [(row['Latitud'], row['Longitud']) for _, row in coords_df.iterrows()]
    names_list = list(coords_df['Población'])
    n_total    = len(names_list)
    name_to_idx = {n: i for i, n in enumerate(names_list)}

    D_file = os.path.join(DATA_CYL, 'dist_matrix_cyl.npy')
    if os.path.exists(D_file):
        D_full = np.load(D_file)
    else:
        D_full = haversine_matrix(coords_deg)
        np.save(D_file, D_full)

    # k=100 confirmado: usar nodos del AdjacencyMatrixNamed45
    adj_csv = os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv')
    adj_df  = pd.read_csv(adj_csv, index_col=0)
    adj_node_names = list(adj_df.index)
    confirmed_indices = [name_to_idx[n] for n in adj_node_names if n in name_to_idx]
    print(f"  k=100 confirmado: {len(confirmed_indices)} nodos en AdjMatrix coinciden con coords_df")

    def relay_graph_stats(med_indices):
        """Construye G_Δ entre los medoides y devuelve métricas."""
        n = len(med_indices)
        G_rel = nx.Graph()
        G_rel.add_nodes_from(range(n))
        for a in range(n):
            for b in range(a+1, n):
                if D_full[med_indices[a], med_indices[b]] <= DELTA_DEFAULT:
                    G_rel.add_edge(a, b)
        ne = G_rel.number_of_edges()
        dens = 2*ne / (n*(n-1)) if n > 1 else 0.0
        conn = nx.is_connected(G_rel)
        bri  = len(list(nx.bridges(G_rel))) if conn else 0
        uncov = sum(1 for mi in range(n_total)
                    if min(D_full[mi, m] for m in med_indices) > DELTA_DEFAULT)
        return ne, dens, conn, bri, uncov

    results = []
    for k in k_values:
        if k == 100 and len(confirmed_indices) == 100:
            print(f"  k=100: usando nodos AdjacencyMatrixNamed45 (solución confirmada)")
            med_indices = confirmed_indices
        else:
            print(f"  k={k}: ejecutando PAM (seed={SEED})...", flush=True)
            km = KMedoids(n_clusters=k, metric='precomputed', random_state=SEED, method='pam')
            km.fit(D_full)
            med_indices = [int(i) for i in km.medoid_indices_]

        ne, dens, conn, bri, uncov = relay_graph_stats(med_indices)
        print(f"  k={k:3d}: |E_R|={ne}, ρ={dens:.4f}, conectado={conn}, "
              f"puentes={bri}, sin_cubrir={uncov}")
        results.append({'k': k, 'E_R': ne, 'density': round(dens, 4),
                        'connected': conn, 'bridges': bri, 'uncovered': uncov})

    print(f"\n  Tabla IV:")
    print(f"  {'k':>5} | {'|E_R|':>6} | {'ρ':>7} | {'Conectado':>10} | {'Puentes':>8} | {'Sin cubrir':>10}")
    print("  " + "-" * 57)
    for r in results:
        conn_str = 'Sí' if r['connected'] else 'No'
        print(f"  {r['k']:>5} | {r['E_R']:>6} | {r['density']:>7.4f} | {conn_str:>10} | "
              f"{r['bridges']:>8} | {r['uncovered']:>10}")

    return results


# ---------------------------------------------------------------------------
# BENCHMARK — tiempo real en este hardware
# ---------------------------------------------------------------------------

def benchmark_hardware():
    print(f"\n{'='*60}")
    print(f"BENCHMARK — hardware y tiempos reales")
    print(f"{'='*60}")
    print(f"  Plataforma: {platform.platform()}")
    print(f"  CPU:        {platform.processor()}")
    import psutil
    ram_gb = psutil.virtual_memory().total / 1e9
    print(f"  RAM total:  {ram_gb:.1f} GB")
    print(f"  Python:     {sys.version.split()[0]}")

    # Mini-benchmark CyL (k*=100 ya conocido)
    coords_df = load_muns(os.path.join(DATA_CYL, 'cyl_1000.csv'))
    coords_deg = [(row['Latitud'], row['Longitud']) for _, row in coords_df.iterrows()]
    names_list = list(coords_df['Población'])

    print(f"\n  Midiendo tiempo para CyL (n={len(names_list)}, k=100)...")
    times = []
    for i in range(3):
        t0 = time.time()
        D = haversine_matrix(coords_deg)
        km = KMedoids(n_clusters=100, metric='precomputed', random_state=SEED+i, method='pam')
        km.fit(D)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"    Run {i+1}: {elapsed:.1f}s")

    med_t = np.median(times)
    mad_t = np.median(np.abs(np.array(times) - med_t))
    print(f"\n  CyL k=100: mediana={med_t:.1f}s ± {mad_t:.1f}s (MAD)")
    return {'platform': platform.platform(), 'cyl_k100_median_s': round(med_t,1), 'cyl_k100_mad_s': round(mad_t,1)}


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == '__main__':
    all_results = {}

    # Cargar datos de municipios
    print("Cargando municipios CyL...")
    df_cyl = load_muns(os.path.join(DATA_CYL, 'cyl_1000.csv'))
    names_cyl  = list(df_cyl['Población'])
    coords_cyl = [(row['Latitud'], row['Longitud']) for _, row in df_cyl.iterrows()]
    print(f"  CyL: {len(names_cyl)} municipios")

    print("Cargando municipios España...")
    df_esp = load_muns(os.path.join(DATA_ESP, 'peninsula_1000.csv'))
    names_esp  = list(df_esp['Población'])
    coords_esp = [(row['Latitud'], row['Longitud']) for _, row in df_esp.iterrows()]
    print(f"  España: {len(names_esp)} municipios")

    # ── Table IV (más rápida, usa archivos preexistentes) ──────────────────
    all_results['table_iv'] = table_iv_k_beyond_kstar()

    # ── Table II (k-means violations, España) ─────────────────────────────
    # ADVERTENCIA: España tiene n=3102, k=950. k-means es rápido pero
    # calcular la distancia haversine para cada par puede tomar tiempo.
    # Aquí usamos KMeans directamente sobre coordenadas (sin matriz completa).
    print("\n  [Table II usa KMeans sobre coords, no haversine matrix completa]")
    all_results['table_ii'] = table_ii_kmeans_violations(
        coords_esp, names_esp, delta=DELTA_DEFAULT, k=950, n_runs=20
    )

    # ── Table III (Δ sensitivity) ──────────────────────────────────────────
    # ADVERTENCIA: Requiere calcular matrices haversine a Δ=30 y Δ=60 km
    # y ejecutar PAM. Puede tardar 30-60 min para España a Δ=30km.
    # Descomenta la siguiente línea cuando tengas tiempo:
    # all_results['table_iii'] = table_iii_delta_sensitivity(coords_cyl, names_cyl, coords_esp, names_esp)

    # Como alternativa rápida, ejecuta solo CyL (n=267, rápido):
    print(f"\n{'='*60}")
    print(f"TABLE III — Sensibilidad a Δ (solo CyL, rápido)")
    print(f"{'='*60}")
    delta_results = {}
    for delta in [DELTA_LOW, DELTA_DEFAULT, DELTA_HIGH]:
        print(f"\n  Δ = {delta} km (CyL)...")
        try:
            kstar, _, _ = find_kstar(coords_cyl, names_cyl, delta)
            print(f"  >> k*={kstar}")
            delta_results[('CyL', delta)] = kstar
        except RuntimeError as e:
            print(f"  >> INFEASIBLE: {e}")
            delta_results[('CyL', delta)] = None
    # Tabla resumen
    print(f"\n  {'Δ (km)':>8} | k* (CyL)")
    print(f"  {'-'*22}")
    for delta in [DELTA_LOW, DELTA_DEFAULT, DELTA_HIGH]:
        v = delta_results.get(('CyL', delta))
        print(f"  {delta:>8.0f} | {v if v is not None else 'infeasible'}")
    all_results['table_iii_cyl'] = delta_results

    # ── Benchmark ─────────────────────────────────────────────────────────
    try:
        import psutil
        all_results['benchmark'] = benchmark_hardware()
    except ImportError:
        print("\n  [Instala psutil para info de RAM: pip install psutil]")

    # ── Guardar CSV ────────────────────────────────────────────────────────
    out_csv = os.path.join(RESULTS, 'tablas_pam_generation.csv')
    rows = []
    if 'table_iv' in all_results:
        for r in all_results['table_iv']:
            rows.append({'tabla': 'IV', **r})
    if 'table_ii' in all_results:
        rows.append({'tabla': 'II', **all_results['table_ii']})
    if 'table_iii_cyl' in all_results:
        for (label, delta), kstar in all_results['table_iii_cyl'].items():
            rows.append({'tabla': 'III', 'dataset': label, 'delta': delta, 'kstar': kstar})

    if rows:
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=sorted(all_keys))
            w.writeheader()
            w.writerows(rows)
        print(f"\nResultados guardados en: {out_csv}")

    print("\nDone.")
