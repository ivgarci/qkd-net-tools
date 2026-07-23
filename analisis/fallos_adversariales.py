"""
Fallos adversariales de dispositivo en redes QKD de relés de confianza (Paper P8).

Agente F-EXP-B. AUTOCONTENIDO: no importa fallos_dispositivo.py; implementa el
modelo de fallo M2 (inyección de cuentas oscuras / blinding) según la espec del
PLAN_PAPER.md usando protocols/skr_bb84.py (parametrizado en eta_det/p_dark/e_det).

Escenario y contramedidas:
  S3  — blinding adversarial dirigido (M2, kappa) sobre top-k relés por C_B^w
        (rank_weighted), por C_B (rank_unweighted) y aleatorio (baseline R=50).
        Barrido k in {1,2,5,10,20,50} (CyL: k<=20). Metrica C/C(0) sobre pares
        fijos + fraccion de pares sin servicio (SKR=0). El grafo NO cambia de
        topologia (no se elimina ningun nodo): S(p)=1 constante.
  CM1 — re-enrutamiento consciente del fallo: C bajo (a) widest-path congelado
        pre-fallo, (b) hop-shortest congelado pre-fallo, (c) widest-path
        recalculado post-fallo. Ganancia de re-enrutamiento = c/a.
  CM2 — refuerzo selectivo: restaurar top-m nodos faulted (orden por C_B^w, C_B,
        aleatorio R=20); curva de recuperacion C(m)/C(0).

Modelo de fallo (M2, locus = relé):
  Un fallo en el relé v degrada TODAS las aristas incidentes a v (el detector
  del receptor esta en el extremo del enlace). Una arista (u,v) esta degradada
  si u o v estan faulted: su SKR se recalcula con p'_dark = kappa * P_DARK.

Salidas:
  datos/resultados_papers/fallos_s3.csv
  datos/resultados_papers/contramedidas_cm1.csv
  datos/resultados_papers/contramedidas_cm2.csv
  logs/exp12_fallos_adversariales.log (o $QKD_LOG_DIR)

Uso:
  python analisis/fallos_adversariales.py
"""

import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, '..'))
DATA = os.path.join(ROOT, 'datos')
OUT_DIR = os.path.join(DATA, 'resultados_papers')
LOG_DIR = os.environ.get('QKD_LOG_DIR', os.path.join(ROOT, 'logs'))

os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, ROOT)
sys.path.insert(0, BASE)

# Maquinaria canonica: cargadores de grafo y motor widest-path validado.
from capacidad_servicio_ataques import CARGADORES, bottlenecks_pares  # noqa: E402
from protocols.skr_bb84 import skr_bb84_asymptotic, P_DARK            # noqa: E402
from routing_core import (                                             # noqa: E402
    max_min_metrics_from_source,
    min_hops_routes_from_source,
    node_key,
)

# ---------------------------------------------------------------------------
# Parametros
# ---------------------------------------------------------------------------

REDES = ['cyl', 'espana', 'adif']
K_GRID = [1, 2, 5, 10, 20, 50]          # CyL se limita a k<=20
K_MAX_CYL = 20
R_RANDOM_S3 = 50                        # baseline aleatorio S3 (semillas 0..49)
R_RANDOM_CM2 = 20                       # baseline aleatorio CM2 (semillas 0..19)

# Escenario CM1/CM2: estrategia mas dañina (se determina de S3) con k medio.
K_MEDIO = {'cyl': 5, 'espana': 10, 'adif': 10}

# kappa: la espec pide kappa=1e3, salvo que SKR(10km, kappa=1e3) > 10% del sano,
# en cuyo caso kappa=1e4. Verificacion en runtime (ver determinar_kappa()).
KAPPA = None  # asignado en main

LOG_FH = None


def log(msg):
    linea = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(linea, flush=True)
    if LOG_FH is not None:
        LOG_FH.write(linea + '\n')
        LOG_FH.flush()


# ---------------------------------------------------------------------------
# Distancias por arista (necesarias para recalcular el SKR degradado)
# ---------------------------------------------------------------------------

def dist_por_arista(G, red):
    """
    Devuelve {frozenset(u,v): dist_km} para todas las aristas.
    - ADIF: el cargador canonico ya pone 'dist_km' en cada arista.
    - CyL/España: se toma dist_km de datos/skr_per_link.csv (caso correspondiente);
      cualquier arista no encontrada invalida el experimento.
    """
    if red == 'adif':
        d = {frozenset((u, v)): float(data['dist_km'])
             for u, v, data in G.edges(data=True)}
        return d
    caso = {'cyl': 'CyL', 'espana': 'España'}[red]
    df = pd.read_csv(os.path.join(DATA, 'skr_per_link.csv'))
    df = df[df['caso'] == caso]
    tabla = {frozenset((r.nodo_u, r.nodo_v)): float(r.dist_km)
             for r in df.itertuples()}
    falt = [
        (u, v) for u, v in G.edges()
        if frozenset((u, v)) not in tabla
    ]
    if falt:
        muestra = ', '.join(f'{u!r}--{v!r}' for u, v in falt[:5])
        raise ValueError(
            f"{red}: {len(falt)} aristas sin distancia en "
            f"datos/skr_per_link.csv; primeras: {muestra}"
        )
    d = {}
    for u, v in G.edges():
        edge = frozenset((u, v))
        d[edge] = tabla[edge]
    return d


def skr_maps(G, dist_map, kappa):
    """
    Devuelve (sano, degradado): dos dicts {frozenset(u,v): skr}.
      sano      = SKR(dist)                            (= atributo 'skr' del grafo)
      degradado = SKR(dist, p_dark=kappa*P_DARK)       (M2 sobre esa arista)
    """
    sano, degr = {}, {}
    cache_degr = {}
    for u, v in G.edges():
        fs = frozenset((u, v))
        dist = dist_map[fs]
        sano[fs] = G[u][v]['skr']
        if dist not in cache_degr:
            cache_degr[dist] = skr_bb84_asymptotic(
                dist, p_dark=kappa * P_DARK
            )
        degr[fs] = cache_degr[dist]
    return sano, degr


# ---------------------------------------------------------------------------
# Aplicacion del fallo M2 a un conjunto de relés faulted
# ---------------------------------------------------------------------------

def aristas_incidentes(G, faulted):
    """Conjunto de frozenset de aristas incidentes a algun nodo faulted."""
    afect = set()
    for v in faulted:
        for w in G.neighbors(v):
            afect.add(frozenset((v, w)))
    return afect


def aplicar_fallo(G, sano, degr, faulted):
    """
    Mutar 'skr' en G: aristas incidentes a faulted -> degradado, resto -> sano.
    Devuelve el conjunto de aristas afectadas (para restaurar despues).
    """
    afect = aristas_incidentes(G, faulted)
    for fs in afect:
        u, v = tuple(fs)
        G[u][v]['skr'] = degr[fs]
    return afect


def restaurar(G, sano, afect):
    """Devolver a sano las aristas afectadas (deshace aplicar_fallo)."""
    for fs in afect:
        u, v = tuple(fs)
        G[u][v]['skr'] = sano[fs]


def C_y_frac(G, pares, C0):
    """C_rel = mediana(bottleneck)/C0 y fraccion de pares con bottleneck 0."""
    vals = bottlenecks_pares(G, pares)
    c_med = float(np.median(vals))
    frac0 = float(np.mean(vals == 0.0))
    return c_med / C0, frac0


def evaluar_faulted(G, sano, degr, faulted, pares, C0):
    """Aplica fallo, mide C_rel y frac0, restaura. Idempotente sobre G."""
    afect = aplicar_fallo(G, sano, degr, faulted)
    try:
        return C_y_frac(G, pares, C0)
    finally:
        restaurar(G, sano, afect)


# ---------------------------------------------------------------------------
# Rankings (top-k por C_B^w y C_B)
# ---------------------------------------------------------------------------

def cargar_rankings():
    df = pd.read_csv(os.path.join(OUT_DIR, 'betweenness_ponderada.csv'),
                     dtype={'nodo': str})
    return df


def topk(rank_df, red, col_rank, k):
    s = rank_df[rank_df['red'] == red].sort_values(col_rank)
    return list(s['nodo'].iloc[:k])


# ---------------------------------------------------------------------------
# S3 — blinding adversarial dirigido
# ---------------------------------------------------------------------------

def correr_s3(red, G, pares, C0, sano, degr, rank_df, nodos_ordenados):
    filas = []
    k_grid = [k for k in K_GRID if not (red == 'cyl' and k > K_MAX_CYL)]
    n = G.number_of_nodes()

    # S(p): el grafo no cambia de topologia -> giant component constante.
    S = len(max(nx.connected_components(G), key=len)) / n
    log(f"  [{red}] S (sin eliminar nodos) = {S:.4f} (debe ser 1.0)")

    for k in k_grid:
        # (a) C_B^w
        f_w = topk(rank_df, red, 'rank_weighted', k)
        c_w, fr_w = evaluar_faulted(G, sano, degr, f_w, pares, C0)
        # (b) C_B
        f_u = topk(rank_df, red, 'rank_unweighted', k)
        c_u, fr_u = evaluar_faulted(G, sano, degr, f_u, pares, C0)
        # (c) aleatorio
        c_r, fr_r = [], []
        for seed in range(R_RANDOM_S3):
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, size=k, replace=False)
            f_rnd = [nodos_ordenados[i] for i in idx]
            cc, ff = evaluar_faulted(G, sano, degr, f_rnd, pares, C0)
            c_r.append(cc)
            fr_r.append(ff)
        c_r = np.array(c_r)
        fr_r = np.array(fr_r)

        filas.append({'red': red, 'estrategia': 'cb_weighted', 'k': k,
                      'C_rel': c_w, 'frac_pares_cero': fr_w, 'C_rel_std': np.nan})
        filas.append({'red': red, 'estrategia': 'cb_unweighted', 'k': k,
                      'C_rel': c_u, 'frac_pares_cero': fr_u, 'C_rel_std': np.nan})
        filas.append({'red': red, 'estrategia': 'random', 'k': k,
                      'C_rel': float(c_r.mean()),
                      'frac_pares_cero': float(fr_r.mean()),
                      'C_rel_std': float(c_r.std())})
        log(f"  [{red}] k={k:<3} | C_B^w={c_w:.4f} (f0={fr_w:.3f}) | "
            f"C_B={c_u:.4f} (f0={fr_u:.3f}) | rand={c_r.mean():.4f}"
            f"±{c_r.std():.4f}")
    return filas, k_grid


# ---------------------------------------------------------------------------
# CM1 — re-enrutamiento consciente del fallo
# ---------------------------------------------------------------------------

def rutas_congeladas(G, sano, pares):
    """
    Precalcula sobre la red SANA, para cada par:
      - wp_route:  max-min SKR; entre empates, mínimos saltos;
      - hop_route: mínimos saltos; entre empates, máximo bottleneck.

    El último empate se resuelve lexicográficamente. No se usa un árbol de
    expansión ni el orden de inserción de NetworkX, porque ambos pueden elegir
    una ruta congelada distinta y alterar después el resultado de CM1.
    Devuelve dos listas de listas de aristas (frozenset). Par sin camino -> None.
    """
    targets_by_source = defaultdict(list)
    for position, (source, target) in enumerate(pares):
        targets_by_source[source].append((position, target))

    wp = [None] * len(pares)
    hop = [None] * len(pares)

    def edge_list(path):
        return [
            frozenset((path[index], path[index + 1]))
            for index in range(len(path) - 1)
        ]

    def lex_shortest_at_threshold(source, target, threshold):
        queue = deque([source])
        paths = {source: (source,)}
        while queue:
            node = queue.popleft()
            if node == target:
                return paths[node]
            for neighbour in sorted(G.neighbors(node), key=node_key):
                if neighbour in paths:
                    continue
                if float(G[node][neighbour]['skr']) < threshold:
                    continue
                paths[neighbour] = paths[node] + (neighbour,)
                queue.append(neighbour)
        return None

    for source in sorted(targets_by_source, key=node_key):
        hop_routes = min_hops_routes_from_source(
            G, source, capacity_attr='skr'
        )
        widest_metrics = max_min_metrics_from_source(
            G, source, capacity_attr='skr'
        )
        for position, target in targets_by_source[source]:
            hop_route = hop_routes.get(target)
            widest_metric = widest_metrics.get(target)
            if hop_route is not None:
                hop[position] = edge_list(hop_route.path)
            if widest_metric is not None:
                widest_path = lex_shortest_at_threshold(
                    source, target, widest_metric.bottleneck
                )
                if widest_path is not None:
                    wp[position] = edge_list(widest_path)
    return wp, hop


def C_rutas_fijas(rutas, skr_actual, C0):
    """
    C_rel y frac0 evaluando rutas congeladas con los SKR degradados actuales.
    skr_actual: dict frozenset->skr (combinado sano/degradado).
    bottleneck = min SKR de las aristas de la ruta (0 si ruta None o vacia).
    """
    vals = []
    for ruta in rutas:
        if not ruta:
            vals.append(0.0)
            continue
        b = min(skr_actual[fs] for fs in ruta)
        vals.append(b)
    vals = np.array(vals)
    return float(np.median(vals)) / C0, float(np.mean(vals == 0.0))


def correr_cm1(red, G, pares, C0, sano, degr, faulted):
    """
    CM1 sobre el conjunto faulted dado (estrategia mas dañina, k medio).
      (a) widest-path congelado pre-fallo
      (b) hop-shortest congelado pre-fallo
      (c) widest-path recalculado post-fallo (= C de S3)
    """
    wp, hop = rutas_congeladas(G, sano, pares)
    afect = aristas_incidentes(G, faulted)
    # SKR actual (post-fallo) por arista para evaluar rutas congeladas.
    skr_actual = dict(sano)
    for fs in afect:
        skr_actual[fs] = degr[fs]

    c_a, fr_a = C_rutas_fijas(wp, skr_actual, C0)
    c_b, fr_b = C_rutas_fijas(hop, skr_actual, C0)
    c_c, fr_c = evaluar_faulted(G, sano, degr, faulted, pares, C0)

    filas = [
        {'red': red, 'politica': 'widest_path_congelado', 'C_rel': c_a,
         'frac_pares_cero': fr_a},
        {'red': red, 'politica': 'hop_shortest_congelado', 'C_rel': c_b,
         'frac_pares_cero': fr_b},
        {'red': red, 'politica': 'widest_path_recalculado', 'C_rel': c_c,
         'frac_pares_cero': fr_c},
    ]
    ganancia = c_c / c_a if c_a > 0 else float('inf')
    log(f"  [{red}] CM1 | (a)wp_cong={c_a:.4f} | (b)hop_cong={c_b:.4f} | "
        f"(c)wp_recalc={c_c:.4f} | ganancia c/a={ganancia:.3f}")
    return filas


# ---------------------------------------------------------------------------
# CM2 — refuerzo selectivo (restauracion)
# ---------------------------------------------------------------------------

def orden_restauracion_por_rank(rank_df, red, faulted, col_rank):
    """Subconjunto faulted ordenado por col_rank ascendente (mas central primero)."""
    s = rank_df[(rank_df['red'] == red) & (rank_df['nodo'].isin(faulted))]
    return list(s.sort_values(col_rank)['nodo'])


def correr_cm2(red, G, pares, C0, sano, degr, faulted, rank_df):
    """
    Restaurar top-m de los k nodos faulted (m=1..k). Estrategias de orden:
      C_B^w, C_B, aleatorio (R=20). Curva C(m)/C(0).
    Restaurar m nodos => faulted_restante = faulted \\ top-m.
    """
    k = len(faulted)
    filas = []

    def eval_restaurando(restaurados):
        restante = [v for v in faulted if v not in set(restaurados)]
        return evaluar_faulted(G, sano, degr, restante, pares, C0)

    # Orden deterministas
    ord_w = orden_restauracion_por_rank(rank_df, red, faulted, 'rank_weighted')
    ord_u = orden_restauracion_por_rank(rank_df, red, faulted, 'rank_unweighted')

    for m in range(1, k + 1):
        c_w, _ = eval_restaurando(ord_w[:m])
        c_u, _ = eval_restaurando(ord_u[:m])
        # aleatorio
        c_r = []
        for seed in range(R_RANDOM_CM2):
            rng = np.random.default_rng(1000 + seed)
            orden = list(faulted)
            rng.shuffle(orden)
            cc, _ = eval_restaurando(orden[:m])
            c_r.append(cc)
        c_r = np.array(c_r)
        filas.append({'red': red, 'estrategia': 'cb_weighted', 'm': m,
                      'C_rel': c_w, 'C_rel_std': np.nan})
        filas.append({'red': red, 'estrategia': 'cb_unweighted', 'm': m,
                      'C_rel': c_u, 'C_rel_std': np.nan})
        filas.append({'red': red, 'estrategia': 'random', 'm': m,
                      'C_rel': float(c_r.mean()), 'C_rel_std': float(c_r.std())})
    log(f"  [{red}] CM2 | curva de recuperacion calculada (m=1..{k}, "
        f"3 estrategias)")
    return filas


# ---------------------------------------------------------------------------
# kappa
# ---------------------------------------------------------------------------

def determinar_kappa():
    s_sano = skr_bb84_asymptotic(10.0)
    s_1e3 = skr_bb84_asymptotic(10.0, p_dark=1e3 * P_DARK)
    ratio = s_1e3 / s_sano
    log(f"Verificacion kappa: SKR(10km) sano={s_sano:.4e}; "
        f"kappa=1e3 -> {s_1e3:.4e} (ratio {ratio:.3f})")
    if ratio > 0.10:
        s_1e4 = skr_bb84_asymptotic(10.0, p_dark=1e4 * P_DARK)
        log(f"  kappa=1e3 deja SKR > 10% del sano -> se usa kappa=1e4 "
            f"(SKR(10km, kappa=1e4)={s_1e4:.4e}, ratio {s_1e4/s_sano:.3f})")
        return 1e4
    log("  kappa=1e3 suficiente.")
    return 1e3


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global LOG_FH, KAPPA
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FH = open(os.path.join(LOG_DIR, 'exp12_fallos_adversariales.log'), 'w')
    t_glob = time.time()

    log("=" * 72)
    log("EXP-12 — Fallos adversariales (S3) y contramedidas (CM1, CM2) — P8")
    log("=" * 72)

    KAPPA = determinar_kappa()
    log(f"KAPPA fijado = {KAPPA:.0e}")

    rank_df = cargar_rankings()

    s3_all, cm1_all, cm2_all = [], [], []
    tiempos = {}

    for red in REDES:
        t_red = time.time()
        log("-" * 72)
        log(f"RED: {red}")
        G = CARGADORES[red]()
        n = G.number_of_nodes()
        nodos_ordenados = sorted(G.nodes(), key=str)

        pares_df = pd.read_csv(
            os.path.join(OUT_DIR, f'pares_muestreo_{red}.csv'),
            dtype={'nodo_u': str, 'nodo_v': str})
        pares = list(zip(pares_df['nodo_u'], pares_df['nodo_v']))

        # C(0) se deriva del modelo, el grafo y los pares versionados.
        c0_vals = bottlenecks_pares(G, pares)
        C0 = float(np.median(c0_vals))
        if not np.isfinite(C0) or C0 <= 0.0:
            raise RuntimeError(
                f"{red}: C(0) inválida ({C0!r}); revisar modelo, grafo y pares"
            )
        log(f"  C(0) mediana derivada={C0:.6e}")

        dist_map = dist_por_arista(G, red)
        sano, degr = skr_maps(G, dist_map, KAPPA)

        # ---- S3 ----
        log(f"  S3 (blinding dirigido, kappa={KAPPA:.0e})...")
        filas_s3, k_grid = correr_s3(red, G, pares, C0, sano, degr,
                                     rank_df, nodos_ordenados)
        s3_all.extend(filas_s3)

        # Estrategia mas dañina a k medio (menor C_rel entre cb_w y cb).
        km = K_MEDIO[red]
        df_km = pd.DataFrame([f for f in filas_s3 if f['k'] == km])
        c_w = df_km[df_km['estrategia'] == 'cb_weighted']['C_rel'].iloc[0]
        c_u = df_km[df_km['estrategia'] == 'cb_unweighted']['C_rel'].iloc[0]
        if c_w <= c_u:
            estr, col = 'cb_weighted', 'rank_weighted'
        else:
            estr, col = 'cb_unweighted', 'rank_unweighted'
        faulted = topk(rank_df, red, col, km)
        log(f"  Escenario CM1/CM2: estrategia mas dañina='{estr}' "
            f"(C_rel C_B^w={c_w:.4f} vs C_B={c_u:.4f}), k_medio={km}, "
            f"|faulted|={len(faulted)}")

        # ---- CM1 ----
        cm1_all.extend(correr_cm1(red, G, pares, C0, sano, degr, faulted))

        # ---- CM2 ----
        cm2_all.extend(correr_cm2(red, G, pares, C0, sano, degr, faulted,
                                  rank_df))

        tiempos[red] = time.time() - t_red
        log(f"  Tiempo {red}: {tiempos[red]:.1f} s")

    # ---- Salidas ----
    pd.DataFrame(s3_all, columns=['red', 'estrategia', 'k', 'C_rel',
                                  'frac_pares_cero', 'C_rel_std']).to_csv(
        os.path.join(OUT_DIR, 'fallos_s3.csv'), index=False)
    pd.DataFrame(cm1_all, columns=['red', 'politica', 'C_rel',
                                   'frac_pares_cero']).to_csv(
        os.path.join(OUT_DIR, 'contramedidas_cm1.csv'), index=False)
    pd.DataFrame(cm2_all, columns=['red', 'estrategia', 'm', 'C_rel',
                                   'C_rel_std']).to_csv(
        os.path.join(OUT_DIR, 'contramedidas_cm2.csv'), index=False)

    log("-" * 72)
    log("Guardado: fallos_s3.csv, contramedidas_cm1.csv, contramedidas_cm2.csv")
    for red in REDES:
        log(f"  tiempo {red}: {tiempos[red]:.1f} s")
    log(f"Tiempo total: {time.time() - t_glob:.1f} s")
    LOG_FH.close()


if __name__ == '__main__':
    main()
