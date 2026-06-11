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
  /Users/igarcia/doctorado/2025_2026/experimentos/exp12_fallos_adversariales.log

Uso:
  cd /Users/igarcia/doctorado/2025_2026/codigo/qkd-net-tools
  python analisis/fallos_adversariales.py
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, '..'))
DATA = os.path.join(ROOT, 'datos')
OUT_DIR = os.path.join(DATA, 'resultados_papers')
LOG_DIR = '/Users/igarcia/doctorado/2025_2026/experimentos'

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

sys.path.insert(0, ROOT)
sys.path.insert(0, BASE)

# Maquinaria canonica: cargadores de grafo y motor widest-path validado.
from capacidad_servicio_ataques import CARGADORES, bottlenecks_pares  # noqa: E402
from protocols.skr_bb84 import skr_bb84_decoy, P_DARK                  # noqa: E402

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

C0_CANON = {'cyl': 4.162e-3, 'espana': 6.0175e-3, 'adif': 6.871e-5}

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
      aristas no encontradas usan 45 km (mismo defecto que el cargador canonico).
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
    d, falt = {}, 0
    for u, v in G.edges():
        dist = tabla.get(frozenset((u, v)))
        if dist is None:
            dist = 45.0
            falt += 1
        d[frozenset((u, v))] = dist
    if falt:
        log(f"    [AVISO] {red}: {falt} aristas sin dist_km en skr_per_link.csv "
            f"(asignado 45 km)")
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
            cache_degr[dist] = skr_bb84_decoy(dist, p_dark=kappa * P_DARK)
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
      - wp_route:  camino widest-path (= camino en el max spanning tree por 'skr')
      - hop_route: camino hop-shortest (BFS no ponderado)
    Devuelve dos listas de listas de aristas (frozenset). Par sin camino -> None.
    """
    # Para los widest-path: arbol de maxima expansion sobre 'skr' (sano).
    F = nx.maximum_spanning_tree(G, weight='skr')
    wp, hop = [], []
    for u, v in pares:
        # widest-path
        try:
            path = nx.shortest_path(F, u, v)
            wp.append([frozenset((path[i], path[i + 1]))
                       for i in range(len(path) - 1)])
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            wp.append(None)
        # hop-shortest
        try:
            path = nx.shortest_path(G, u, v)
            hop.append([frozenset((path[i], path[i + 1]))
                        for i in range(len(path) - 1)])
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            hop.append(None)
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
    s_sano = skr_bb84_decoy(10.0)
    s_1e3 = skr_bb84_decoy(10.0, p_dark=1e3 * P_DARK)
    ratio = s_1e3 / s_sano
    log(f"Verificacion kappa: SKR(10km) sano={s_sano:.4e}; "
        f"kappa=1e3 -> {s_1e3:.4e} (ratio {ratio:.3f})")
    if ratio > 0.10:
        s_1e4 = skr_bb84_decoy(10.0, p_dark=1e4 * P_DARK)
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

        # C(0) check
        c0_vals = bottlenecks_pares(G, pares)
        C0 = float(np.median(c0_vals))
        canon = C0_CANON[red]
        ok = abs(C0 - canon) / canon < 1e-3
        log(f"  C(0) mediana={C0:.6e} (canonico {canon:.4e}) "
            f"{'OK' if ok else 'DISCREPANCIA'}")
        if not ok:
            raise RuntimeError(f"C(0) de {red} no coincide con el canonico; "
                               f"carga de grafo divergente.")

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
