"""
Capacidad de servicio QKD bajo ataques de eliminación de nodos.

Script canónico del experimento EXP-8 (paper QKD-Service-Resilience).
Para una red dada (--red cyl|adif|espana) mide, bajo cinco protocolos de
eliminación de nodos, dos familias de métricas en función de la fracción
eliminada p ∈ [0, 0.50]:

  - S(p): fracción del componente gigante respecto a |V| original
    (métrica topológica clásica de percolación).
  - C(p): capacidad de servicio sobre N pares fijos de nodos — para cada
    par, el SKR de cuello de botella de su ruta widest-path (max-min);
    vale 0 si algún extremo está eliminado o no existe camino.
    Se reportan la mediana (C_median) y la media (C_mean) sobre los pares.

Protocolos de eliminación:
  1. random                — fallos aleatorios (R=300 realizaciones, semillas 0..299)
  2. degree_static         — ranking C_D del grafo original, fijo
  3. degree_adaptive       — recalcula grados tras cada eliminación
  4. betweenness_static    — ranking C_B del grafo original, fijo
  5. betweenness_adaptive  — recalcula C_B tras cada eliminación

Umbrales resumen por protocolo:
  p*_topo = min{p : S(p) < 0.5}
  p*_svc  = min{p : C_median(p) < 0.5·C_median(0)}

Física del canal:
  - CyL / España: SKR por arista precalculado en datos/skr_per_link.csv
    (modelo BB84 asintótico ideal, η_det = 0.10; ver
    protocols/skr_bb84.py).
  - ADIF: grafo de junctions (contracción de cadenas de grado 2, misma
    metodología que adif/analisis_adif_junctions.py); SKR por arista
    evaluado con skr_bb84_asymptotic() sobre la distancia acumulada del
    corredor ferroviario contraído. Se conserva el valor que devuelve el
    modelo, incluido 0 cuando la expresión deja de producir clave; no se
    filtran aristas y la topología no cambia.

Enrutamiento:
  La ruta widest-path (max-min bottleneck) es la misma lógica que
  analisis/enrutamiento_qkd.py::max_skr_path. Para abaratar las ~3·10⁶
  consultas por red se usa la equivalencia exacta «bottleneck max-min =
  mínima arista del camino en el bosque de máxima expansión», validada
  numéricamente contra el núcleo single-source al inicio de cada ejecución.

Genera:
  datos/resultados_papers/pares_muestreo_<red>.csv
  datos/resultados_papers/capacidad_<red>.csv
  datos/resultados_papers/capacidad_umbrales_<red>.csv
  logs/exp8_capacidad_<red>.log (o $QKD_LOG_DIR)

Uso:
    python analisis/capacidad_servicio_ataques.py --red cyl
    python analisis/capacidad_servicio_ataques.py --red adif
"""

import os
import sys
import math
import time
import argparse
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx

BASE      = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.abspath(os.path.join(BASE, '..'))
DATA      = os.path.join(ROOT, 'datos')
DATA_CYL  = os.path.join(DATA, 'cyl')
DATA_ESP  = os.path.join(DATA, 'espana')
DATA_ADIF = os.path.join(DATA, 'adif')
OUT_DIR   = os.path.join(DATA, 'resultados_papers')
LOG_DIR   = os.environ.get('QKD_LOG_DIR', os.path.join(ROOT, 'logs'))

os.makedirs(OUT_DIR, exist_ok=True)

# Reutilizamos la implementación widest-path existente como referencia
# (analisis/enrutamiento_qkd.py) y el modelo BB84 asintótico ideal.
sys.path.insert(0, ROOT)
sys.path.insert(0, BASE)
from analisis.routing_core import max_min_metrics_from_source  # noqa: E402
from protocols.skr_bb84 import skr_bb84_asymptotic  # noqa: E402

# ---------------------------------------------------------------------------
# Parámetros del experimento
# ---------------------------------------------------------------------------

SEED_PARES    = 42                                  # semilla muestreo de pares
N_PARES       = 1000                                # pares objetivo por red
R_RANDOM      = 300                                 # realizaciones de fallos aleatorios
P_GRID        = [round(0.01 * i, 2) for i in range(51)]   # 0.00 .. 0.50
P_GRID_C_RAND = [round(0.05 * j, 2) for j in range(11)]   # C en random: 0.00 .. 0.50
N_BOOTSTRAP   = 1000                                # remuestreos IC bootstrap
SEED_BOOT     = 1234                                # semilla bootstrap
UMBRAL_S      = 0.50                                # p*_topo: S(p) < 0.5
UMBRAL_C      = 0.50                                # p*_svc: C_med(p) < 0.5·C_med(0)

LOG_FH = None


def log(msg):
    """Log con timestamp a fichero y stdout."""
    linea = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(linea, flush=True)
    if LOG_FH is not None:
        LOG_FH.write(linea + '\n')
        LOG_FH.flush()


# ---------------------------------------------------------------------------
# Carga de grafos con atributo 'skr' (bits/pulso) por arista
# ---------------------------------------------------------------------------

def _asignar_skr_desde_csv(G, caso):
    """Asigna SKR por arista desde datos/skr_per_link.csv (clave: nombres)."""
    df = pd.read_csv(os.path.join(DATA, 'skr_per_link.csv'))
    df = df[df['caso'] == caso]
    tabla = {frozenset((r.nodo_u, r.nodo_v)): float(r.SKR_bits_pulso)
             for r in df.itertuples()}
    faltantes = [
        (u, v) for u, v in G.edges()
        if frozenset((u, v)) not in tabla
    ]
    if faltantes:
        muestra = ', '.join(f'{u!r}--{v!r}' for u, v in faltantes[:5])
        raise ValueError(
            f"{caso}: {len(faltantes)} aristas sin SKR en "
            f"datos/skr_per_link.csv; primeras: {muestra}"
        )
    for u, v in G.edges():
        skr = tabla[frozenset((u, v))]
        G[u][v]['skr'] = skr
        G[u][v]['SKR'] = skr   # compatibilidad con max_skr_path()
    return G


def cargar_grafo_cyl():
    adj = pd.read_csv(os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv'),
                      index_col=0)
    G = nx.from_pandas_adjacency(adj)
    return _asignar_skr_desde_csv(G, 'CyL')


def cargar_grafo_espana():
    adj = pd.read_csv(os.path.join(DATA_ESP, 'AdjacencyMatrixNamed45.csv'),
                      index_col=0)
    G = nx.from_pandas_adjacency(adj)
    return _asignar_skr_desde_csv(G, 'España')


def _construir_junctions(G):
    """
    Contrae las cadenas de nodos de grado 2 (misma metodología que
    adif/analisis_adif_junctions.py::build_junction_graph). Conserva los
    nodos con grado ≠ 2; el peso de cada arista resultante es la suma de
    dist_km del camino contraído (si hay rutas paralelas, la más corta).
    """
    keep = {n for n in G.nodes() if G.degree(n) != 2}
    J = nx.Graph()
    for n in keep:
        J.add_node(n, **G.nodes[n])

    visitadas = set()
    for inicio in keep:
        for vecino in list(G.neighbors(inicio)):
            ek = frozenset([inicio, vecino])
            if ek in visitadas:
                continue
            visitadas.add(ek)
            acc = G[inicio][vecino].get('dist_km', 0.0) or 0.0
            prev, cur = inicio, vecino
            while cur not in keep:
                vecinos = list(G.neighbors(cur))
                nxt = vecinos[0] if vecinos[1] == prev else vecinos[1]
                acc += G[cur][nxt].get('dist_km', 0.0) or 0.0
                visitadas.add(frozenset([cur, nxt]))
                prev, cur = cur, nxt
            if cur == inicio:
                continue   # bucle, ignorar
            if not J.has_edge(inicio, cur):
                J.add_edge(inicio, cur, dist_km=acc)
            elif acc < J[inicio][cur]['dist_km']:
                J[inicio][cur]['dist_km'] = acc
    return J


def cargar_grafo_adif():
    """Proxy ADIF contraído con SKR ideal sobre longitudes acumuladas."""
    nodos_df = pd.read_csv(os.path.join(DATA_ADIF, 'nodos_red_adif.csv'),
                           quotechar='"', on_bad_lines='skip')
    adj_df = pd.read_csv(os.path.join(DATA_ADIF, 'adyacencia_red_adif.csv'),
                         quotechar='"', on_bad_lines='skip')

    G_full = nx.Graph()
    conectados = nodos_df[nodos_df['conectado'] == 'SI']
    for _, row in conectados.iterrows():
        G_full.add_node(str(row['cod']), nombre=str(row['nombre']))

    vistos = set()
    for _, row in adj_df.iterrows():
        u, v = str(row['cod']), str(row['vecino_cod'])
        key = frozenset([u, v])
        if key in vistos:
            continue
        vistos.add(key)
        if not (G_full.has_node(u) and G_full.has_node(v)):
            continue
        try:
            d = float(row['dist_km'])
        except (ValueError, TypeError):
            continue
        G_full.add_edge(u, v, dist_km=d)

    lcc = max(nx.connected_components(G_full), key=len)
    J = _construir_junctions(G_full.subgraph(lcc).copy())
    comps = list(nx.connected_components(J))
    if len(comps) > 1:
        J = J.subgraph(max(comps, key=len)).copy()

    # Evaluación del modelo ideal sobre cada longitud acumulada del proxy.
    # No se filtran corredores: una tasa nula se conserva como tal.
    n_cero = 0
    for u, v in J.edges():
        skr = skr_bb84_asymptotic(J[u][v]['dist_km'])
        J[u][v]['skr'] = skr
        J[u][v]['SKR'] = skr
        if skr <= 0.0:
            n_cero += 1
    dists = [d['dist_km'] for _, _, d in J.edges(data=True)]
    log(f"  ADIF junctions: dist aristas min={min(dists):.1f} km, "
        f"max={max(dists):.1f} km; aristas con SKR=0: {n_cero}")
    return J


CARGADORES = {'cyl': cargar_grafo_cyl,
              'adif': cargar_grafo_adif,
              'espana': cargar_grafo_espana}


# ---------------------------------------------------------------------------
# Muestreo de pares fijos (los mismos para todos los protocolos)
# ---------------------------------------------------------------------------

def muestrear_pares(G, red):
    """N pares sin reemplazo de todos los nodos, semilla fija. Guarda CSV."""
    nodos = sorted(G.nodes(), key=str)
    todos = list(combinations(nodos, 2))
    rng = np.random.default_rng(SEED_PARES)
    if N_PARES >= len(todos):
        pares = todos
    else:
        idx = rng.choice(len(todos), size=N_PARES, replace=False)
        pares = [todos[i] for i in sorted(idx)]
    out = os.path.join(OUT_DIR, f'pares_muestreo_{red}.csv')
    pd.DataFrame([{'red': red, 'idx': i, 'nodo_u': u, 'nodo_v': v}
                  for i, (u, v) in enumerate(pares)]).to_csv(out, index=False)
    log(f"  Pares muestreados: {len(pares)} (de {len(todos)} posibles, "
        f"semilla {SEED_PARES}) → {out}")
    return pares


# ---------------------------------------------------------------------------
# Capacidad de servicio: bottleneck widest-path para todos los pares
# ---------------------------------------------------------------------------

def bottlenecks_pares(G, pares):
    """
    Bottleneck max-min (SKR) para cada par. Equivalencia exacta: el camino
    widest-path entre u y v tiene como cuello de botella la arista mínima
    del camino u–v en el bosque de máxima expansión respecto a 'skr'.
    Pares con extremo eliminado o sin camino (o bottleneck 0) → 0.0,
    idéntico al retorno de enrutamiento_qkd.max_skr_path.
    """
    F = nx.maximum_spanning_tree(G, weight='skr')

    # Etiquetado por componente: padre, profundidad y peso hacia el padre
    comp, padre, prof, w_padre = {}, {}, {}, {}
    cid = 0
    for raiz in F.nodes():
        if raiz in comp:
            continue
        comp[raiz], padre[raiz], prof[raiz] = cid, None, 0
        pila = [raiz]
        while pila:
            x = pila.pop()
            for y in F.neighbors(x):
                if y not in comp:
                    comp[y], padre[y], prof[y] = cid, x, prof[x] + 1
                    w_padre[y] = F[x][y]['skr']
                    pila.append(y)
        cid += 1

    valores = []
    for u, v in pares:
        if u not in comp or v not in comp or comp[u] != comp[v]:
            valores.append(0.0)
            continue
        b = math.inf
        x, y = u, v
        while prof[x] > prof[y]:
            b = min(b, w_padre[x]); x = padre[x]
        while prof[y] > prof[x]:
            b = min(b, w_padre[y]); y = padre[y]
        while x != y:
            b = min(b, w_padre[x]); x = padre[x]
            b = min(b, w_padre[y]); y = padre[y]
        valores.append(0.0 if b is math.inf else float(b))
    return np.array(valores)


def validar_motor(G, pares, n_check=100):
    """Compara el motor rápido con el núcleo exacto, agrupado por origen."""
    rng = np.random.default_rng(7)
    idx = rng.choice(len(pares), size=min(n_check, len(pares)), replace=False)
    sub = [pares[i] for i in idx]
    rapidos = bottlenecks_pares(G, sub)
    rutas_por_origen = {}
    for origen, _ in sub:
        if origen not in rutas_por_origen:
            rutas_por_origen[origen] = max_min_metrics_from_source(
                G, origen, capacity_attr='skr'
            )
    refs = np.array([
        rutas_por_origen[origen][destino].bottleneck
        for origen, destino in sub
    ])
    if not np.allclose(rapidos, refs, rtol=1e-9, atol=1e-15):
        peor = np.max(np.abs(rapidos - refs))
        raise RuntimeError(f"Motor de bottleneck inconsistente con "
                           f"routing_core (desv. máx = {peor:.3e})")
    log(f"  Validación motor widest-path: OK ({len(sub)} pares, "
        f"coincidencia exacta con routing_core)")


# ---------------------------------------------------------------------------
# Órdenes de eliminación
# ---------------------------------------------------------------------------

def _ranking_estatico(cent):
    """Orden descendente por centralidad; desempate por nombre (determinista)."""
    return [n for n, _ in sorted(cent.items(), key=lambda kv: (-kv[1], str(kv[0])))]


def orden_degree_static(G, k_max):
    return _ranking_estatico(dict(G.degree()))[:k_max]


def orden_betweenness_static(G, k_max):
    return _ranking_estatico(nx.betweenness_centrality(G))[:k_max]


def orden_adaptativo(G0, metrica, k_max, etiqueta=''):
    """Elimina uno a uno el nodo más central del grafo restante."""
    G = G0.copy()
    orden = []
    t0 = time.time()
    for paso in range(k_max):
        if metrica == 'degree':
            cent = dict(G.degree())
        else:
            cent = nx.betweenness_centrality(G)
        nodo = _ranking_estatico(cent)[0]
        G.remove_node(nodo)
        orden.append(nodo)
        if metrica != 'degree' and (paso + 1) % 25 == 0:
            log(f"    {etiqueta}: paso {paso + 1}/{k_max} "
                f"({time.time() - t0:.0f} s)")
    return orden


# ---------------------------------------------------------------------------
# Medición de una curva S(p), C(p) para un orden de eliminación dado
# ---------------------------------------------------------------------------

def k_de_p(p, n):
    """Número entero de nodos a eliminar más cercano a p·n."""
    return int(math.floor(p * n + 0.5))


def medir_curva(G0, orden, pares, p_medir_c=None):
    """
    Recorre P_GRID eliminando acumulativamente según 'orden'.
    p_medir_c: conjunto de p donde medir C (None = todos).
    Devuelve lista de dicts con p, S, C_median, C_mean (C = NaN si no medida).
    """
    n = G0.number_of_nodes()
    G = G0.copy()
    eliminados = 0
    filas = []
    for p in P_GRID:
        k = k_de_p(p, n)
        while eliminados < k and eliminados < len(orden):
            G.remove_node(orden[eliminados])
            eliminados += 1
        if G.number_of_nodes() == 0:
            S = 0.0
        else:
            S = len(max(nx.connected_components(G), key=len)) / n
        fila = {'p': p, 'S': S, 'C_median': np.nan, 'C_mean': np.nan}
        if p_medir_c is None or p in p_medir_c:
            vals = bottlenecks_pares(G, pares)
            fila['C_median'] = float(np.median(vals))
            fila['C_mean'] = float(np.mean(vals))
        filas.append(fila)
    return filas


# ---------------------------------------------------------------------------
# Protocolo random: ensamble de R realizaciones + bootstrap
# ---------------------------------------------------------------------------

def medir_random(G0, pares, red):
    n = G0.number_of_nodes()
    nodos = sorted(G0.nodes(), key=str)
    k_max = k_de_p(P_GRID[-1], n)
    p_c = set(P_GRID_C_RAND)

    S_mat, Cmed_mat, Cmean_mat = [], [], []
    t0 = time.time()
    for r in range(R_RANDOM):
        rng = np.random.default_rng(r)        # semillas 0..R-1
        orden = [nodos[i] for i in rng.permutation(n)[:k_max]]
        filas = medir_curva(G0, orden, pares, p_medir_c=p_c)
        S_mat.append([f['S'] for f in filas])
        Cmed_mat.append([f['C_median'] for f in filas if f['p'] in p_c])
        Cmean_mat.append([f['C_mean'] for f in filas if f['p'] in p_c])
        if (r + 1) % 50 == 0:
            log(f"    random: realización {r + 1}/{R_RANDOM} "
                f"({time.time() - t0:.0f} s)")

    S_mat = np.array(S_mat)          # (R, |P_GRID|)
    Cmed_mat = np.array(Cmed_mat)    # (R, |P_GRID_C_RAND|)
    Cmean_mat = np.array(Cmean_mat)

    # IC bootstrap 95% de la media de C_median entre realizaciones
    rng_b = np.random.default_rng(SEED_BOOT)
    ci_low, ci_high = [], []
    for j in range(Cmed_mat.shape[1]):
        muestras = rng_b.choice(Cmed_mat[:, j],
                                size=(N_BOOTSTRAP, R_RANDOM), replace=True)
        medias = muestras.mean(axis=1)
        ci_low.append(float(np.percentile(medias, 2.5)))
        ci_high.append(float(np.percentile(medias, 97.5)))

    filas = []
    j = 0
    for i, p in enumerate(P_GRID):
        fila = {'red': red, 'protocolo': 'random', 'p': p,
                'S': float(S_mat[:, i].mean()),
                'S_std': float(S_mat[:, i].std()),
                'C_median': np.nan, 'C_median_std': np.nan,
                'C_mean': np.nan, 'C_ci_low': np.nan, 'C_ci_high': np.nan}
        if p in p_c:
            fila['C_median'] = float(Cmed_mat[:, j].mean())
            fila['C_median_std'] = float(Cmed_mat[:, j].std())
            fila['C_mean'] = float(Cmean_mat[:, j].mean())
            fila['C_ci_low'] = ci_low[j]
            fila['C_ci_high'] = ci_high[j]
            j += 1
        filas.append(fila)
    return filas


# ---------------------------------------------------------------------------
# Umbrales p*_topo y p*_svc
# ---------------------------------------------------------------------------

def calcular_umbrales(df_proto):
    """p*_topo y p*_svc a partir de la curva de un protocolo (DataFrame)."""
    df = df_proto.sort_values('p')
    p_topo = np.nan
    for _, row in df.iterrows():
        if row['S'] < UMBRAL_S:
            p_topo = row['p']
            break
    con_c = df.dropna(subset=['C_median'])
    c0 = float(con_c.iloc[0]['C_median'])
    p_svc = np.nan
    for _, row in con_c.iterrows():
        if row['C_median'] < UMBRAL_C * c0:
            p_svc = row['p']
            break
    return p_topo, p_svc, c0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global LOG_FH, N_PARES, R_RANDOM
    parser = argparse.ArgumentParser(
        description='Capacidad de servicio QKD bajo ataques de nodos')
    parser.add_argument('--red', required=True, choices=['cyl', 'adif', 'espana'])
    args = parser.parse_args()
    red = args.red

    # Espec del paper para España: N=2000 pares, random con R=100 (por coste)
    if red == 'espana':
        N_PARES = 2000
        R_RANDOM = 100

    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FH = open(os.path.join(LOG_DIR, f'exp8_capacidad_{red}.log'), 'w')
    t_inicio = time.time()

    log("=" * 70)
    log(f"EXP-8 — Capacidad de servicio bajo ataques — red: {red}")
    log("=" * 70)

    G = CARGADORES[red]()
    n = G.number_of_nodes()
    skrs = [d['skr'] for _, _, d in G.edges(data=True)]
    log(f"  Grafo: |V|={n}, |E|={G.number_of_edges()}, "
        f"SKR aristas: min={min(skrs):.3e}, max={max(skrs):.3e} bits/pulso")

    pares = muestrear_pares(G, red)
    validar_motor(G, pares)

    c0_vals = bottlenecks_pares(G, pares)
    log(f"  C(0): mediana={np.median(c0_vals):.4e}, media={np.mean(c0_vals):.4e}, "
        f"pares con C=0: {int((c0_vals == 0).sum())}/{len(pares)}")

    k_max = k_de_p(P_GRID[-1], n)
    todas_filas = []

    # --- Protocolos deterministas -----------------------------------------
    log("Generando órdenes de eliminación deterministas...")
    t = time.time()
    ordenes = {'degree_static': orden_degree_static(G, k_max),
               'betweenness_static': orden_betweenness_static(G, k_max)}
    log(f"  Órdenes estáticos listos ({time.time() - t:.1f} s)")
    t = time.time()
    ordenes['degree_adaptive'] = orden_adaptativo(G, 'degree', k_max)
    log(f"  Orden degree_adaptive listo ({time.time() - t:.1f} s)")
    t = time.time()
    ordenes['betweenness_adaptive'] = orden_adaptativo(
        G, 'betweenness', k_max, etiqueta='betweenness_adaptive')
    log(f"  Orden betweenness_adaptive listo ({time.time() - t:.1f} s)")

    for proto in ['degree_static', 'degree_adaptive',
                  'betweenness_static', 'betweenness_adaptive']:
        t = time.time()
        filas = medir_curva(G, ordenes[proto], pares, p_medir_c=None)
        for f in filas:
            todas_filas.append({'red': red, 'protocolo': proto, 'p': f['p'],
                                'S': f['S'], 'S_std': np.nan,
                                'C_median': f['C_median'],
                                'C_median_std': np.nan,
                                'C_mean': f['C_mean'],
                                'C_ci_low': np.nan, 'C_ci_high': np.nan})
        log(f"  Protocolo {proto}: medido ({time.time() - t:.1f} s)")

    # --- Protocolo random ---------------------------------------------------
    log(f"Protocolo random: R={R_RANDOM} realizaciones "
        f"(C solo en p ∈ {{0, 0.05, ..., 0.50}})...")
    t = time.time()
    todas_filas.extend(medir_random(G, pares, red))
    log(f"  Protocolo random: medido ({time.time() - t:.1f} s)")

    # --- Salidas --------------------------------------------------------------
    df = pd.DataFrame(todas_filas, columns=[
        'red', 'protocolo', 'p', 'S', 'S_std', 'C_median', 'C_median_std',
        'C_mean', 'C_ci_low', 'C_ci_high'])
    out_curvas = os.path.join(OUT_DIR, f'capacidad_{red}.csv')
    df.to_csv(out_curvas, index=False)
    log(f"Guardado: {out_curvas}")

    filas_umb = []
    log("Umbrales por protocolo (p*_topo | p*_svc | C0_median):")
    for proto in ['random', 'degree_static', 'degree_adaptive',
                  'betweenness_static', 'betweenness_adaptive']:
        p_topo, p_svc, c0 = calcular_umbrales(df[df['protocolo'] == proto])
        filas_umb.append({'red': red, 'protocolo': proto,
                          'p_star_topo': p_topo, 'p_star_svc': p_svc,
                          'C0_median': c0})
        log(f"  {proto:<22} p*_topo={p_topo if not np.isnan(p_topo) else '>0.50'} "
            f" p*_svc={p_svc if not np.isnan(p_svc) else '>0.50'} "
            f" C0_median={c0:.4e}")
        if not np.isnan(p_svc) and not np.isnan(p_topo) and p_svc > p_topo:
            log(f"    [AVISO] {proto}: p*_svc > p*_topo — verificar/reportar")

    out_umb = os.path.join(OUT_DIR, f'capacidad_umbrales_{red}.csv')
    pd.DataFrame(filas_umb).to_csv(out_umb, index=False)
    log(f"Guardado: {out_umb}")

    log(f"Tiempo total ({red}): {time.time() - t_inicio:.1f} s")
    LOG_FH.close()


if __name__ == '__main__':
    main()
