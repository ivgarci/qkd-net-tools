"""
Fallos de dispositivo en redes QKD de relés de confianza (paper P8, Fault-Aware QKD).

Experimento EXP-11. Modela fallos a nivel de dispositivo (detector) y propaga su
efecto al nivel de red SIN eliminar nodos ni aristas: la topología permanece
intacta (S=1) mientras la capacidad de servicio C se degrada.

Modelos de fallo (espec PLAN_PAPER.md, cerrada):
  M1 — degradación de eficiencia:   η'_det  = (1−δ)·η_det
  M2 — inyección de cuentas oscuras: p'_dark = κ·p_dark
  M3 — exceso de QBER:               e'_det  = e_det + Δe
  Corte: SKR = 0 si QBER ≥ 0.11 o si la fórmula da ≤ 0.
  Locus: un fallo en el relé v degrada TODAS las aristas incidentes a v
  (supuesto: el detector del receptor está en el extremo del enlace). Si los
  dos extremos de una arista están degradados se aplica el peor caso de cada
  parámetro (max δ, max κ, max Δe).

Escenarios:
  S1 — envejecimiento de flota: δ uniforme en todos los relés,
       δ ∈ {0.0, 0.1, ..., 0.9} → fallos_s1.csv
  S2 — fallos localizados aleatorios: fracción f de relés con fallo severo
       (M1 δ=0.9 y M2 κ según sanity-check), R realizaciones; cota de
       comparación: ELIMINACIÓN de los mismos nodos (metodología P7)
       → fallos_s2.csv

Métrica: C = mediana del bottleneck widest-path sobre los pares fijos de P7
(datos/resultados_papers/pares_muestreo_<red>.csv, semilla 42 — NO se
remuestrean). Motor widest-path reutilizado de capacidad_servicio_ataques.py.

Genera:
  datos/resultados_papers/fallos_sanity_skr.csv
  datos/resultados_papers/fallos_s1.csv
  datos/resultados_papers/fallos_s2.csv
  logs/exp11_fallos_dispositivo.log (o $QKD_LOG_DIR)

Uso:
    python analisis/fallos_dispositivo.py
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx

BASE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.abspath(os.path.join(BASE, '..'))
DATA    = os.path.join(ROOT, 'datos')
OUT_DIR = os.path.join(DATA, 'resultados_papers')
LOG_DIR = os.environ.get('QKD_LOG_DIR', os.path.join(ROOT, 'logs'))
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, ROOT)
sys.path.insert(0, BASE)

# Maquinaria canónica existente — NO se duplica
from protocols.skr_bb84 import (skr_bb84_asymptotic, qber,       # noqa: E402
                                ETA_DET, P_DARK, E_DETECTOR)
from capacidad_servicio_ataques import (CARGADORES,               # noqa: E402
                                        bottlenecks_pares)

# ---------------------------------------------------------------------------
# Parámetros del experimento (espec cerrada del plan)
# ---------------------------------------------------------------------------

QBER_CUT      = 0.11                                   # corte de seguridad BB84
S1_DELTAS     = [round(0.1 * i, 1) for i in range(10)]  # 0.0 .. 0.9
S2_FRACCIONES = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
S2_R          = 100                                    # realizaciones, semillas 0..R-1
S2_DELTA      = 0.9                                    # M1 severo
S2_KAPPA_BASE = 1e3                                    # M2 severo (plan)
SANITY_DS     = [10.0, 30.0, 45.0]                     # km
SANITY_DELTAS = [0.3, 0.6, 0.9]
SANITY_KAPPAS = [10.0, 100.0, 1000.0]
SANITY_DES    = [0.01, 0.02, 0.05]
PRESUPUESTO_S = 3 * 3600                               # 3 h → si se proyecta más, R=50

LOG_FH = None


def log(msg):
    linea = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(linea, flush=True)
    if LOG_FH is not None:
        LOG_FH.write(linea + '\n')
        LOG_FH.flush()


# ---------------------------------------------------------------------------
# Modelo de fallo: SKR con parámetros de dispositivo degradados
# ---------------------------------------------------------------------------

def skr_con_fallo(dist_km, delta=0.0, kappa=1.0, d_e=0.0):
    """
    SKR BB84 asintótico ideal con dispositivo degradado:
      M1: η' = (1−δ)·η_det ;  M2: p' = κ·p_dark ;  M3: e' = e_det + Δe.
    Corte: SKR = 0 si QBER ≥ 0.11 o si la fórmula da ≤ 0.
    """
    eta = (1.0 - delta) * ETA_DET
    p_d = kappa * P_DARK
    e_d = E_DETECTOR + d_e
    if eta <= 0.0:
        return 0.0
    if qber(dist_km, eta, p_d, e_d) >= QBER_CUT:
        return 0.0
    return skr_bb84_asymptotic(
        dist_km, eta_det=eta, p_dark=p_d, e_det=e_d
    )


def aplicar_fallos(G, fallos):
    """
    fallos: dict nodo → (delta, kappa, d_e). Devuelve la lista de
    (u, v, skr_original, skr_degradado) de las aristas afectadas (las
    incidentes a algún nodo con fallo) y MODIFICA G en sitio.
    Si ambos extremos fallan: peor caso por parámetro (max δ, max κ, max Δe).
    """
    sano = (0.0, 1.0, 0.0)
    afectadas = []
    vistas = set()
    for v in fallos:
        for u in G.neighbors(v):
            ek = frozenset((u, v))
            if ek in vistas:
                continue
            vistas.add(ek)
            fu = fallos.get(u, sano)
            fv = fallos.get(v, sano)
            delta = max(fu[0], fv[0])
            kappa = max(fu[1], fv[1])
            d_e   = max(fu[2], fv[2])
            skr0 = G[u][v]['skr']
            skr1 = skr_con_fallo(G[u][v]['dist_km'], delta, kappa, d_e)
            G[u][v]['skr'] = skr1
            afectadas.append((u, v, skr0, skr1))
    return afectadas


def restaurar(G, afectadas):
    for u, v, skr0, _ in afectadas:
        G[u][v]['skr'] = skr0


# ---------------------------------------------------------------------------
# Carga de redes: grafo canónico + dist_km por arista + pares fijos de P7
# ---------------------------------------------------------------------------

def cargar_red(red):
    """Grafo de CARGADORES con 'skr' y 'dist_km' en cada arista + pares P7."""
    G = CARGADORES[red]()

    # CyL / España: los cargadores asignan 'skr' desde skr_per_link.csv pero
    # NO conservan dist_km → se recupera del mismo CSV. ADIF ya lleva dist_km.
    sin_dist = [(u, v) for u, v, d in G.edges(data=True) if 'dist_km' not in d]
    if sin_dist:
        caso = {'cyl': 'CyL', 'espana': 'España'}[red]
        df = pd.read_csv(os.path.join(DATA, 'skr_per_link.csv'))
        df = df[df['caso'] == caso]
        tabla = {frozenset((r.nodo_u, r.nodo_v)): float(r.dist_km)
                 for r in df.itertuples()}
        faltan = [
            (u, v) for u, v in sin_dist
            if frozenset((u, v)) not in tabla
        ]
        if faltan:
            muestra = ', '.join(f'{u!r}--{v!r}' for u, v in faltan[:5])
            raise ValueError(
                f"{red}: {len(faltan)} aristas sin distancia en "
                f"datos/skr_per_link.csv; primeras: {muestra}"
            )
        for u, v in sin_dist:
            G[u][v]['dist_km'] = tabla[frozenset((u, v))]
        log(f"  dist_km recuperada de skr_per_link.csv para "
            f"{len(sin_dist)} aristas")

    # Consistencia: SKR recalculado con parámetros sanos == SKR almacenado
    desv = max(abs(skr_con_fallo(d['dist_km']) - d['skr'])
               for _, _, d in G.edges(data=True))
    if desv > 1e-12:
        raise RuntimeError(f"{red}: SKR recalculado != almacenado "
                           f"(desv máx {desv:.3e})")
    log(f"  Consistencia SKR(dist) sano vs almacenado: OK (desv máx {desv:.1e})")

    # Pares fijos de P7 — NO se remuestrean (comparabilidad)
    df_p = pd.read_csv(os.path.join(OUT_DIR, f'pares_muestreo_{red}.csv'),
                       dtype={'nodo_u': str, 'nodo_v': str})
    pares = list(zip(df_p['nodo_u'], df_p['nodo_v']))
    fuera = [p for p in pares for x in p if x not in G]
    if fuera:
        raise RuntimeError(f"{red}: {len(fuera)} pares con nodos fuera del grafo")
    log(f"  Pares fijos P7 cargados: {len(pares)}")
    return G, pares


def medir_C(G, pares):
    """C_median, C_mean, frac de pares con bottleneck 0, y S (frac LCC)."""
    vals = bottlenecks_pares(G, pares)
    S = (len(max(nx.connected_components(G), key=len)) / G.number_of_nodes()
         if G.number_of_nodes() else 0.0)
    return (float(np.median(vals)), float(np.mean(vals)),
            float(np.mean(vals == 0.0)), S)


# ---------------------------------------------------------------------------
# Sanity-check del modelo: tabla SKR(d) bajo M1/M2/M3
# ---------------------------------------------------------------------------

def sanity_check():
    """Tabla SKR(d) sano vs degradado. Decide si añadir κ=10⁴ a S2."""
    log("=" * 70)
    log("SANITY-CHECK del modelo de fallo: SKR(d) bajo M1/M2/M3")
    log("=" * 70)
    configs = [('sano', '-', 0.0, 1.0, 0.0)]
    configs += [('M1', f'delta={x}', x, 1.0, 0.0) for x in SANITY_DELTAS]
    configs += [('M2', f'kappa={x:g}', 0.0, x, 0.0) for x in SANITY_KAPPAS]
    configs += [('M3', f'de={x}', 0.0, 1.0, x) for x in SANITY_DES]

    skr_sano = {d: skr_con_fallo(d) for d in SANITY_DS}
    filas = []
    for modelo, param, delta, kappa, d_e in configs:
        for d in SANITY_DS:
            eta = (1 - delta) * ETA_DET
            q = qber(d, eta, kappa * P_DARK, E_DETECTOR + d_e)
            s = skr_con_fallo(d, delta, kappa, d_e)
            rel = s / skr_sano[d] if skr_sano[d] > 0 else np.nan
            filas.append({'modelo': modelo, 'parametro': param, 'd_km': d,
                          'qber': q, 'skr_bits_pulso': s, 'skr_rel_sano': rel})

    # Regla del plan: si con κ=1000 el SKR a d=10 km sigue > 10 % del sano,
    # añadir κ=10⁴ al barrido de S2/S3 (y a esta tabla) y documentarlo.
    rel_k1000 = skr_con_fallo(10.0, kappa=1e3) / skr_sano[10.0]
    log(f"  SKR(10 km, kappa=1e3) / SKR_sano = {rel_k1000:.3f}")
    anadir_k4 = rel_k1000 > 0.10
    if anadir_k4:
        log("  [DECISIÓN] kappa=1e3 NO basta (>10% del sano a 10 km) → "
            "se añade kappa=1e4 a la tabla y al barrido de S2 (espec del plan)")
        for d in SANITY_DS:
            q = qber(d, ETA_DET, 1e4 * P_DARK, E_DETECTOR)
            s = skr_con_fallo(d, kappa=1e4)
            rel = s / skr_sano[d] if skr_sano[d] > 0 else np.nan
            filas.append({'modelo': 'M2', 'parametro': 'kappa=10000',
                          'd_km': d, 'qber': q, 'skr_bits_pulso': s,
                          'skr_rel_sano': rel})

    df = pd.DataFrame(filas)
    out = os.path.join(OUT_DIR, 'fallos_sanity_skr.csv')
    df.to_csv(out, index=False)
    log(f"Guardado: {out}")
    for _, r in df.iterrows():
        log(f"  {r['modelo']:<4} {r['parametro']:<12} d={r['d_km']:4.0f} km  "
            f"QBER={r['qber']:.4f}  SKR={r['skr_bits_pulso']:.3e}  "
            f"rel={r['skr_rel_sano']:.3f}")
    return anadir_k4


# ---------------------------------------------------------------------------
# S1 — envejecimiento de flota (δ uniforme en todos los relés)
# ---------------------------------------------------------------------------

def escenario_s1(red, G, pares, c0):
    log("-" * 70)
    log(f"S1 ({red}) — envejecimiento de flota: delta uniforme en {len(G)} relés")
    filas = []
    delta_star = None
    for delta in S1_DELTAS:
        t = time.time()
        fallos = {v: (delta, 1.0, 0.0) for v in G.nodes()}
        afectadas = aplicar_fallos(G, fallos)
        c_med, c_mean, frac0, S = medir_C(G, pares)
        restaurar(G, afectadas)
        c_rel = c_med / c0 if c0 > 0 else np.nan
        if delta_star is None and c_med < 0.5 * c0:
            delta_star = delta
        filas.append({'red': red, 'delta': delta, 'C_median': c_med,
                      'C_rel': c_rel, 'frac_pares_cero': frac0})
        log(f"  delta={delta:.1f}  C={c_med:.4e}  C/C0={c_rel:.4f}  "
            f"frac_pares_0={frac0:.4f}  S={S:.3f}  ({time.time()-t:.1f} s)")
        if S < 1.0:
            log(f"    [AVISO] S<1 en modo fallo — inesperado, investigar")
    # Monotonía esperada
    cs = [f['C_median'] for f in filas]
    if any(cs[i+1] > cs[i] + 1e-15 for i in range(len(cs) - 1)):
        log(f"  [AVISO] C(delta) no monótona decreciente — investigar")
    else:
        log(f"  Monotonía C(delta) decreciente: OK")
    log(f"  delta* (C < 0.5·C0) = "
        f"{delta_star if delta_star is not None else '>0.9'}")
    return filas, delta_star


# ---------------------------------------------------------------------------
# S2 — fallos localizados aleatorios vs eliminación (cota P7)
# ---------------------------------------------------------------------------

def escenario_s2(red, G, pares, c0, modos, R):
    """
    Para cada f y cada realización r: k=max(1, round(f·n)) relés con fallo
    severo. Modo fallo (parámetros degradados, topología intacta) vs
    eliminación de los MISMOS nodos (cota P7). La eliminación se evalúa una
    sola vez por (f, r) y se reutiliza entre modos (mismo conjunto de nodos).
    """
    log("-" * 70)
    log(f"S2 ({red}) — fallos localizados: modos={[m[0] for m in modos]}, "
        f"R={R}, f={S2_FRACCIONES}")
    n = G.number_of_nodes()
    nodos = sorted(G.nodes(), key=str)
    filas = []
    violaciones = 0
    s_min_fallo = 1.0

    for f in S2_FRACCIONES:
        k = max(1, int(np.floor(f * n + 0.5)))
        t_f = time.time()
        # conjuntos de nodos por realización (semillas 0..R-1) — compartidos
        conjuntos = [
            [nodos[i] for i in np.random.default_rng(r).permutation(n)[:k]]
            for r in range(R)]

        # cota P7: eliminación de los mismos nodos (independiente del modo)
        c_rem = np.empty(R)
        for r, conj in enumerate(conjuntos):
            H = nx.restricted_view(G, nodes=conj, edges=[])
            c_rem[r] = float(np.median(bottlenecks_pares(H, pares)))
        c_rem_rel = c_rem / c0

        for nombre_modo, params in modos:
            c_fault = np.empty(R)
            frac0 = np.empty(R)
            for r, conj in enumerate(conjuntos):
                fallos = {v: params for v in conj}
                afectadas = aplicar_fallos(G, fallos)
                c_med, _, fr0, S = medir_C(G, pares)
                restaurar(G, afectadas)
                c_fault[r] = c_med
                frac0[r] = fr0
                s_min_fallo = min(s_min_fallo, S)
                # cota: el fallo nunca daña más que la eliminación del mismo set
                if c_med < c_rem[r] - 1e-15:
                    violaciones += 1
                    log(f"    [VIOLACIÓN COTA] {nombre_modo} f={f} r={r}: "
                        f"C_fallo={c_med:.3e} < C_elim={c_rem[r]:.3e}")
            cf_rel = c_fault / c0
            filas.append({
                'red': red, 'modo': nombre_modo, 'f': f, 'k_nodos': k,
                'C_fault_rel_mean': float(cf_rel.mean()),
                'C_fault_rel_std': float(cf_rel.std()),
                'C_removal_rel_mean': float(c_rem_rel.mean()),
                'C_removal_rel_std': float(c_rem_rel.std()),
                'frac_pares_cero_mean': float(frac0.mean()),
                'ratio_fallo_eliminacion': float(
                    (1.0 - cf_rel.mean()) / (1.0 - c_rem_rel.mean()))
                    if c_rem_rel.mean() < 1.0 else np.nan,
            })
            log(f"  f={f:.2f} (k={k:3d}) {nombre_modo:<10} "
                f"C_fallo/C0={cf_rel.mean():.4f}±{cf_rel.std():.4f}  "
                f"C_elim/C0={c_rem_rel.mean():.4f}±{c_rem_rel.std():.4f}  "
                f"frac0={frac0.mean():.4f}  ({time.time()-t_f:.0f} s)")
    log(f"  S(p) mínimo observado en modo fallo: {s_min_fallo:.4f} "
        f"{'== 1 (degradación INVISIBLE a la topología) — OK' if s_min_fallo >= 1.0 else '— [AVISO] S<1, investigar'}")
    log(f"  Violaciones de la cota C_fallo >= C_elim: {violaciones}"
        f"{' — OK' if violaciones == 0 else ' — BUG, investigar'}")
    return filas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global LOG_FH
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FH = open(os.path.join(LOG_DIR, 'exp11_fallos_dispositivo.log'), 'w')
    t_inicio = time.time()
    log("=" * 70)
    log("EXP-11 — Fallos de dispositivo → capacidad de red (paper P8)")
    log(f"Calibración base: eta_det={ETA_DET}, p_dark={P_DARK}, "
        f"e_det={E_DETECTOR}; corte QBER >= {QBER_CUT}")
    log("=" * 70)

    # 1. Sanity-check del modelo (decide kappa de S2)
    anadir_k4 = sanity_check()
    modos = [('M1_d0.9', (S2_DELTA, 1.0, 0.0)),
             ('M2_k1e3', (0.0, S2_KAPPA_BASE, 0.0))]
    if anadir_k4:
        modos.append(('M2_k1e4', (0.0, 1e4, 0.0)))
    log(f"Modos S2: {[m[0] for m in modos]}")

    filas_s1, filas_s2 = [], []
    deltas_star = {}

    for red in ['cyl', 'espana', 'adif']:
        log("=" * 70)
        log(f"RED: {red}")
        G, pares = cargar_red(red)
        log(f"  |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}, "
            f"conexo={nx.is_connected(G)}")

        # C(0) se deriva del grafo, del modelo y de los pares versionados.
        t = time.time()
        c0, c0_mean, frac0, S0 = medir_C(G, pares)
        t_eval = time.time() - t
        if not np.isfinite(c0) or c0 <= 0.0:
            raise RuntimeError(
                f"{red}: C(0) inválida ({c0!r}); revisar modelo, grafo y pares"
            )
        log(f"  C(0) derivada={c0:.4e}; "
            f"S(0)={S0:.3f}; t_eval_C={t_eval:.2f} s")

        # Presupuesto: nº de evaluaciones de C en S2 con esta red
        R = S2_R
        n_eval = len(S2_FRACCIONES) * R * (1 + len(modos))  # elim + modos
        proyeccion = n_eval * t_eval + len(S1_DELTAS) * t_eval
        log(f"  Proyección S1+S2: ~{n_eval + len(S1_DELTAS)} evaluaciones de C "
            f"≈ {proyeccion/60:.1f} min")
        if proyeccion > PRESUPUESTO_S:
            R = 50
            log(f"  [DECISIÓN] proyección > 3 h → R reducido a {R} "
                f"(documentado, espec del plan)")

        f_s1, d_star = escenario_s1(red, G, pares, c0)
        filas_s1.extend(f_s1)
        deltas_star[red] = d_star
        filas_s2.extend(escenario_s2(red, G, pares, c0, modos, R))

    out_s1 = os.path.join(OUT_DIR, 'fallos_s1.csv')
    pd.DataFrame(filas_s1).to_csv(out_s1, index=False)
    log(f"Guardado: {out_s1}")
    out_s2 = os.path.join(OUT_DIR, 'fallos_s2.csv')
    pd.DataFrame(filas_s2).to_csv(out_s2, index=False)
    log(f"Guardado: {out_s2}")

    log("=" * 70)
    log(f"delta* por red (C < 0.5·C0): {deltas_star}")
    log(f"Tiempo total: {(time.time() - t_inicio)/60:.1f} min")
    log("EXP-11 completado.")
    LOG_FH.close()


if __name__ == '__main__':
    main()
