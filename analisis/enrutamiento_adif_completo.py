"""
Enrutamiento consciente de QKD - Cómputo completo para la red ADIF España.

Aplica umbral Delta=45 km: excluye aristas con dist_km > 45.
Los nodos aislados son eliminados. Se trabaja sobre la componente
conexa más grande (LCC) del grafo filtrado.

Compara dos estrategias:
  1. Ruta mínima en saltos (Dijkstra estándar)
  2. Ruta máxima-SKR bottleneck (widest-path Dijkstra)

Modelo físico:
  R(d) = R0 * exp(-d / L_att), L_att = 22.0 km, R0 = 1.0 (normalizado).
  Las distancias ADIF son medidas de fibra real — no se aplica factor rho_f.

Genera:
  - datos/resultados_papers/enrutamiento_adif_allpairs.csv
  - Actualiza datos/resultados_papers/tablas_skr_routing.json  (clave: adif_routing)

Run with:
    cd /Users/igarcia/doctorado/2025_2026/codigo/qkd-net-tools
    python analisis/enrutamiento_adif_completo.py
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

BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_ADIF  = os.path.join(BASE, '..', 'datos', 'adif')
OUT_DIR    = os.path.join(BASE, '..', 'datos', 'resultados_papers')

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Parametros del modelo fisico
# ---------------------------------------------------------------------------
DELTA_KM  = 45.0       # umbral de distancia fibra (km)
L_ATT     = 22.0       # longitud de atenuacion (km)
R0        = 1.0        # SKR normalizada a d=0
ETA_DET   = 0.10       # (nominal, no usado en modelo exponencial simplificado)
P_DARK    = 1e-6       # (nominal, no usado en modelo exponencial simplificado)


# ---------------------------------------------------------------------------
# Haversine (km) — para distancia geografica entre nodos
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Modelo SKR exponencial
# ---------------------------------------------------------------------------
def skr_exp(dist_km, R0=R0, L_att=L_ATT):
    """R(d) = R0 * exp(-d / L_att). Devuelve 0 si d > DELTA_KM."""
    if dist_km > DELTA_KM:
        return 0.0
    return R0 * math.exp(-dist_km / L_att)


# ---------------------------------------------------------------------------
# Carga y construccion del grafo ADIF con umbral Delta=45 km
# ---------------------------------------------------------------------------
def build_adif_graph():
    """
    Carga el grafo ADIF desde los CSV de nodos y adyacencia.
    Filtra aristas con dist_km > DELTA_KM.
    Elimina nodos aislados.
    Devuelve la LCC con atributos lat/lon y SKR por arista.
    """
    nodes_df = pd.read_csv(
        os.path.join(DATA_ADIF, 'nodos_red_adif.csv'),
        quotechar='"', on_bad_lines='skip'
    )
    adj_df = pd.read_csv(
        os.path.join(DATA_ADIF, 'adyacencia_red_adif.csv'),
        quotechar='"', on_bad_lines='skip'
    )

    print(f"  Nodos CSV: {len(nodes_df)} filas, "
          f"Adyacencia CSV: {len(adj_df)} filas")

    # Construir grafo completo (solo nodos conectados)
    G_full = nx.Graph()
    connected_nodes = nodes_df[nodes_df['conectado'] == 'SI'].copy()
    for _, row in connected_nodes.iterrows():
        G_full.add_node(str(row['cod']),
                        lat=float(row['lat']),
                        lon=float(row['lon']),
                        nombre=str(row['nombre']))

    print(f"  Nodos conectados: {len(G_full.nodes())}")

    # Agregar aristas (desduplicar)
    seen = set()
    edges_total = 0
    edges_added = 0
    edges_filtered_delta = 0

    for _, row in adj_df.iterrows():
        u, v = str(row['cod']), str(row['vecino_cod'])
        key = frozenset([u, v])
        if key in seen:
            continue
        seen.add(key)
        edges_total += 1

        if not (G_full.has_node(u) and G_full.has_node(v)):
            continue

        try:
            dist = float(row['dist_km'])
        except (ValueError, TypeError):
            continue

        if dist > DELTA_KM:
            edges_filtered_delta += 1
            continue

        skr = skr_exp(dist)
        G_full.add_edge(u, v,
                        dist_km=dist,
                        SKR=max(skr, 1e-15))
        edges_added += 1

    print(f"  Aristas en CSV (unicas): {edges_total}")
    print(f"  Aristas excluidas (dist_km > {DELTA_KM} km): {edges_filtered_delta}")
    print(f"  Aristas incluidas: {edges_added}")

    # Eliminar nodos aislados
    isolated = list(nx.isolates(G_full))
    G_full.remove_nodes_from(isolated)
    print(f"  Nodos aislados eliminados: {len(isolated)}")
    print(f"  Grafo filtrado: |V|={G_full.number_of_nodes()}, "
          f"|E|={G_full.number_of_edges()}")

    # Verificar conectividad
    comps = sorted(nx.connected_components(G_full), key=len, reverse=True)
    n_comps = len(comps)
    lcc_size = len(comps[0])
    print(f"  Componentes conexas: {n_comps}")
    print(f"  LCC size: {lcc_size} nodos")

    is_connected = (n_comps == 1)
    if not is_connected:
        print(f"  [AVISO] Grafo DESCONECTADO — se usa la LCC ({lcc_size} nodos)")

    G_lcc = G_full.subgraph(comps[0]).copy()

    # Construir diccionario de coordenadas
    coords = {n: (G_lcc.nodes[n]['lat'], G_lcc.nodes[n]['lon'])
              for n in G_lcc.nodes()}

    return G_lcc, coords, {
        'n_nodes_full': G_full.number_of_nodes() + len(isolated),
        'n_edges_csv': edges_total,
        'n_edges_added': edges_added,
        'n_edges_filtered': edges_filtered_delta,
        'n_isolated': len(isolated),
        'n_components': n_comps,
        'lcc_size': lcc_size,
        'lcc_edges': G_lcc.number_of_edges(),
        'pct_edges_excluded': round(100.0 * edges_filtered_delta / max(edges_total, 1), 2),
    }


# ---------------------------------------------------------------------------
# Widest-path (max-min bottleneck) Dijkstra — single source
# ---------------------------------------------------------------------------
def widest_path_dijkstra(G, source, weight='SKR'):
    """
    Dijkstra variante max-min bottleneck.
    Devuelve:
      bottleneck[v]  — max bottleneck SKR desde source hasta v
      prev[v]        — predecesor para reconstruir camino
    """
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
    """Reconstruye el camino desde source hasta target usando el diccionario prev."""
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    if not path or path[0] != source:
        return []
    return path


# ---------------------------------------------------------------------------
# All-pairs computation
# ---------------------------------------------------------------------------
def compute_all_pairs(G, coords, label="adif"):
    """
    Computa todas las rutas all-pairs (unordered) sobre la LCC.
    Devuelve DataFrame con metricas por par.
    """
    nodes = list(G.nodes())
    N = len(nodes)
    n_pairs = N * (N - 1) // 2
    print(f"  [{label}] Nodos: {N}, Pares: {n_pairs:,}")

    t0 = time.time()
    print(f"  [{label}] Calculando rutas all-pairs (widest-path + hop)...")

    rows = []
    for i, src in enumerate(nodes):
        if i % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(i, 1) * (N - i) if i > 0 else 0
            print(f"    source {i}/{N}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                  flush=True)

        # Widest-path from src
        btl_wide, prev_wide = widest_path_dijkstra(G, src, weight='SKR')

        # Hop-count Dijkstra from src (unit weights)
        hop_lengths, hop_paths = nx.single_source_dijkstra(
            G, src, weight=lambda u, v, d: 1
        )

        lat_s, lon_s = coords.get(src, (None, None))

        for tgt in nodes:
            if tgt <= src:   # process only i < j (string comparison)
                continue
            if tgt not in hop_lengths:
                continue
            if btl_wide.get(tgt, -1) < 0:
                continue

            # Hop-count path metrics
            h_path = hop_paths[tgt]
            hop_n  = len(h_path) - 1
            skr_hop = (min(G[h_path[k]][h_path[k+1]].get('SKR', 0)
                           for k in range(len(h_path) - 1))
                       if hop_n > 0 else 0.0)

            # Widest-path metrics
            w_path = reconstruct_path(prev_wide, src, tgt)
            if not w_path:
                continue
            hop_w = len(w_path) - 1
            skr_w = btl_wide[tgt]

            # Improvement factor
            I  = skr_w / skr_hop if skr_hop > 0 else np.nan
            dh = hop_w - hop_n

            # Geographic distance (haversine, no routing factor)
            lat_t, lon_t = coords.get(tgt, (None, None))
            if lat_s is not None and lat_t is not None:
                dist_geo = haversine(lat_s, lon_s, lat_t, lon_t)
            else:
                dist_geo = np.nan

            rows.append({
                'node_s':        src,
                'node_t':        tgt,
                'dist_km':       round(dist_geo, 2) if not np.isnan(dist_geo) else np.nan,
                'hop_shortest':  hop_n,
                'hop_widest':    hop_w,
                'delta_h':       dh,
                'skr_hop':       skr_hop,
                'skr_widest':    skr_w,
                'I':             I,
            })

    t_total = time.time() - t0
    print(f"  [{label}] All-pairs done in {t_total:.1f}s")
    df = pd.DataFrame(rows)
    print(f"  [{label}] Pares computados: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Estadisticas agregadas
# ---------------------------------------------------------------------------
def aggregate_stats(df, label="adif"):
    I  = df['I'].dropna()
    dh = df['delta_h'].dropna()

    stats = {
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
    return stats


def geo_stratified_stats(df):
    """Estadisticas estratificadas por distancia geografica (mismos bins que Table 2)."""
    bins   = [0, 100, 300, 500, 700, 1e9]
    labels = ['[0,100)', '[100,300)', '[300,500)', '[500,700)', '>700']
    df = df.copy()
    df['dist_bin'] = pd.cut(df['dist_km'], bins=bins, labels=labels, right=False)

    rows = []
    for lbl in labels:
        sub = df[df['dist_bin'] == lbl]
        if sub.empty:
            continue
        I  = sub['I'].dropna()
        dh = sub['delta_h'].dropna()
        rows.append({
            'bin':      lbl,
            'n_pairs':  len(sub),
            'I_mean':   round(float(I.mean()), 4),
            'I_median': round(float(I.median()), 4),
            'I_max':    round(float(I.max()), 4),
            'dh_mean':  round(float(dh.mean()), 4),
        })

    # All pairs
    I_all  = df['I'].dropna()
    dh_all = df['delta_h'].dropna()
    rows.append({
        'bin':      'All',
        'n_pairs':  len(df),
        'I_mean':   round(float(I_all.mean()), 4),
        'I_median': round(float(I_all.median()), 4),
        'I_max':    round(float(I_all.max()), 4),
        'dh_mean':  round(float(dh_all.mean()), 4),
    })

    return rows


# ---------------------------------------------------------------------------
# Extremos
# ---------------------------------------------------------------------------
def print_extreme_pairs(df, G):
    """Imprime los pares extremos mas interesantes."""
    valid = df[df['I'].notna()].copy()

    # Par con mayor I
    row_max_I = valid.loc[valid['I'].idxmax()]

    # Par con menor I > 1.0 (menor ganancia positiva)
    above1 = valid[valid['I'] > 1.0]
    row_min_I = above1.loc[above1['I'].idxmin()] if not above1.empty else None

    # Par con mayor delta_h
    row_max_dh = valid.loc[valid['delta_h'].idxmax()]

    print("\n" + "="*70)
    print("EXTREME PAIRS")
    print("="*70)

    def _fmt(row, label):
        print(f"\n  {label}:")
        s, t = row['node_s'], row['node_t']
        sn = G.nodes[s].get('nombre', s) if s in G.nodes else s
        tn = G.nodes[t].get('nombre', t) if t in G.nodes else t
        print(f"    Pair: {s} ({sn}) <-> {t} ({tn})")
        print(f"    Geo dist:      {row['dist_km']:.1f} km")
        print(f"    Hops (hop):    {int(row['hop_shortest'])}")
        print(f"    Hops (widest): {int(row['hop_widest'])}")
        print(f"    delta_h:       {int(row['delta_h'])}")
        print(f"    SKR_hop:       {row['skr_hop']:.4e}")
        print(f"    SKR_widest:    {row['skr_widest']:.4e}")
        print(f"    I:             {row['I']:.4f}")

    _fmt(row_max_I,  "Highest I (max improvement)")
    if row_min_I is not None:
        _fmt(row_min_I,  "Lowest I > 1.0 (min positive gain)")
    _fmt(row_max_dh, "Most extra hops (max delta_h)")


# ---------------------------------------------------------------------------
# Print tables
# ---------------------------------------------------------------------------
def print_table1(s, lcc_info):
    print("\n" + "="*70)
    print("TABLE — Aggregate routing statistics (ADIF, Δ=45 km)")
    print("="*70)
    print(f"  LCC: |V|={lcc_info['lcc_size']}, |E|={lcc_info['lcc_edges']}")
    print(f"  Pairs: {s['n_pairs']:,}")
    print(f"  Edges excluded (dist_km > 45): {lcc_info['n_edges_filtered']} "
          f"({lcc_info['pct_edges_excluded']:.1f}%)")
    print()
    print(f"  {'Statistic':<20} {'I(s,t)':>12} {'Δh(s,t)':>12}")
    print("  " + "-"*45)
    print(f"  {'Mean':<20} {s['I_mean']:>12.4f} {s['dh_mean']:>12.4f}")
    print(f"  {'Median':<20} {s['I_median']:>12.4f} {s['dh_median']:>12.4f}")
    print(f"  {'Std':<20} {s['I_std']:>12.4f} {s['dh_std']:>12.4f}")
    print(f"  {'P90':<20} {s['I_P90']:>12.4f} {s['dh_P90']:>12.4f}")
    print(f"  {'P99':<20} {s['I_P99']:>12.4f} {s['dh_P99']:>12.4f}")
    print(f"  {'Max':<20} {s['I_max']:>12.4f} {s['dh_max']:>12}")


def print_table2(geo_rows):
    print("\n" + "="*75)
    print("TABLE — Geographic distance stratification (ADIF)")
    print("="*75)
    print(f"  {'Distance bin':<16} {'Pairs':>10} {'Mean I':>8} "
          f"{'Median I':>10} {'Max I':>8} {'Mean Δh':>10}")
    print("  " + "-"*65)
    for r in geo_rows:
        print(f"  {r['bin']:<16} {r['n_pairs']:>10,} {r['I_mean']:>8.4f} "
              f"{r['I_median']:>10.4f} {r['I_max']:>8.4f} {r['dh_mean']:>10.4f}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    t_start = time.time()
    print("=" * 70)
    print("QKD-Aware Routing — ADIF All-Pairs Computation (Δ=45 km)")
    print("=" * 70)

    # ---- Step 1: Build graph ----
    print("\n[1/4] Building ADIF graph (Delta=45 km threshold)...")
    G, coords, lcc_info = build_adif_graph()

    print(f"\n  === Network after filtering ===")
    print(f"  Original connected nodes (CSV): {lcc_info['n_nodes_full']}")
    print(f"  Edges in CSV (unique):          {lcc_info['n_edges_csv']}")
    print(f"  Edges excluded (>45 km):        {lcc_info['n_edges_filtered']} "
          f"({lcc_info['pct_edges_excluded']:.1f}%)")
    print(f"  Isolated nodes removed:         {lcc_info['n_isolated']}")
    print(f"  Connected components:           {lcc_info['n_components']}")
    print(f"  LCC |V|:                        {lcc_info['lcc_size']}")
    print(f"  LCC |E|:                        {lcc_info['lcc_edges']}")

    # ---- Step 2: All-pairs routing ----
    print("\n[2/4] Computing all-pairs routing...")
    t_comp = time.time()
    df = compute_all_pairs(G, coords, label="adif")
    t_comp = time.time() - t_comp

    # Save pair-level CSV
    out_csv = os.path.join(OUT_DIR, 'enrutamiento_adif_allpairs.csv')
    df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    # ---- Step 3: Aggregate statistics ----
    print("\n[3/4] Computing statistics...")
    s = aggregate_stats(df, label="adif")
    geo_rows = geo_stratified_stats(df)

    print_table1(s, lcc_info)
    print_table2(geo_rows)
    print_extreme_pairs(df, G)

    # ---- Comparison with Spain ----
    SPAIN_I_MEAN  = 2.8811
    SPAIN_I_MED   = 2.9780
    SPAIN_DH_MEAN = 45.2047
    print("\n" + "="*70)
    print("COMPARISON vs SPAIN (PAM results)")
    print("="*70)
    print(f"  {'Metric':<20} {'ADIF':>12} {'Spain':>12} {'Diff':>12}")
    print("  " + "-"*57)
    print(f"  {'Mean I':<20} {s['I_mean']:>12.4f} {SPAIN_I_MEAN:>12.4f} "
          f"{s['I_mean']-SPAIN_I_MEAN:>+12.4f}")
    print(f"  {'Median I':<20} {s['I_median']:>12.4f} {SPAIN_I_MED:>12.4f} "
          f"{s['I_median']-SPAIN_I_MED:>+12.4f}")
    print(f"  {'Mean Δh':<20} {s['dh_mean']:>12.4f} {SPAIN_DH_MEAN:>12.4f} "
          f"{s['dh_mean']-SPAIN_DH_MEAN:>+12.4f}")

    # ---- Step 4: Update JSON ----
    print("\n[4/4] Updating JSON results file...")
    json_path = os.path.join(OUT_DIR, 'tablas_skr_routing.json')

    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    else:
        json_data = {}

    json_data['adif_routing'] = {
        'metadata': {
            'generated':      time.strftime('%Y-%m-%dT%H:%M:%S'),
            'network':        'ADIF junction graph',
            'delta_km':       DELTA_KM,
            'L_att_km':       L_ATT,
            'R0':             R0,
            'eta_det':        ETA_DET,
            'p_dark':         P_DARK,
            'rho_f':          1.0,
            'note':           'ADIF distances are real fibre lengths; no routing factor applied',
            'n_pairs':        s['n_pairs'],
            'runtime_s':      round(t_comp, 1),
        },
        'network_stats': lcc_info,
        'aggregate':     s,
        'geographic':    geo_rows,
        'spain_comparison': {
            'spain_I_mean':    SPAIN_I_MEAN,
            'adif_I_mean':     s['I_mean'],
            'diff_I_mean':     round(s['I_mean'] - SPAIN_I_MEAN, 4),
            'spain_dh_mean':   SPAIN_DH_MEAN,
            'adif_dh_mean':    s['dh_mean'],
            'diff_dh_mean':    round(s['dh_mean'] - SPAIN_DH_MEAN, 4),
        },
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    # ---- Final summary ----
    t_total = time.time() - t_start
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    print(f"  Total runtime: {t_total:.1f}s")
    print(f"  Pairs computed: {s['n_pairs']:,}")
    print(f"  LCC: |V|={lcc_info['lcc_size']}, |E|={lcc_info['lcc_edges']}")
    print(f"  Mean I = {s['I_mean']:.4f}, Mean Δh = {s['dh_mean']:.4f}")
