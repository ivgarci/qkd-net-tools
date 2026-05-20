"""
Modelo de tasa de clave secreta (SKR) para QKD con protocolo BB84 con decoy states.

Referencia principal:
  Lo, H.-K., Ma, X., & Chen, K. (2005). Decoy state quantum key distribution.
  Physical Review Letters, 94(23), 230504.

Calibración:
  Los parámetros por defecto reproducen el orden de magnitud reportado en
  Sasaki et al. (2011) Tokyo QKD Network: SKR ≈ 10⁻³ a 10⁻⁴ bits/pulso a 50 km.

Genera:
  figuras/skr_vs_distancia.pdf/.png  — curva SKR(d) en escala log
  datos/skr_per_link.csv             — SKR por arista para CyL, España y ADIF
"""

import os
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_CYL  = os.path.join(BASE, '..', 'datos', 'cyl')
DATA_ESP  = os.path.join(BASE, '..', 'datos', 'espana')
DATA_ADIF = os.path.join(BASE, '..', 'datos', 'adif')
FIGS_OUT  = os.path.join(BASE, '..', 'figuras')
os.makedirs(FIGS_OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Parámetros físicos del canal y protocolo
# ---------------------------------------------------------------------------

ALPHA_DB_KM  = 0.2    # Pérdida fibra monomodo @ 1550 nm (dB/km)
ETA_DET      = 0.10   # Eficiencia del detector APD (10 %)
P_DARK       = 1e-6   # Probabilidad de conteo oscuro por pulso
MU           = 0.5    # Intensidad media pulso señal (fotones/pulso)
MU_DECOY     = 0.1    # Intensidad media pulso decoy
E_DETECTOR   = 0.015  # QBER intrínseca del detector (1,5 %)
F_EC         = 1.16   # Factor de eficiencia de corrección de errores (Shannon ≥ 1)


def channel_transmittance(distance_km: float, eta_det: float = ETA_DET) -> float:
    """T(d) = η_det · 10^(−α·d/10). Transmitancia total del canal."""
    return eta_det * 10 ** (-ALPHA_DB_KM * distance_km / 10)


def qber(distance_km: float, eta_det: float = ETA_DET,
         p_dark: float = P_DARK, e_det: float = E_DETECTOR) -> float:
    """
    QBER estimada incluyendo errores ópticos (e_det) y conteos oscuros.
    e = (e_det·T + p_dark/2) / (T + p_dark)
    """
    T = channel_transmittance(distance_km, eta_det)
    num = e_det * T + p_dark / 2
    den = T + p_dark
    if den <= 0:
        return 0.5
    return min(num / den, 0.5)


def h2(p: float) -> float:
    """Entropía binaria h(p) = -p·log2(p) - (1-p)·log2(1-p)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def skr_bb84_decoy(distance_km: float,
                   mu: float = MU,
                   mu_decoy: float = MU_DECOY,
                   eta_det: float = ETA_DET,
                   p_dark: float = P_DARK,
                   e_det: float = E_DETECTOR,
                   f_ec: float = F_EC) -> float:
    """
    Tasa de clave secreta BB84 con decoy states (cota inferior).

    Modelo simplificado de Lo-Ma-Chen (2005):
      R ≥ Q₁[1 − h(e₁)] − Q_μ·f_ec·h(e_μ)

    donde Q_μ ≈ T + p_dark es la tasa de detección total y e₁ es el QBER
    de los fotones de un solo fotón (estimado vía decoy).

    Devuelve bits/pulso. Retorna 0.0 si la distancia supera el rango máximo.
    """
    T = channel_transmittance(distance_km, eta_det)

    # Tasa de detección para pulso señal (intensidad mu)
    Q_mu = 1 - math.exp(-mu * T) + p_dark
    if Q_mu <= 0:
        return 0.0

    # QBER global
    e_mu = qber(distance_km, eta_det, p_dark, e_det)
    if e_mu >= 0.5:
        return 0.0

    # Estimación del término de fotón único mediante decoy
    T_decoy = channel_transmittance(distance_km, eta_det)
    Q1 = mu * T_decoy * math.exp(-mu) + p_dark  # Contribución 1-fotón

    # QBER del término de 1 fotón (límite inferior decoy)
    e1 = min(e_mu * Q_mu / max(Q1, 1e-15), 0.5)

    rate = Q1 * (1 - h2(e1)) - Q_mu * f_ec * h2(e_mu)
    return max(rate, 0.0)


def max_range_km(threshold: float = 1e-8, step: float = 1.0) -> float:
    """Distancia máxima donde SKR > threshold bits/pulso."""
    d = 0.0
    while d < 500:
        if skr_bb84_decoy(d) < threshold:
            return d
        d += step
    return 500.0


# ---------------------------------------------------------------------------
# Curva SKR(d)
# ---------------------------------------------------------------------------

def plot_skr_vs_distance(out_dir: str,
                         distances_km: np.ndarray = None) -> None:
    if distances_km is None:
        distances_km = np.linspace(0.1, 200, 400)

    skr_vals = np.array([skr_bb84_decoy(d) for d in distances_km])
    positive_mask = skr_vals > 0

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(distances_km[positive_mask], skr_vals[positive_mask],
                color='steelblue', lw=2.0, label='BB84 decoy (cota inferior)')

    # Referencia: distancias típicas de redes QKD
    ref_points = [(50, 'Tokyo QKD\n(50 km)'),
                  (100, 'SECOQC Vienna\n(~100 km)')]
    for d_ref, lbl in ref_points:
        s = skr_bb84_decoy(d_ref)
        if s > 0:
            ax.scatter([d_ref], [s], color='darkorange', zorder=5, s=60)
            ax.annotate(lbl, (d_ref, s), xytext=(d_ref + 3, s * 3),
                        fontsize=8, color='darkorange',
                        arrowprops=dict(arrowstyle='->', color='darkorange',
                                        lw=0.8))

    # Umbral operativo QKD: 10⁻⁵ bits/pulso
    ax.axhline(1e-5, color='gray', lw=0.8, ls='--', alpha=0.6,
               label=r'Umbral operativo ($10^{-5}$ bits/pulso)')

    ax.set_xlabel('Distancia (km)', fontsize=11)
    ax.set_ylabel('SKR (bits/pulso)', fontsize=11)
    ax.set_title('Tasa de clave secreta BB84 con decoy states\n'
                 rf'($\alpha={ALPHA_DB_KM}$ dB/km, $\eta={ETA_DET}$, '
                 rf'$\mu={MU}$, $e_{{det}}={E_DETECTOR}$)',
                 fontsize=10)
    ax.set_xlim(0, 200)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f'skr_vs_distancia.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Guardado: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# SKR por arista para los tres casos
# ---------------------------------------------------------------------------

def skr_for_graph_edges(G: nx.Graph, case_name: str) -> pd.DataFrame:
    """Calcula SKR para cada arista del grafo (requiere atributo dist_km o weight)."""
    rows = []
    for u, v, data in G.edges(data=True):
        dist = data.get('dist_km', data.get('weight', None))
        if dist is None:
            continue
        skr_val = skr_bb84_decoy(float(dist))
        rows.append({
            'caso': case_name,
            'nodo_u': str(u),
            'nodo_v': str(v),
            'dist_km': float(dist),
            'SKR_bits_pulso': skr_val,
            'viable_QKD': skr_val > 1e-8,
        })
    return pd.DataFrame(rows)


def compute_skr_from_adjacency(adj_csv: str, case_name: str,
                                coords_csv: str = None,
                                coords_sep: str = ';') -> pd.DataFrame:
    """
    Carga una matriz de adyacencia binaria y asigna distancias haversine
    desde el archivo de coordenadas si se proporciona.
    """
    adj = pd.read_csv(adj_csv, index_col=0)
    G = nx.from_pandas_adjacency(adj)

    if coords_csv and os.path.exists(coords_csv):
        coords_df = pd.read_csv(coords_csv, delimiter=coords_sep, decimal=',')
        coords_df.columns = [c.strip().lstrip('﻿') for c in coords_df.columns]
        col_pob = 'Población' if 'Población' in coords_df.columns else coords_df.columns[0]
        coords = {row[col_pob]: (float(row['Latitud']), float(row['Longitud']))
                  for _, row in coords_df.iterrows()
                  if row[col_pob] in G.nodes()}

        for u, v in G.edges():
            if u in coords and v in coords:
                lat1, lon1 = coords[u]
                lat2, lon2 = coords[v]
                dist = _haversine(lat1, lon1, lat2, lon2)
                G[u][v]['dist_km'] = dist

    return skr_for_graph_edges(G, case_name)


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Modelo SKR BB84 con decoy states")
    print("=" * 60)

    # Verificación de calibración
    print("\nCalibración (orden de magnitud esperado ~10⁻³ a 10⁻⁴ a 50 km):")
    for d in [10, 25, 50, 75, 100, 150]:
        s = skr_bb84_decoy(d)
        e = qber(d)
        print(f"  d={d:3d} km  SKR={s:.3e} bits/pulso  QBER={e:.4f}")

    r_max = max_range_km(threshold=1e-8)
    print(f"\n  Rango máximo (SKR > 1e-8): {r_max:.0f} km")

    # Figura SKR(d)
    print("\nGenerando figura SKR(d)...")
    plot_skr_vs_distance(FIGS_OUT)

    # SKR por arista
    all_dfs = []

    casos_adj = [
        ('CyL',    os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv'),
                   os.path.join(DATA_CYL, 'cyl_1000.csv'), ';'),
        ('España', os.path.join(DATA_ESP, 'AdjacencyMatrixNamed45.csv'),
                   os.path.join(DATA_ESP, 'peninsula_1000.csv'), ';'),
    ]

    for name, adj_csv, coords_csv, sep in casos_adj:
        if not os.path.exists(adj_csv):
            print(f"No encontrado: {adj_csv}")
            continue
        df_skr = compute_skr_from_adjacency(adj_csv, name, coords_csv, sep)
        n_viable = df_skr['viable_QKD'].sum()
        n_total  = len(df_skr)
        if n_total:
            print(f"\n{name}: {n_viable}/{n_total} aristas viables QKD "
                  f"(SKR>0)  SKR_min={df_skr['SKR_bits_pulso'].min():.2e}  "
                  f"SKR_max={df_skr['SKR_bits_pulso'].max():.2e}")
        all_dfs.append(df_skr)

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        out_csv = os.path.join(BASE, '..', 'datos', 'skr_per_link.csv')
        df_all.to_csv(out_csv, index=False)
        print(f"\nGuardado: {out_csv}")

    print("\nDone.")
