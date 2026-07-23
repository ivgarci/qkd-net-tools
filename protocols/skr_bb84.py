"""
Modelo asintótico ideal de tasa de clave secreta (SKR) para BB84.

Referencia principal:
  Lo, H.-K., Ma, X., & Chen, K. (2005). Decoy state quantum key distribution.
  Physical Review Letters, 94(23), 230504.

Alcance:
  Se supone que la componente de un fotón se estima exactamente (límite
  asintótico de estados señuelo). No se simulan intensidades señuelo finitas,
  fluctuaciones estadísticas, tamaño finito ni una planta óptica real. Los
  parámetros por defecto son ilustrativos, no una calibración experimental.

Genera:
  figuras/skr_vs_distancia.pdf/.png  — curva SKR(d) en escala log
  datos/skr_per_link.csv             — SKR por arista para CyL y España
"""

from __future__ import annotations

import os
import math

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
E_DETECTOR   = 0.015  # QBER intrínseca del detector (1,5 %)
F_EC         = 1.16   # Factor de eficiencia de corrección de errores (Shannon ≥ 1)
Q_SIFT       = 0.5    # Cribado de bases BB84 equiprobables


def channel_transmittance(distance_km: float, eta_det: float = ETA_DET) -> float:
    """η(d) = η_det · 10^(−α·d/10), incluida la detección."""
    if distance_km < 0:
        raise ValueError("distance_km debe ser no negativa")
    if not 0 <= eta_det <= 1:
        raise ValueError("eta_det debe pertenecer a [0, 1]")
    return eta_det * 10 ** (-ALPHA_DB_KM * distance_km / 10)


def qber(distance_km: float, eta_det: float = ETA_DET,
         p_dark: float = P_DARK, e_det: float = E_DETECTOR,
         mu: float = MU) -> float:
    """
    QBER E_μ del estado señal.

    Q_μ = 1 - (1-Y₀) exp(-μη)
    E_μ Q_μ = e₀Y₀ + e_det [1-exp(-μη)], con e₀=1/2.
    """
    _validate_protocol_parameters(mu, p_dark, e_det=e_det)
    eta = channel_transmittance(distance_km, eta_det)
    q_mu = signal_gain(eta, mu, p_dark)
    if q_mu <= 0:
        return 0.5
    errors = 0.5 * p_dark + e_det * (1 - math.exp(-mu * eta))
    return min(errors / q_mu, 0.5)


def signal_gain(eta: float, mu: float = MU,
                p_dark: float = P_DARK) -> float:
    """Ganancia total Q_μ del estado señal."""
    _validate_protocol_parameters(mu, p_dark)
    if not 0 <= eta <= 1:
        raise ValueError("eta debe pertenecer a [0, 1]")
    return 1 - (1 - p_dark) * math.exp(-mu * eta)


def single_photon_terms(eta: float, mu: float = MU,
                        p_dark: float = P_DARK,
                        e_det: float = E_DETECTOR) -> tuple[float, float, float]:
    """Devuelve (Y₁, Q₁, e₁) bajo estimación asintótica ideal."""
    _validate_protocol_parameters(mu, p_dark, e_det=e_det)
    if not 0 <= eta <= 1:
        raise ValueError("eta debe pertenecer a [0, 1]")
    y1 = 1 - (1 - p_dark) * (1 - eta)
    q1 = mu * math.exp(-mu) * y1
    e1 = (0.5 * p_dark + e_det * eta) / y1 if y1 > 0 else 0.5
    return y1, q1, min(e1, 0.5)


def _validate_protocol_parameters(mu: float, p_dark: float,
                                  e_det: float = E_DETECTOR,
                                  f_ec: float = F_EC,
                                  q: float = Q_SIFT) -> None:
    if mu <= 0:
        raise ValueError("mu debe ser positiva")
    if not 0 <= p_dark <= 1:
        raise ValueError("p_dark debe pertenecer a [0, 1]")
    if not 0 <= e_det <= 0.5:
        raise ValueError("e_det debe pertenecer a [0, 0.5]")
    if f_ec < 1:
        raise ValueError("f_ec debe ser al menos 1")
    if not 0 <= q <= 1:
        raise ValueError("q debe pertenecer a [0, 1]")


def h2(p: float) -> float:
    """Entropía binaria h(p) = -p·log2(p) - (1-p)·log2(1-p)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def skr_bb84_asymptotic(distance_km: float,
                        mu: float = MU,
                        eta_det: float = ETA_DET,
                        p_dark: float = P_DARK,
                        e_det: float = E_DETECTOR,
                        f_ec: float = F_EC,
                        q: float = Q_SIFT) -> float:
    """
    Cota asintótica ideal de SKR BB84 por pulso de estado señal emitido.

    Modelo de Lo--Ma--Chen (2005):
      R ≥ q {Q₁[1 − h₂(e₁)] − Q_μ f_ec h₂(E_μ)}

    Q₁ y e₁ se tratan como conocidos exactamente. Esto corresponde al límite
    asintótico de una estimación ideal mediante estados señuelo; no implementa
    un protocolo de intensidades señuelo ni sus cotas estadísticas.

    Devuelve bits por pulso de estado señal. Para obtener bits por pulso total
    habría que multiplicar por la probabilidad de elegir el estado señal.
    """
    _validate_protocol_parameters(mu, p_dark, e_det, f_ec, q)
    eta = channel_transmittance(distance_km, eta_det)
    q_mu = signal_gain(eta, mu, p_dark)
    e_mu = qber(distance_km, eta_det, p_dark, e_det, mu)
    _, q1, e1 = single_photon_terms(eta, mu, p_dark, e_det)
    rate = q * (q1 * (1 - h2(e1)) - q_mu * f_ec * h2(e_mu))
    return max(rate, 0.0)


def skr_bb84_decoy(distance_km: float, **kwargs) -> float:
    """
    Alias compatible para :func:`skr_bb84_asymptotic`.

    Se conserva el nombre histórico usado por otros scripts del repositorio,
    pero el modelo es una estimación asintótica ideal: no recibe ``mu_decoy``
    porque ninguna intensidad señuelo finita participa en el cálculo.
    """
    if "mu_decoy" in kwargs:
        raise TypeError(
            "mu_decoy no forma parte del modelo asintótico ideal; "
            "no se simulan intensidades señuelo finitas"
        )
    return skr_bb84_asymptotic(distance_km, **kwargs)


def max_range_km(threshold: float = 1e-8, step: float = 1.0) -> float:
    """Primer punto de la malla donde SKR cae por debajo de ``threshold``."""
    if threshold < 0:
        raise ValueError("threshold debe ser no negativo")
    if step <= 0:
        raise ValueError("step debe ser positivo")
    d = 0.0
    while d < 500:
        if skr_bb84_asymptotic(d) < threshold:
            return d
        d += step
    return 500.0


# ---------------------------------------------------------------------------
# Curva SKR(d)
# ---------------------------------------------------------------------------

def plot_skr_vs_distance(out_dir: str,
                         distances_km=None) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if distances_km is None:
        distances_km = np.linspace(0.1, 200, 600)

    skr_vals = np.array([skr_bb84_asymptotic(float(d)) for d in distances_km])
    positive_mask = skr_vals > 0

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(distances_km[positive_mask], skr_vals[positive_mask],
                color='steelblue', lw=2.0,
                label='BB84 asintótico (estimación ideal decoy)')

    ax.axvline(45, color='darkorange', lw=1.0, ls='--',
               label=r'Hipótesis de diseño $\Delta=45$ km')
    ax.axhline(1e-8, color='gray', lw=0.8, ls='--', alpha=0.7,
               label=r'Corte numérico de referencia ($10^{-8}$ bits/pulso)')

    ax.set_xlabel('Distancia (km)', fontsize=11)
    ax.set_ylabel('SKR (bits/pulso)', fontsize=11)
    ax.set_title('Tasa de clave secreta BB84: modelo asintótico ideal\n'
                 rf'($\alpha={ALPHA_DB_KM}$ dB/km, $\eta={ETA_DET}$, '
                 rf'$\mu={MU}$, $e_{{det}}={E_DETECTOR}$, $q=1/2$)',
                 fontsize=10)
    ax.set_xlim(0, 200)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f'skr_vs_distancia.{ext}')
        metadata = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else None
        fig.savefig(path, dpi=150, bbox_inches='tight', metadata=metadata)
        print(f"Guardado: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# SKR por arista para los tres casos
# ---------------------------------------------------------------------------

def skr_for_graph_edges(G, case_name: str):
    """Calcula SKR para cada arista del grafo (requiere atributo dist_km o weight)."""
    import pandas as pd

    rows = []
    for u, v, data in G.edges(data=True):
        dist = data.get('dist_km', data.get('weight', None))
        if dist is None:
            continue
        skr_val = skr_bb84_asymptotic(float(dist))
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
                               coords_sep: str = ';'):
    """
    Carga una matriz de adyacencia binaria y asigna distancias haversine
    desde el archivo de coordenadas si se proporciona.
    """
    import pandas as pd
    import networkx as nx

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
    import pandas as pd

    print("=" * 60)
    print("Modelo SKR BB84 asintótico (estimación ideal decoy)")
    print("=" * 60)

    # Verificación de calibración
    print("\nPuntos de comprobación del modelo:")
    for d in [10, 25, 50, 75, 100, 150]:
        s = skr_bb84_asymptotic(d)
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
            print(f"\n{name}: {n_viable}/{n_total} aristas por encima del corte "
                  f"numérico (SKR>1e-8)  "
                  f"SKR_min={df_skr['SKR_bits_pulso'].min():.2e}  "
                  f"SKR_max={df_skr['SKR_bits_pulso'].max():.2e}")
        all_dfs.append(df_skr)

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        out_csv = os.path.join(BASE, '..', 'datos', 'skr_per_link.csv')
        df_all.to_csv(out_csv, index=False)
        print(f"\nGuardado: {out_csv}")

    print("\nDone.")
