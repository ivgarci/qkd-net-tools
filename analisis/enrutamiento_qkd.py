"""
Análisis reproducible de enrutamiento consciente de QKD para CyL y España.

Para todos los pares no ordenados del caso elegido se comparan:

1. ``min_hops``: mínimo número de saltos y, entre esos caminos, máximo
   cuello de botella SKR.
2. ``max_min_skr``: máximo cuello de botella SKR y, entre esos caminos,
   mínimo número de saltos.

Si todavía existe un empate, se elige el camino lexicográficamente menor
según el nombre Unicode de los nodos. Esta última regla no cambia las métricas,
pero hace que el camino publicado sea independiente del orden de inserción de
NetworkX.

CyL es el caso predeterminado y conserva rutas completas para sus 4.950 pares.
España usa el mismo núcleo exacto en modo métrico para sus 450.775 pares, sin
almacenar centenares de miles de secuencias de nodos.

Salidas CyL:
  datos/enrutamiento_qkd_{allpairs,summary,bottleneck}.csv
  figuras/comparacion_rutas_qkd.pdf/.png

Salidas España:
  datos/resultados_papers/enrutamiento_espana_{allpairs,summary}.csv
  figuras/comparacion_rutas_qkd_espana.pdf/.png
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import platform
import sys
from typing import Hashable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_CYL = os.path.join(BASE, "..", "datos", "cyl")
DATA_ESP = os.path.join(BASE, "..", "datos", "espana")
DATA_OUT = os.path.join(BASE, "..", "datos")
PAPER_OUT = os.path.join(DATA_OUT, "resultados_papers")
FIGS_OUT = os.path.join(BASE, "..", "figuras")
os.makedirs(FIGS_OUT, exist_ok=True)

sys.path.insert(0, os.path.join(BASE, ".."))
from protocols.skr_bb84 import _haversine, skr_bb84_asymptotic  # noqa: E402
from analisis.routing_core import (  # noqa: E402
    compare_route_metrics,
    compare_routes,
    load_qkd_graph,
    max_bottleneck_min_hops_path,
    shortest_max_bottleneck_path,
)


def build_qkd_graph(
    adj_csv: str,
    coords_csv: str,
    coords_sep: str = ";",
    distance_factor: float = 1.0,
) -> nx.Graph:
    """Carga el grafo y asigna distancia de enlace y SKR a cada arista.

    No se imputan distancias. Una coordenada ausente invalidaría el análisis y
    por ello se informa como error. ``distance_factor=1`` conserva la distancia
    geodésica canónica; otros factores son escenarios explícitos.
    """
    return load_qkd_graph(
        adj_csv,
        coords_csv,
        skr_bb84_asymptotic,
        coordinates_sep=coords_sep,
        distance_factor=distance_factor,
        haversine=_haversine,
    )


def max_skr_path(
    G: nx.Graph,
    source,
    target,
):
    """Camino max-min SKR; entre empates, menor número de saltos."""
    return max_bottleneck_min_hops_path(G, source, target)


def compare_routing(
    G: nx.Graph,
    pairs: list[tuple[Hashable, Hashable]] | None = None,
    *,
    metrics_only: bool = False,
) -> pd.DataFrame:
    """Compara ambos criterios para todos los pares no ordenados.

    El modo métrico evita reconstruir 450.775 caminos completos en España,
    pero conserva exactamente los dos criterios y sus desempates cuantitativos.
    """
    runner = compare_route_metrics if metrics_only else compare_routes
    return pd.DataFrame(runner(G, pairs))


def summarize(df: pd.DataFrame, G: nx.Graph) -> pd.DataFrame:
    """Resumen de magnitudes verificables usado por texto y figura."""
    improved = df["mqr_skr_bottleneck"] > df["sp_skr_bottleneck"]
    same = np.isclose(
        df["mqr_skr_bottleneck"],
        df["sp_skr_bottleneck"],
        rtol=1e-12,
        atol=0.0,
    )
    return pd.DataFrame([{
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "unordered_pairs": len(df),
        "mean_skr_gain": df["skr_gain"].mean(),
        "median_skr_gain": df["skr_gain"].median(),
        "mean_hop_overhead": df["hop_overhead"].mean(),
        "median_hop_overhead": df["hop_overhead"].median(),
        "pairs_skr_improved": int(improved.sum()),
        "pairs_same_skr": int(same.sum()),
        "max_skr_gain": df["skr_gain"].max(),
        "max_hop_overhead": int(df["hop_overhead"].max()),
    }])


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_results(df: pd.DataFrame, G: nx.Graph) -> None:
    expected_pairs = G.number_of_nodes() * (G.number_of_nodes() - 1) // 2
    if len(df) != expected_pairs:
        raise AssertionError(f"Se esperaban {expected_pairs} pares y hay {len(df)}")
    if df[["origen", "destino"]].duplicated().any():
        raise AssertionError("Hay pares duplicados")
    if (df["mqr_skr_bottleneck"] + 1e-15 < df["sp_skr_bottleneck"]).any():
        raise AssertionError("Una ruta max-min tiene menor cuello de botella")
    if (df["hop_overhead"] < 0).any():
        raise AssertionError("Una ruta max-min usa menos saltos que una ruta mínima")


def plot_routing_comparison(
    df: pd.DataFrame,
    label: str,
    out_dir: str,
    *,
    output_stem: str = "comparacion_rutas_qkd",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    valid = df[
        (df["sp_skr_bottleneck"] > 0)
        & (df["mqr_skr_bottleneck"] > 0)
    ]
    scatter = valid
    if len(scatter) > 5_000:
        scatter = scatter.sample(n=5_000, random_state=42)
    ax.scatter(
        scatter["sp_skr_bottleneck"],
        scatter["mqr_skr_bottleneck"],
        alpha=0.35,
        s=14,
        color="steelblue",
        rasterized=len(valid) > 5_000,
    )
    lim_min = min(
        valid["sp_skr_bottleneck"].min(),
        valid["mqr_skr_bottleneck"].min(),
    ) * 0.8
    lim_max = max(
        valid["sp_skr_bottleneck"].max(),
        valid["mqr_skr_bottleneck"].max(),
    ) * 1.2
    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        "k--",
        lw=0.8,
        alpha=0.5,
        label="Igualdad",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cuello de botella: mínimos saltos (bits/pulso)")
    ax.set_ylabel("Cuello de botella: max-min (bits/pulso)")
    ax.set_title("(a) Comparación por par")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax2 = axes[1]
    gains = df["skr_gain"].replace([np.inf, -np.inf], np.nan).dropna()
    ax2.hist(gains, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
    ax2.axvline(
        1.0,
        color="black",
        lw=0.8,
        ls="--",
        label="Sin mejora",
    )
    mean_gain = gains.mean()
    ax2.axvline(
        mean_gain,
        color="darkorange",
        lw=1.0,
        label=f"Media = {mean_gain:.2f}",
    )
    ax2.set_xlabel("Razón SKR max-min / mínimos saltos")
    ax2.set_ylabel("Número de pares")
    ax2.set_title("(b) Distribución de la mejora")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Enrutamiento consciente del canal — {label}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{output_stem}.{ext}")
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else {}
        fig.savefig(path, dpi=150, bbox_inches="tight", metadata=metadata)
        print(f"Guardado: {path}")
    plt.close(fig)


def run_case(case: str, distance_factor: float = 1.0) -> pd.DataFrame:
    """Ejecuta un caso con rutas completas (CyL) o métricas exactas (España)."""
    if case == "cyl":
        if not math.isclose(distance_factor, 1.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(
                "CyL solo admite el escenario canónico distance_factor=1.0; "
                "no se sobrescribirán sus artefactos con una escala alternativa"
            )
        adj_csv = os.path.join(DATA_CYL, "AdjacencyMatrixNamed45.csv")
        coords_csv = os.path.join(DATA_CYL, "cyl_1000.csv")
        network_label = "CyL"
        metrics_only = False
        allpairs_csv = os.path.join(DATA_OUT, "enrutamiento_qkd_allpairs.csv")
        summary_csv = os.path.join(DATA_OUT, "enrutamiento_qkd_summary.csv")
        bottleneck_csv = os.path.join(DATA_OUT, "enrutamiento_qkd_bottleneck.csv")
        figure_stem = "comparacion_rutas_qkd"
        figure_label = "CyL (4.950 pares no ordenados)"
    elif case == "espana":
        adj_csv = os.path.join(DATA_ESP, "AdjacencyMatrixNamed45.csv")
        coords_csv = os.path.join(DATA_ESP, "peninsula_1000.csv")
        network_label = "España"
        metrics_only = True
        os.makedirs(PAPER_OUT, exist_ok=True)
        allpairs_csv = os.path.join(
            PAPER_OUT, "enrutamiento_espana_allpairs.csv"
        )
        summary_csv = os.path.join(
            PAPER_OUT, "enrutamiento_espana_summary.csv"
        )
        bottleneck_csv = None
        figure_stem = "comparacion_rutas_qkd_espana"
        figure_label = "España (450.775 pares no ordenados)"
    else:
        raise ValueError(f"Caso no admitido: {case!r}")

    G = build_qkd_graph(
        adj_csv,
        coords_csv,
        distance_factor=distance_factor,
    )
    print(
        f"Grafo {network_label}: |V|={G.number_of_nodes()}, "
        f"|E|={G.number_of_edges()}, factor de distancia={distance_factor:g}"
    )

    df = compare_routing(G, metrics_only=metrics_only)
    validate_results(df, G)
    summary = summarize(df, G)
    if case == "espana":
        summary = summary.assign(
            case=case,
            distance_factor=distance_factor,
            distance_scenario=(
                "geodesic_canonical"
                if distance_factor == 1.0
                else "explicit_scaled_distance_scenario"
            ),
            routing_output_mode="metrics_only",
        )
    summary = summary.assign(
        adjacency_sha256=_sha256(adj_csv),
        coordinates_sha256=_sha256(coords_csv),
        skr_model_sha256=_sha256(
            os.path.join(BASE, "..", "protocols", "skr_bb84.py")
        ),
        routing_core_sha256=_sha256(
            os.path.join(BASE, "routing_core.py")
        ),
        python_version=platform.python_version(),
        networkx_version=nx.__version__,
        pandas_version=pd.__version__,
        numpy_version=np.__version__,
        matplotlib_version=matplotlib.__version__,
    )

    df.to_csv(allpairs_csv, index=False, float_format="%.17g")
    summary.to_csv(summary_csv, index=False, float_format="%.17g")
    if bottleneck_csv is not None:
        df.nsmallest(10, "sp_skr_bottleneck").to_csv(
            bottleneck_csv,
            index=False,
            float_format="%.17g",
        )

    row = summary.iloc[0]
    print(f"Pares analizados: {int(row['unordered_pairs'])}")
    print(f"Mejora media SKR (ratio): {row['mean_skr_gain']:.12f}x")
    print(f"Incremento medio de saltos: {row['mean_hop_overhead']:.12f}")
    print(f"Guardado: {allpairs_csv}")
    print(f"Guardado: {summary_csv}")
    if bottleneck_csv is not None:
        print(f"Guardado: {bottleneck_csv}")
    plot_routing_comparison(
        df,
        figure_label,
        FIGS_OUT,
        output_stem=figure_stem,
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrutamiento QKD reproducible para CyL o España."
    )
    parser.add_argument(
        "--case",
        choices=("cyl", "espana"),
        default="cyl",
        help="Caso de estudio (por defecto: cyl).",
    )
    parser.add_argument(
        "--distance-factor",
        type=float,
        default=1.0,
        help=(
            "Multiplicador explícito de la distancia geodésica. "
            "El escenario canónico usa 1.0; para España, 1.25 está disponible "
            "solo como escenario hipotético de longitud de fibra. CyL rechaza "
            "factores no canónicos para no sobrescribir sus artefactos."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_case(args.case, distance_factor=args.distance_factor)


if __name__ == "__main__":
    main()
