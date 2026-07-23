"""
Análisis reproducible de enrutamiento consciente de QKD para Castilla y León.

Para cada uno de los 4.950 pares no ordenados de los 100 nodos se comparan:

1. ``min_hops``: mínimo número de saltos y, entre esos caminos, máximo
   cuello de botella SKR.
2. ``max_min_skr``: máximo cuello de botella SKR y, entre esos caminos,
   mínimo número de saltos.

Si todavía existe un empate, se elige el camino lexicográficamente menor
según el nombre Unicode de los nodos. Esta última regla no cambia las métricas,
pero hace que el camino publicado sea independiente del orden de inserción de
NetworkX.

Salidas:
  datos/enrutamiento_qkd_allpairs.csv
  datos/enrutamiento_qkd_summary.csv
  datos/enrutamiento_qkd_bottleneck.csv
  figuras/comparacion_rutas_qkd.pdf/.png
"""

from __future__ import annotations

import heapq
import hashlib
import itertools
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
DATA_OUT = os.path.join(BASE, "..", "datos")
FIGS_OUT = os.path.join(BASE, "..", "figuras")
os.makedirs(FIGS_OUT, exist_ok=True)

sys.path.insert(0, os.path.join(BASE, ".."))
from protocols.skr_bb84 import _haversine, skr_bb84_asymptotic  # noqa: E402

def _node_key(node: Hashable) -> tuple[str, str]:
    """Clave total y estable incluso si un grafo mezcla tipos de nodo."""
    return type(node).__name__, str(node)


def _path_distance(G: nx.Graph, path: list[Hashable]) -> float:
    return sum(
        float(G[u][v]["dist_km"])
        for u, v in itertools.pairwise(path)
    )


def _path_bottleneck(G: nx.Graph, path: list[Hashable]) -> float:
    if len(path) < 2:
        return math.inf
    return min(
        float(G[u][v]["SKR"])
        for u, v in itertools.pairwise(path)
    )


def build_qkd_graph(
    adj_csv: str,
    coords_csv: str,
    coords_sep: str = ";",
) -> nx.Graph:
    """Carga el grafo y asigna distancia geodésica y SKR a cada arista.

    No se imputan distancias. Una coordenada ausente invalidaría el análisis y
    por ello se informa como error.
    """
    adj = pd.read_csv(adj_csv, index_col=0)
    if list(adj.index) != list(adj.columns):
        raise ValueError("La matriz de adyacencia no tiene iguales filas y columnas")
    if not np.array_equal(adj.to_numpy(), adj.to_numpy().T):
        raise ValueError("La matriz de adyacencia no es simétrica")

    G = nx.from_pandas_adjacency(adj)
    coords_df = pd.read_csv(coords_csv, delimiter=coords_sep)
    coords_df.columns = [str(c).strip().lstrip("\ufeff") for c in coords_df.columns]
    required = {"Población", "Latitud", "Longitud"}
    if not required.issubset(coords_df.columns):
        raise ValueError(
            f"Faltan columnas en {coords_csv}: {sorted(required - set(coords_df.columns))}"
        )

    coords = {
        row["Población"]: (float(row["Latitud"]), float(row["Longitud"]))
        for _, row in coords_df.iterrows()
    }
    missing = sorted(set(G.nodes) - set(coords), key=_node_key)
    if missing:
        raise ValueError(f"Faltan coordenadas para {len(missing)} nodos: {missing[:5]}")

    for u, v in G.edges:
        lat1, lon1 = coords[u]
        lat2, lon2 = coords[v]
        dist = _haversine(lat1, lon1, lat2, lon2)
        G[u][v]["dist_km"] = dist
        G[u][v]["SKR"] = skr_bb84_asymptotic(dist)

    if not nx.is_connected(G):
        raise ValueError("El grafo de CyL no es conexo; no existen los 4.950 pares")
    return G


def _shortest_preferred_path(
    G: nx.Graph,
    source: Hashable,
    target: Hashable,
) -> tuple[float, list[Hashable]]:
    """Dijkstra por (saltos, -cuello de botella, camino lexicográfico)."""
    if source not in G or target not in G:
        raise nx.NodeNotFound(f"Nodo ausente: {source!r} o {target!r}")
    if source == target:
        return math.inf, [source]

    source_path_key = (_node_key(source),)
    # heap: (prioridad primaria, secundaria, camino canónico, nodo, camino)
    heap = [(0.0, -math.inf, source_path_key, source, [source])]
    best: dict[Hashable, tuple[float, float, tuple[tuple[str, str], ...]]] = {
        source: (0.0, -math.inf, source_path_key)
    }

    while heap:
        first, second, path_key, node, path = heapq.heappop(heap)
        label = (first, second, path_key)
        if label != best.get(node):
            continue
        if node == target:
            bottleneck = _path_bottleneck(G, path)
            return bottleneck, path

        current_hops = len(path) - 1
        current_bottleneck = _path_bottleneck(G, path)
        for nbr in sorted(G.neighbors(node), key=_node_key):
            edge_skr = float(G[node][nbr].get("SKR", 0.0))
            candidate_bottleneck = min(current_bottleneck, edge_skr)
            candidate_hops = current_hops + 1
            candidate_path_key = path_key + (_node_key(nbr),)
            candidate = (
                float(candidate_hops),
                -candidate_bottleneck,
                candidate_path_key,
            )
            if candidate < best.get(nbr, (math.inf, math.inf, ())):
                best[nbr] = candidate
                heapq.heappush(
                    heap,
                    (*candidate, nbr, path + [nbr]),
                )

    return 0.0, []


def shortest_max_bottleneck_path(
    G: nx.Graph,
    source: Hashable,
    target: Hashable,
) -> tuple[float, list[Hashable]]:
    """Camino de saltos mínimos; entre empates, mayor SKR mínima."""
    return _shortest_preferred_path(G, source, target)


def max_skr_path(
    G: nx.Graph,
    source: Hashable,
    target: Hashable,
) -> tuple[float, list[Hashable]]:
    """Camino max-min SKR; entre empates, menor número de saltos."""
    if source not in G or target not in G:
        raise nx.NodeNotFound(f"Nodo ausente: {source!r} o {target!r}")
    if source == target:
        return math.inf, [source]

    # Fase 1: capacidad widest-path. Aquí una capacidad mayor sí domina a una
    # menor para cualquier continuación, por lo que basta una etiqueta por nodo.
    best = {source: math.inf}
    heap = [(-math.inf, _node_key(source), source)]
    while heap:
        neg_capacity, _, node = heapq.heappop(heap)
        capacity = -neg_capacity
        if capacity != best.get(node):
            continue
        for nbr in sorted(G.neighbors(node), key=_node_key):
            candidate = min(capacity, float(G[node][nbr].get("SKR", 0.0)))
            if candidate > best.get(nbr, -math.inf):
                best[nbr] = candidate
                heapq.heappush(heap, (-candidate, _node_key(nbr), nbr))

    max_bottleneck = best.get(target, 0.0)
    if max_bottleneck <= 0:
        return 0.0, []

    # Fase 2: el camino con menos saltos dentro del subgrafo que conserva
    # exactamente la capacidad max-min. _shortest_preferred_path añade el desempate
    # lexicográfico final.
    eligible = nx.subgraph_view(
        G,
        filter_edge=lambda u, v: float(G[u][v].get("SKR", 0.0))
        >= max_bottleneck,
    )
    bottleneck, path = _shortest_preferred_path(eligible, source, target)
    if not path:
        raise AssertionError("La segunda fase no recuperó el camino max-min")
    return bottleneck, path


def compare_routing(
    G: nx.Graph,
    pairs: list[tuple[Hashable, Hashable]] | None = None,
) -> pd.DataFrame:
    """Compara ambos criterios para todos los pares no ordenados."""
    nodes = sorted(G.nodes, key=_node_key)
    if pairs is None:
        pairs = list(itertools.combinations(nodes, 2))

    rows = []
    for u, v in pairs:
        sp_skr, sp_path = shortest_max_bottleneck_path(G, u, v)
        mqr_skr, mqr_path = max_skr_path(G, u, v)
        if not sp_path or not mqr_path:
            raise nx.NetworkXNoPath(f"No hay camino entre {u!r} y {v!r}")

        sp_hops = len(sp_path) - 1
        mqr_hops = len(mqr_path) - 1
        rows.append({
            "origen": str(u),
            "destino": str(v),
            "sp_hops": sp_hops,
            "sp_dist_km": _path_distance(G, sp_path),
            "sp_skr_bottleneck": sp_skr,
            "sp_path": " -> ".join(map(str, sp_path)),
            "mqr_hops": mqr_hops,
            "mqr_dist_km": _path_distance(G, mqr_path),
            "mqr_skr_bottleneck": mqr_skr,
            "mqr_path": " -> ".join(map(str, mqr_path)),
            "skr_gain": mqr_skr / sp_skr,
            "hop_overhead": mqr_hops - sp_hops,
        })
    return pd.DataFrame(rows)


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


def plot_routing_comparison(df: pd.DataFrame, label: str, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    valid = df[
        (df["sp_skr_bottleneck"] > 0)
        & (df["mqr_skr_bottleneck"] > 0)
    ]
    ax.scatter(
        valid["sp_skr_bottleneck"],
        valid["mqr_skr_bottleneck"],
        alpha=0.35,
        s=14,
        color="steelblue",
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
        path = os.path.join(out_dir, f"comparacion_rutas_qkd.{ext}")
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else {}
        fig.savefig(path, dpi=150, bbox_inches="tight", metadata=metadata)
        print(f"Guardado: {path}")
    plt.close(fig)


def main() -> None:
    adj_csv = os.path.join(DATA_CYL, "AdjacencyMatrixNamed45.csv")
    coords_csv = os.path.join(DATA_CYL, "cyl_1000.csv")
    G = build_qkd_graph(adj_csv, coords_csv)
    print(
        f"Grafo CyL: |V|={G.number_of_nodes()}, "
        f"|E|={G.number_of_edges()}"
    )

    df = compare_routing(G)
    validate_results(df, G)
    summary = summarize(df, G)
    summary = summary.assign(
        adjacency_sha256=_sha256(adj_csv),
        coordinates_sha256=_sha256(coords_csv),
        skr_model_sha256=_sha256(
            os.path.join(BASE, "..", "protocols", "skr_bb84.py")
        ),
        python_version=platform.python_version(),
        networkx_version=nx.__version__,
        pandas_version=pd.__version__,
        numpy_version=np.__version__,
        matplotlib_version=matplotlib.__version__,
    )

    allpairs_csv = os.path.join(DATA_OUT, "enrutamiento_qkd_allpairs.csv")
    summary_csv = os.path.join(DATA_OUT, "enrutamiento_qkd_summary.csv")
    bottleneck_csv = os.path.join(DATA_OUT, "enrutamiento_qkd_bottleneck.csv")
    df.to_csv(allpairs_csv, index=False, float_format="%.17g")
    summary.to_csv(summary_csv, index=False, float_format="%.17g")
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
    print(f"Guardado: {bottleneck_csv}")
    plot_routing_comparison(df, "CyL (4.950 pares no ordenados)", FIGS_OUT)


if __name__ == "__main__":
    main()
