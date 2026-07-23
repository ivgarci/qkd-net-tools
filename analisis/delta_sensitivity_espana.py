"""Escenario alternativo de sensibilidad al umbral de enlace en España.

Este análisis NO modifica ni reproduce el snapshot base de encaminamiento.
Reconstruye grafos geométricos alternativos sobre los mismos 950 nodos PAM:
existe una arista cuando la distancia haversine es menor o igual que ``delta``.
La longitud evaluada por el modelo SKR es ``rho * haversine``; ``rho`` es un
supuesto explícito y vale 1 por defecto.

Las dos políticas de encaminamiento y sus desempates proceden de
``analisis/routing_core.py``. La salida es propia y determinista:

``datos/resultados_papers/delta_sensitivity_espana.json``

Nunca se escribe ``tablas_skr_routing.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATA_ESP = ROOT / "datos" / "espana"
OUT_DIR = ROOT / "datos" / "resultados_papers"
COORDS_CSV = DATA_ESP / "peninsula_1000.csv"
ADJ_MAT_CSV = DATA_ESP / "AdjacencyMatrixNamed45.csv"
JSON_OUT = OUT_DIR / "delta_sensitivity_espana.json"

import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))
from protocols.skr_bb84 import _haversine, skr_bb84_asymptotic  # noqa: E402
from routing_core import compare_route_metrics, node_key  # noqa: E402


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_edge(first: object, second: object) -> tuple[str, str]:
    left, right = str(first), str(second)
    return (left, right) if left <= right else (right, left)


def edge_set(graph: nx.Graph) -> set[tuple[str, str]]:
    return {canonical_edge(first, second) for first, second in graph.edges}


def edge_set_sha256(edges: set[tuple[str, str]]) -> str:
    payload = "".join(f"{first}\t{second}\n" for first, second in sorted(edges))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_snapshot(
    adjacency_csv: str | Path = ADJ_MAT_CSV,
) -> tuple[nx.Graph, list[str]]:
    adjacency = pd.read_csv(adjacency_csv, index_col=0)
    if list(adjacency.index) != list(adjacency.columns):
        raise ValueError("El snapshot no tiene iguales filas y columnas")
    values = adjacency.to_numpy()
    if not np.array_equal(values, values.T):
        raise ValueError("El snapshot no es simétrico")
    graph = nx.from_pandas_adjacency(adjacency)
    return graph, list(adjacency.index)


def load_coordinates(
    coordinates_csv: str | Path = COORDS_CSV,
    nodes: list[str] | None = None,
) -> dict[str, tuple[float, float]]:
    frame = pd.read_csv(coordinates_csv, delimiter=";")
    frame.columns = [str(column).strip().lstrip("\ufeff") for column in frame]
    required = {"Población", "Latitud", "Longitud"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas: {sorted(missing_columns)}")

    coordinates = {
        str(row["Población"]): (
            float(str(row["Latitud"]).replace(",", ".")),
            float(str(row["Longitud"]).replace(",", ".")),
        )
        for _, row in frame.iterrows()
    }
    if nodes is None:
        return coordinates
    missing_nodes = sorted(set(nodes) - set(coordinates), key=node_key)
    if missing_nodes:
        raise ValueError(
            f"Faltan coordenadas para {len(missing_nodes)} nodos: "
            f"{missing_nodes[:5]}"
        )
    return {node: coordinates[node] for node in nodes}


def build_alternative_graph(
    coordinates: dict[str, tuple[float, float]],
    delta_km: float,
    rho: float = 1.0,
) -> nx.Graph:
    """Reconstruye un grafo geométrico; no reutiliza aristas del snapshot."""
    if not math.isfinite(delta_km) or delta_km <= 0:
        raise ValueError("delta_km debe ser finito y positivo")
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError("rho debe ser finito y positivo")

    graph = nx.Graph()
    nodes = sorted(coordinates, key=node_key)
    graph.add_nodes_from(nodes)
    for first, second in itertools.combinations(nodes, 2):
        lat1, lon1 = coordinates[first]
        lat2, lon2 = coordinates[second]
        haversine_km = float(_haversine(lat1, lon1, lat2, lon2))
        if haversine_km <= delta_km:
            model_distance = rho * haversine_km
            graph.add_edge(
                first,
                second,
                haversine_km=haversine_km,
                dist_km=model_distance,
                SKR=skr_bb84_asymptotic(model_distance),
            )
    return graph


def select_largest_component(graph: nx.Graph) -> tuple[nx.Graph, dict[str, object]]:
    isolated = sorted(nx.isolates(graph), key=node_key)
    active = graph.copy()
    active.remove_nodes_from(isolated)
    components = sorted(
        nx.connected_components(active),
        key=lambda component: (
            -len(component),
            [node_key(n) for n in sorted(component, key=node_key)],
        ),
    )
    if not components:
        raise ValueError("El escenario no contiene ninguna arista")
    largest = active.subgraph(components[0]).copy()
    return largest, {
        "isolated_nodes": len(isolated),
        "components_after_isolates": len(components),
        "component_sizes": sorted((len(c) for c in components), reverse=True),
        "used_largest_component": len(components) > 1,
    }


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    gains = np.asarray([float(row["skr_gain"]) for row in rows], dtype=float)
    overhead = np.asarray(
        [int(row["hop_overhead"]) for row in rows], dtype=float
    )
    finite = np.isfinite(gains)
    return {
        "unordered_pairs": len(rows),
        "finite_gain_pairs": int(finite.sum()),
        "non_finite_gain_pairs": int((~finite).sum()),
        "mean_skr_gain": float(gains[finite].mean()) if finite.any() else None,
        "median_skr_gain": (
            float(np.median(gains[finite])) if finite.any() else None
        ),
        "max_skr_gain": float(gains[finite].max()) if finite.any() else None,
        "mean_hop_overhead": float(overhead.mean()),
        "median_hop_overhead": float(np.median(overhead)),
        "max_hop_overhead": int(overhead.max()),
    }


def edge_diff_record(
    rebuilt: nx.Graph,
    snapshot_edges: set[tuple[str, str]],
) -> dict[str, object]:
    rebuilt_edges = edge_set(rebuilt)
    added = rebuilt_edges - snapshot_edges
    removed = snapshot_edges - rebuilt_edges
    return {
        "snapshot_edges": len(snapshot_edges),
        "rebuilt_edges": len(rebuilt_edges),
        "common_edges": len(rebuilt_edges & snapshot_edges),
        "added_edges": len(added),
        "removed_edges": len(removed),
        "snapshot_edge_set_sha256": edge_set_sha256(snapshot_edges),
        "rebuilt_edge_set_sha256": edge_set_sha256(rebuilt_edges),
        "added_edge_set_sha256": edge_set_sha256(added),
        "removed_edge_set_sha256": edge_set_sha256(removed),
        "added_sample": [list(edge) for edge in sorted(added)[:10]],
        "removed_sample": [list(edge) for edge in sorted(removed)[:10]],
    }


def run_sensitivity(
    deltas: tuple[float, ...] = (35.0, 40.0, 45.0, 50.0),
    rho: float = 1.0,
    adjacency_csv: str | Path = ADJ_MAT_CSV,
    coordinates_csv: str | Path = COORDS_CSV,
) -> dict[str, object]:
    snapshot, nodes = load_snapshot(adjacency_csv)
    coordinates = load_coordinates(coordinates_csv, nodes)
    snapshot_edges = edge_set(snapshot)
    scenarios = []

    for delta in deltas:
        rebuilt = build_alternative_graph(coordinates, delta, rho)
        used, component_info = select_largest_component(rebuilt)
        rows = compare_route_metrics(used)
        scenarios.append({
            "delta_km": float(delta),
            "rho": float(rho),
            "model_distance": "rho * haversine_km",
            "full_nodes": rebuilt.number_of_nodes(),
            "full_edges": rebuilt.number_of_edges(),
            "active_nodes": used.number_of_nodes(),
            "active_edges": used.number_of_edges(),
            "connected_full_graph": nx.is_connected(rebuilt),
            "component_selection": component_info,
            "edge_diff_vs_snapshot": edge_diff_record(rebuilt, snapshot_edges),
            "routing": aggregate(rows),
        })

    return {
        "schema_version": 1,
        "analysis": "alternative_geometric_delta_sensitivity_spain",
        "warning": (
            "Cada escenario reconstruye un grafo alternativo desde coordenadas; "
            "no sustituye ni modifica el snapshot base."
        ),
        "routing_policies": {
            "min_hops": "min hops, then max bottleneck",
            "max_min": "max bottleneck, then min hops",
        },
        "inputs": {
            "adjacency_csv": os.path.relpath(adjacency_csv, ROOT),
            "adjacency_sha256": sha256_file(adjacency_csv),
            "coordinates_csv": os.path.relpath(coordinates_csv, ROOT),
            "coordinates_sha256": sha256_file(coordinates_csv),
            "skr_model_sha256": sha256_file(ROOT / "protocols" / "skr_bb84.py"),
            "routing_core_sha256": sha256_file(BASE / "routing_core.py"),
        },
        "scenarios": scenarios,
    }


def write_json_deterministic(payload: dict[str, object], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delta",
        nargs="+",
        type=float,
        default=[35.0, 40.0, 45.0, 50.0],
        help="Umbrales geométricos en km",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=1.0,
        help="Factor explícito entre haversine y distancia del modelo",
    )
    parser.add_argument("--output", type=Path, default=JSON_OUT)
    args = parser.parse_args()

    result = run_sensitivity(tuple(args.delta), args.rho)
    write_json_deterministic(result, args.output)
    for scenario in result["scenarios"]:
        routing = scenario["routing"]
        diff = scenario["edge_diff_vs_snapshot"]
        print(
            f"Δ={scenario['delta_km']:g} km, rho={scenario['rho']:g}: "
            f"|V|={scenario['active_nodes']}, |E|={scenario['active_edges']}, "
            f"pares={routing['unordered_pairs']}, "
            f"gain={routing['mean_skr_gain']:.6f}, "
            f"Δh={routing['mean_hop_overhead']:.6f}, "
            f"aristas +{diff['added_edges']}/-{diff['removed_edges']}"
        )
    print(f"Guardado: {args.output}")


if __name__ == "__main__":
    main()
