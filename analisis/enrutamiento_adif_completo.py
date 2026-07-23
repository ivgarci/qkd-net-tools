"""Escenario proxy de encaminamiento QKD sobre el grafo ferroviario ADIF.

Las longitudes ``dist_km`` proceden de las adyacencias ferroviarias publicadas.
No acreditan fibra, continuidad óptica ni disponibilidad de planta. En este
escenario se introducen como longitudes proxy en el modelo BB84 asintótico
ideal de ``protocols/skr_bb84.py``.

No se filtran aristas por distancia y no se sustituuyen tasas nulas por un
suelo artificial. Las políticas exactas son las de ``routing_core``:

* menos saltos; entre empates, mayor cuello de botella;
* máximo cuello de botella; entre empates, menos saltos.

El resumen determinista se guarda en:

``datos/resultados_papers/enrutamiento_adif_summary.json``.

El CSV de millones de pares solo se escribe con ``--write-pairs``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterator

import networkx as nx
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATA_ADIF = ROOT / "datos" / "adif"
OUT_DIR = ROOT / "datos" / "resultados_papers"
NODES_CSV = DATA_ADIF / "nodos_red_adif.csv"
EDGES_CSV = DATA_ADIF / "adyacencia_red_adif.csv"
SUMMARY_JSON = OUT_DIR / "enrutamiento_adif_summary.json"
PAIRS_CSV = OUT_DIR / "enrutamiento_adif_allpairs.csv"

import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))
from protocols.skr_bb84 import _haversine, skr_bb84_asymptotic  # noqa: E402
from routing_core import (  # noqa: E402
    max_min_metrics_from_source,
    min_hops_metrics_from_source,
    node_key,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError(f"Valor no finito en estadística: {value!r}")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def as_dict(self) -> dict[str, object]:
        if not self.count:
            return {
                "count": 0,
                "mean": None,
                "std_population": None,
                "min": None,
                "max": None,
            }
        return {
            "count": self.count,
            "mean": self.mean,
            "std_population": math.sqrt(self.m2 / self.count),
            "min": self.minimum,
            "max": self.maximum,
        }


def build_adif_proxy_graph(
    nodes_csv: str | Path = NODES_CSV,
    edges_csv: str | Path = EDGES_CSV,
) -> tuple[nx.Graph, dict[str, tuple[float, float]], dict[str, object]]:
    """Construye la LCC del proxy sin filtrar ni imputar longitudes."""
    nodes = pd.read_csv(
        nodes_csv,
        quotechar='"',
        dtype={"cod": str, "conectado": str},
    )
    edges = pd.read_csv(
        edges_csv,
        quotechar='"',
        dtype={"cod": str, "vecino_cod": str},
    )
    required_nodes = {"cod", "nombre", "lat", "lon", "conectado"}
    required_edges = {"cod", "vecino_cod", "dist_km"}
    if missing := required_nodes - set(nodes.columns):
        raise ValueError(f"Faltan columnas de nodos: {sorted(missing)}")
    if missing := required_edges - set(edges.columns):
        raise ValueError(f"Faltan columnas de aristas: {sorted(missing)}")

    graph = nx.Graph()
    connected = nodes[nodes["conectado"] == "SI"]
    for row in connected.itertuples():
        latitude, longitude = float(row.lat), float(row.lon)
        if not (math.isfinite(latitude) and math.isfinite(longitude)):
            raise ValueError(f"Coordenadas no finitas para {row.cod!r}")
        graph.add_node(
            str(row.cod),
            nombre=str(row.nombre),
            lat=latitude,
            lon=longitude,
        )

    unique_distances: dict[frozenset[str], float] = {}
    rows_outside_nodes = 0
    for row in edges.itertuples():
        first, second = str(row.cod), str(row.vecino_cod)
        if first not in graph or second not in graph:
            rows_outside_nodes += 1
            continue
        distance = float(row.dist_km)
        if not math.isfinite(distance) or distance < 0:
            raise ValueError(f"Distancia no válida para {first!r}--{second!r}")
        key = frozenset((first, second))
        previous = unique_distances.get(key)
        if previous is not None and not math.isclose(
            previous, distance, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"Distancias contradictorias para {first!r}--{second!r}: "
                f"{previous} y {distance}"
            )
        unique_distances[key] = distance

    for key, distance in unique_distances.items():
        if len(key) != 2:
            raise ValueError(f"Autobucle no admitido en ADIF: {tuple(key)}")
        first, second = sorted(key)
        graph.add_edge(
            first,
            second,
            dist_km=distance,
            SKR=skr_bb84_asymptotic(distance),
        )

    if not graph.number_of_edges():
        raise ValueError("El proxy ADIF no contiene aristas")
    components = sorted(
        nx.connected_components(graph),
        key=lambda component: (
            -len(component),
            [node_key(node) for node in sorted(component, key=node_key)],
        ),
    )
    largest = graph.subgraph(components[0]).copy()
    coordinates = {
        node: (largest.nodes[node]["lat"], largest.nodes[node]["lon"])
        for node in largest
    }
    rates = [
        float(attributes["SKR"])
        for _, _, attributes in largest.edges(data=True)
    ]
    distances = [
        float(attributes["dist_km"])
        for _, _, attributes in largest.edges(data=True)
    ]
    info = {
        "node_rows": len(nodes),
        "edge_rows": len(edges),
        "connected_node_rows": len(connected),
        "edge_rows_outside_connected_nodes": rows_outside_nodes,
        "unique_edges_before_lcc": len(unique_distances),
        "components": len(components),
        "component_sizes": [len(component) for component in components],
        "lcc_nodes": largest.number_of_nodes(),
        "lcc_edges": largest.number_of_edges(),
        "distance_min_km": min(distances),
        "distance_max_km": max(distances),
        "zero_skr_edges": sum(rate == 0.0 for rate in rates),
        "positive_skr_edges": sum(rate > 0.0 for rate in rates),
        "skr_min_positive": min((rate for rate in rates if rate > 0), default=None),
        "skr_max": max(rates),
    }
    return largest, coordinates, info


def geographic_bin(distance: float) -> str:
    if distance < 100:
        return "[0,100)"
    if distance < 300:
        return "[100,300)"
    if distance < 500:
        return "[300,500)"
    if distance < 700:
        return "[500,700)"
    return "[700,+inf)"


def iter_pair_metrics(
    graph: nx.Graph,
    coordinates: dict[str, tuple[float, float]],
) -> Iterator[dict[str, object]]:
    """Itera pares sin acumular millones de filas en memoria."""
    nodes = sorted(graph, key=node_key)
    for source_index, source in enumerate(nodes):
        shortest = min_hops_metrics_from_source(graph, source)
        widest = max_min_metrics_from_source(graph, source)
        lat1, lon1 = coordinates[source]
        for target in nodes[source_index + 1:]:
            first = shortest[target]
            second = widest[target]
            lat2, lon2 = coordinates[target]
            distance = float(_haversine(lat1, lon1, lat2, lon2))
            if first.bottleneck > 0:
                gain_status = "finite"
                gain = second.bottleneck / first.bottleneck
            elif second.bottleneck > 0:
                gain_status = "rescued_from_zero"
                gain = None
            else:
                gain_status = "both_zero"
                gain = None
            yield {
                "node_s": str(source),
                "node_t": str(target),
                "distance_geodesic_km": distance,
                "sp_hops": first.hops,
                "sp_skr_bottleneck": first.bottleneck,
                "mqr_hops": second.hops,
                "mqr_skr_bottleneck": second.bottleneck,
                "hop_overhead": second.hops - first.hops,
                "gain_status": gain_status,
                "skr_gain": gain,
            }


PAIR_FIELDS = [
    "node_s",
    "node_t",
    "distance_geodesic_km",
    "sp_hops",
    "sp_skr_bottleneck",
    "mqr_hops",
    "mqr_skr_bottleneck",
    "hop_overhead",
    "gain_status",
    "skr_gain",
]


def summarize_routing(
    graph: nx.Graph,
    coordinates: dict[str, tuple[float, float]],
    pairs_csv: str | Path | None = None,
) -> dict[str, object]:
    gains = RunningStats()
    overhead = RunningStats()
    geographic = {
        label: {"gain": RunningStats(), "overhead": RunningStats(), "pairs": 0}
        for label in (
            "[0,100)",
            "[100,300)",
            "[300,500)",
            "[500,700)",
            "[700,+inf)",
        )
    }
    statuses = {"finite": 0, "rescued_from_zero": 0, "both_zero": 0}
    writer = None
    stream = None
    try:
        if pairs_csv is not None:
            path = Path(pairs_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(stream, fieldnames=PAIR_FIELDS)
            writer.writeheader()

        for row in iter_pair_metrics(graph, coordinates):
            status = str(row["gain_status"])
            statuses[status] += 1
            hop_overhead = float(row["hop_overhead"])
            overhead.add(hop_overhead)
            label = geographic_bin(float(row["distance_geodesic_km"]))
            geographic[label]["pairs"] += 1
            geographic[label]["overhead"].add(hop_overhead)
            if row["skr_gain"] is not None:
                gain = float(row["skr_gain"])
                gains.add(gain)
                geographic[label]["gain"].add(gain)
            if writer is not None:
                serializable = dict(row)
                for key, value in serializable.items():
                    if isinstance(value, float):
                        serializable[key] = format(value, ".17g")
                    elif value is None:
                        serializable[key] = ""
                writer.writerow(serializable)
    finally:
        if stream is not None:
            stream.close()

    expected = graph.number_of_nodes() * (graph.number_of_nodes() - 1) // 2
    if sum(statuses.values()) != expected:
        raise AssertionError("El número de pares resumido no coincide con la LCC")
    return {
        "unordered_pairs": expected,
        "gain_status_counts": statuses,
        "finite_skr_gain": gains.as_dict(),
        "hop_overhead": overhead.as_dict(),
        "geographic_bins": {
            label: {
                "pairs": values["pairs"],
                "finite_skr_gain": values["gain"].as_dict(),
                "hop_overhead": values["overhead"].as_dict(),
            }
            for label, values in geographic.items()
        },
        "ratio_policy": (
            "skr_gain solo se calcula cuando el cuello de botella de la ruta "
            "min-hops es positivo; los casos ambos cero y rescatados desde cero "
            "se cuentan por separado."
        ),
    }


def run_adif_analysis(
    nodes_csv: str | Path = NODES_CSV,
    edges_csv: str | Path = EDGES_CSV,
    pairs_csv: str | Path | None = None,
) -> dict[str, object]:
    graph, coordinates, graph_info = build_adif_proxy_graph(
        nodes_csv, edges_csv
    )
    routing = summarize_routing(graph, coordinates, pairs_csv)
    return {
        "schema_version": 1,
        "analysis": "adif_railway_graph_qkd_proxy_routing",
        "warning": (
            "Las longitudes son ferroviarias y se usan como proxy de escenario; "
            "no acreditan fibra ni viabilidad óptica."
        ),
        "model": {
            "name": "ideal_asymptotic_bb84",
            "zero_skr_edges_preserved": True,
            "distance_filter_applied": False,
        },
        "routing_policies": {
            "min_hops": "min hops, then max bottleneck",
            "max_min": "max bottleneck, then min hops",
        },
        "inputs": {
            "nodes_csv": os.path.relpath(nodes_csv, ROOT),
            "nodes_sha256": sha256_file(nodes_csv),
            "edges_csv": os.path.relpath(edges_csv, ROOT),
            "edges_sha256": sha256_file(edges_csv),
            "skr_model_sha256": sha256_file(ROOT / "protocols" / "skr_bb84.py"),
            "routing_core_sha256": sha256_file(BASE / "routing_core.py"),
        },
        "graph": graph_info,
        "routing": routing,
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
    parser.add_argument("--summary", type=Path, default=SUMMARY_JSON)
    parser.add_argument(
        "--write-pairs",
        action="store_true",
        help=f"Escribe además el CSV grande: {PAIRS_CSV}",
    )
    parser.add_argument("--pairs-output", type=Path, default=PAIRS_CSV)
    args = parser.parse_args()

    pairs_output = args.pairs_output if args.write_pairs else None
    result = run_adif_analysis(pairs_csv=pairs_output)
    write_json_deterministic(result, args.summary)
    graph = result["graph"]
    routing = result["routing"]
    print(
        f"ADIF proxy LCC: |V|={graph['lcc_nodes']}, |E|={graph['lcc_edges']}, "
        f"SKR=0 en {graph['zero_skr_edges']} aristas"
    )
    print(
        f"Pares={routing['unordered_pairs']}, "
        f"gain finito medio={routing['finite_skr_gain']['mean']}, "
        f"Δh medio={routing['hop_overhead']['mean']}, "
        f"estados={routing['gain_status_counts']}"
    )
    print(f"Guardado: {args.summary}")
    if pairs_output is not None:
        print(f"Guardado: {pairs_output}")


if __name__ == "__main__":
    main()
