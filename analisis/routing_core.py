"""Núcleo común y determinista para análisis de encaminamiento QKD.

Las dos políticas implementadas son, por este orden:

* ``min_hops``: menos saltos, mayor cuello de botella, camino lexicográfico.
* ``max_min``: mayor cuello de botella, menos saltos, camino lexicográfico.

El desempate lexicográfico usa ``(nombre del tipo, representación textual)`` y
evita que el resultado dependa del orden de las aristas en los CSV o NetworkX.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import heapq
import itertools
import math
from pathlib import Path
from typing import Callable, Hashable, Iterable

import networkx as nx
import numpy as np
import pandas as pd


Node = Hashable
SkrModel = Callable[[float], float]


@dataclass(frozen=True)
class Route:
    """Resultado inmutable de una política para un par de nodos."""

    bottleneck: float
    path: tuple[Node, ...]

    @property
    def hops(self) -> int:
        return max(0, len(self.path) - 1)


@dataclass(frozen=True)
class RouteMetrics:
    """Métricas de una ruta sin el coste de reconstruir su secuencia."""

    bottleneck: float
    hops: int


def node_key(node: Node) -> tuple[str, str]:
    """Clave estable incluso para grafos que mezclan tipos de nodo."""
    return type(node).__name__, str(node)


def path_distance(
    graph: nx.Graph,
    path: Iterable[Node],
    distance_attr: str = "dist_km",
) -> float:
    nodes = tuple(path)
    return sum(
        float(graph[u][v][distance_attr])
        for u, v in itertools.pairwise(nodes)
    )


def path_bottleneck(
    graph: nx.Graph,
    path: Iterable[Node],
    capacity_attr: str = "SKR",
) -> float:
    nodes = tuple(path)
    if len(nodes) < 2:
        return math.inf
    return min(
        float(graph[u][v][capacity_attr])
        for u, v in itertools.pairwise(nodes)
    )


def _as_float(value: object) -> float:
    """Acepta tanto decimales con punto como los CSV españoles con coma."""
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    return float(value)


def load_qkd_graph(
    adjacency_csv: str | Path,
    coordinates_csv: str | Path,
    skr_model: SkrModel,
    *,
    coordinates_sep: str = ";",
    node_column: str = "Población",
    latitude_column: str = "Latitud",
    longitude_column: str = "Longitud",
    distance_factor: float = 1.0,
    haversine: Callable[[float, float, float, float], float],
    require_connected: bool = True,
    distance_attr: str = "dist_km",
    capacity_attr: str = "SKR",
) -> nx.Graph:
    """Carga una topología y calcula distancia y SKR sin imputar datos.

    ``distance_factor`` permite representar longitud de fibra a partir de la
    distancia geodésica (por ejemplo, 1 para CyL o 1,25 para Península).
    Las coordenadas ausentes son un error deliberado: una imputación silenciosa
    alteraría las conclusiones científicas.
    """
    if not math.isfinite(distance_factor) or distance_factor <= 0:
        raise ValueError("distance_factor debe ser finito y mayor que cero")

    adjacency = pd.read_csv(adjacency_csv, index_col=0)
    if list(adjacency.index) != list(adjacency.columns):
        raise ValueError("La matriz de adyacencia no tiene iguales filas y columnas")
    values = adjacency.to_numpy()
    if not np.array_equal(values, values.T):
        raise ValueError("La matriz de adyacencia no es simétrica")

    graph = nx.from_pandas_adjacency(adjacency)
    coordinates = pd.read_csv(coordinates_csv, delimiter=coordinates_sep)
    coordinates.columns = [
        str(column).strip().lstrip("\ufeff") for column in coordinates.columns
    ]
    required = {node_column, latitude_column, longitude_column}
    missing_columns = required - set(coordinates.columns)
    if missing_columns:
        raise ValueError(
            f"Faltan columnas en {coordinates_csv}: {sorted(missing_columns)}"
        )

    coordinate_by_node = {
        row[node_column]: (
            _as_float(row[latitude_column]),
            _as_float(row[longitude_column]),
        )
        for _, row in coordinates.iterrows()
    }
    missing_nodes = sorted(set(graph) - set(coordinate_by_node), key=node_key)
    if missing_nodes:
        raise ValueError(
            f"Faltan coordenadas para {len(missing_nodes)} nodos: "
            f"{missing_nodes[:5]}"
        )

    for first, second in graph.edges:
        lat1, lon1 = coordinate_by_node[first]
        lat2, lon2 = coordinate_by_node[second]
        distance = float(haversine(lat1, lon1, lat2, lon2)) * distance_factor
        capacity = float(skr_model(distance))
        if not math.isfinite(distance) or distance < 0:
            raise ValueError(f"Distancia no válida para {first!r}-{second!r}")
        if not math.isfinite(capacity) or capacity < 0:
            raise ValueError(f"SKR no válida para {first!r}-{second!r}")
        graph[first][second][distance_attr] = distance
        graph[first][second][capacity_attr] = capacity

    if require_connected and graph and not nx.is_connected(graph):
        raise ValueError("El grafo no es conexo")
    return graph


@dataclass(frozen=True)
class _Label:
    capacity: float
    hops: int
    path_key: tuple[tuple[str, str], ...]
    path: tuple[Node, ...]


def _dominates(first: _Label, second: _Label) -> bool:
    """Dominancia segura también para el desempate lexicográfico futuro."""
    if first.capacity < second.capacity or first.hops > second.hops:
        return False
    if first.hops < second.hops:
        return True
    # Con los mismos saltos una capacidad mayor puede igualarse al añadir una
    # arista limitante. Solo domina si su prefijo tampoco es peor lexicalmente.
    return first.path_key <= second.path_key


def min_hops_routes_from_source(
    graph: nx.Graph,
    source: Node,
    *,
    capacity_attr: str = "SKR",
) -> dict[Node, Route]:
    """Obtiene todas las rutas min-hops desde ``source`` en un recorrido."""
    if source not in graph:
        raise nx.NodeNotFound(f"Nodo ausente: {source!r}")

    source_label = _Label(math.inf, 0, (node_key(source),), (source,))
    frontiers: dict[Node, list[_Label]] = defaultdict(list)
    frontiers[source].append(source_label)
    active = {id(source_label)}
    serial = itertools.count()
    heap = [
        (
            source_label.hops,
            -source_label.capacity,
            source_label.path_key,
            next(serial),
            source,
            source_label,
        )
    ]

    while heap:
        _, _, _, _, node, label = heapq.heappop(heap)
        if id(label) not in active:
            continue
        for neighbour in sorted(graph.neighbors(node), key=node_key):
            edge_capacity = float(graph[node][neighbour][capacity_attr])
            candidate = _Label(
                min(label.capacity, edge_capacity),
                label.hops + 1,
                label.path_key + (node_key(neighbour),),
                label.path + (neighbour,),
            )
            existing = frontiers[neighbour]
            if existing and candidate.hops > existing[0].hops:
                continue
            if existing and candidate.hops < existing[0].hops:
                for item in existing:
                    active.discard(id(item))
                existing.clear()
            if any(_dominates(item, candidate) for item in existing):
                continue
            dominated = [item for item in existing if _dominates(candidate, item)]
            for item in dominated:
                active.discard(id(item))
                existing.remove(item)
            existing.append(candidate)
            active.add(id(candidate))
            heapq.heappush(
                heap,
                (
                    candidate.hops,
                    -candidate.capacity,
                    candidate.path_key,
                    next(serial),
                    neighbour,
                    candidate,
                ),
            )

    result = {}
    for node, labels in frontiers.items():
        optimum = min(
            labels,
            key=lambda label: (
                label.hops,
                -label.capacity,
                label.path_key,
            ),
        )
        result[node] = Route(optimum.capacity, optimum.path)
    return result


def min_hops_metrics_from_source(
    graph: nx.Graph,
    source: Node,
    *,
    capacity_attr: str = "SKR",
) -> dict[Node, RouteMetrics]:
    """Métricas min-hops en tiempo lineal tras el BFS.

    Sobre el DAG inducido por las capas del BFS, una pasada calcula el máximo
    cuello de botella entre todos los caminos que tienen el mínimo de saltos.
    """
    if source not in graph:
        raise nx.NodeNotFound(f"Nodo ausente: {source!r}")
    distances = nx.single_source_shortest_path_length(graph, source)
    bottlenecks = {source: math.inf}
    for node in sorted(
        (item for item in distances if item != source),
        key=lambda item: (distances[item], node_key(item)),
    ):
        predecessors = (
            neighbour
            for neighbour in graph.neighbors(node)
            if distances.get(neighbour) == distances[node] - 1
        )
        bottlenecks[node] = max(
            min(
                bottlenecks[predecessor],
                float(graph[predecessor][node][capacity_attr]),
            )
            for predecessor in predecessors
        )
    return {
        node: RouteMetrics(bottlenecks[node], hops)
        for node, hops in distances.items()
    }


def max_min_metrics_from_source(
    graph: nx.Graph,
    source: Node,
    *,
    capacity_attr: str = "SKR",
) -> dict[Node, RouteMetrics]:
    """Métricas max-min exactas mediante BFS incremental por umbral.

    Las aristas se activan por grupos de capacidad descendente. Cuando un nodo
    se hace alcanzable por primera vez, ese umbral es su máximo cuello de
    botella; tras propagar todas las relajaciones del grupo, su distancia es el
    mínimo número de saltos dentro del subgrafo que alcanza dicho óptimo.
    """
    if source not in graph:
        raise nx.NodeNotFound(f"Nodo ausente: {source!r}")

    edges_by_capacity: dict[float, list[tuple[Node, Node]]] = defaultdict(list)
    for first, second, attributes in graph.edges(data=True):
        capacity = float(attributes[capacity_attr])
        if not math.isfinite(capacity) or capacity < 0:
            raise ValueError(f"Capacidad no válida para {first!r}-{second!r}")
        edges_by_capacity[capacity].append((first, second))

    active: dict[Node, set[Node]] = defaultdict(set)
    infinity = graph.number_of_nodes() + 1
    distances = {node: infinity for node in graph}
    distances[source] = 0
    result = {source: RouteMetrics(math.inf, 0)}

    for capacity in sorted(edges_by_capacity, reverse=True):
        queue: deque[Node] = deque()
        queued: set[Node] = set()
        newly_reached: set[Node] = set()

        def relax(node: Node, candidate: int) -> None:
            if candidate >= distances[node]:
                return
            if distances[node] == infinity:
                newly_reached.add(node)
            distances[node] = candidate
            if node not in queued:
                queued.add(node)
                queue.append(node)

        for first, second in edges_by_capacity[capacity]:
            active[first].add(second)
            active[second].add(first)
            if distances[first] < infinity:
                relax(second, distances[first] + 1)
            if distances[second] < infinity:
                relax(first, distances[second] + 1)

        while queue:
            node = queue.popleft()
            queued.remove(node)
            candidate = distances[node] + 1
            for neighbour in active[node]:
                relax(neighbour, candidate)

        for node in newly_reached:
            result[node] = RouteMetrics(capacity, distances[node])
    return result


def max_min_routes_from_source(
    graph: nx.Graph,
    source: Node,
    *,
    capacity_attr: str = "SKR",
    targets: Iterable[Node] | None = None,
) -> dict[Node, Route]:
    """Obtiene rutas max-min exactas con una frontera de Pareto por nodo.

    Guardar solo la ruta de mayor capacidad a cada nodo no basta para el
    desempate global por saltos: una alternativa algo peor pero más corta puede
    alcanzar después el mismo cuello de botella óptimo. La frontera conserva
    exactamente las etiquetas no dominadas ``(capacidad, saltos)`` y ejecuta
    una sola exploración multi-etiqueta por origen.
    """
    if source not in graph:
        raise nx.NodeNotFound(f"Nodo ausente: {source!r}")
    requested = set(graph if targets is None else targets)
    unknown = requested - set(graph)
    if unknown:
        raise nx.NodeNotFound(f"Nodos ausentes: {sorted(unknown, key=node_key)}")

    source_label = _Label(
        math.inf, 0, (node_key(source),), (source,)
    )
    frontiers: dict[Node, list[_Label]] = defaultdict(list)
    frontiers[source].append(source_label)
    active = {id(source_label)}
    serial = itertools.count()
    heap = [
        (
            -source_label.capacity,
            source_label.hops,
            source_label.path_key,
            next(serial),
            source,
            source_label,
        )
    ]

    while heap:
        _, _, _, _, node, label = heapq.heappop(heap)
        if id(label) not in active:
            continue
        for neighbour in sorted(graph.neighbors(node), key=node_key):
            edge_capacity = float(graph[node][neighbour][capacity_attr])
            candidate = _Label(
                min(label.capacity, edge_capacity),
                label.hops + 1,
                label.path_key + (node_key(neighbour),),
                label.path + (neighbour,),
            )
            existing = frontiers[neighbour]
            if any(_dominates(item, candidate) for item in existing):
                continue
            dominated = [item for item in existing if _dominates(candidate, item)]
            for item in dominated:
                active.discard(id(item))
                existing.remove(item)
            existing.append(candidate)
            active.add(id(candidate))
            heapq.heappush(
                heap,
                (
                    -candidate.capacity,
                    candidate.hops,
                    candidate.path_key,
                    next(serial),
                    neighbour,
                    candidate,
                ),
            )

    result = {}
    for target in requested:
        labels = frontiers.get(target)
        if not labels:
            continue
        optimum = min(
            labels,
            key=lambda label: (
                -label.capacity,
                label.hops,
                label.path_key,
            ),
        )
        result[target] = Route(optimum.capacity, optimum.path)
    return result


def shortest_max_bottleneck_path(
    graph: nx.Graph,
    source: Node,
    target: Node,
    *,
    capacity_attr: str = "SKR",
) -> tuple[float, list[Node]]:
    """API de compatibilidad: min-hops, max bottleneck, desempate estable."""
    if target not in graph:
        raise nx.NodeNotFound(f"Nodo ausente: {target!r}")
    route = min_hops_routes_from_source(
        graph, source, capacity_attr=capacity_attr
    ).get(target)
    return (route.bottleneck, list(route.path)) if route else (0.0, [])


def max_bottleneck_min_hops_path(
    graph: nx.Graph,
    source: Node,
    target: Node,
    *,
    capacity_attr: str = "SKR",
) -> tuple[float, list[Node]]:
    """API de compatibilidad: max-min, min-hops, desempate estable."""
    if target not in graph:
        raise nx.NodeNotFound(f"Nodo ausente: {target!r}")
    route = max_min_routes_from_source(
        graph, source, capacity_attr=capacity_attr, targets=[target]
    ).get(target)
    return (route.bottleneck, list(route.path)) if route else (0.0, [])


def compare_route_metrics(
    graph: nx.Graph,
    pairs: Iterable[tuple[Node, Node]] | None = None,
    *,
    capacity_attr: str = "SKR",
) -> list[dict[str, object]]:
    """Compara todos los pares sin reconstruir caminos ni distancias físicas."""
    nodes = sorted(graph, key=node_key)
    pair_list = (
        list(itertools.combinations(nodes, 2))
        if pairs is None
        else list(pairs)
    )
    positions_by_source: dict[Node, list[tuple[int, Node]]] = defaultdict(list)
    for position, (source, target) in enumerate(pair_list):
        if source not in graph or target not in graph:
            raise nx.NodeNotFound(f"Nodo ausente: {source!r} o {target!r}")
        positions_by_source[source].append((position, target))

    rows: list[dict[str, object] | None] = [None] * len(pair_list)
    for source in sorted(positions_by_source, key=node_key):
        shortest = min_hops_metrics_from_source(
            graph, source, capacity_attr=capacity_attr
        )
        widest = max_min_metrics_from_source(
            graph, source, capacity_attr=capacity_attr
        )
        for position, target in positions_by_source[source]:
            if target not in shortest or target not in widest:
                raise nx.NetworkXNoPath(
                    f"No hay camino entre {source!r} y {target!r}"
                )
            first = shortest[target]
            second = widest[target]
            if first.bottleneck == 0:
                gain = math.nan if second.bottleneck == 0 else math.inf
            else:
                gain = second.bottleneck / first.bottleneck
            rows[position] = {
                "origen": str(source),
                "destino": str(target),
                "sp_hops": first.hops,
                "sp_skr_bottleneck": first.bottleneck,
                "mqr_hops": second.hops,
                "mqr_skr_bottleneck": second.bottleneck,
                "skr_gain": gain,
                "hop_overhead": second.hops - first.hops,
            }
    return [row for row in rows if row is not None]


def compare_routes(
    graph: nx.Graph,
    pairs: Iterable[tuple[Node, Node]] | None = None,
    *,
    capacity_attr: str = "SKR",
    distance_attr: str = "dist_km",
) -> list[dict[str, object]]:
    """Compara políticas, reutilizando los recorridos de cada origen."""
    nodes = sorted(graph, key=node_key)
    pair_list = (
        list(itertools.combinations(nodes, 2))
        if pairs is None
        else list(pairs)
    )
    positions_by_source: dict[Node, list[tuple[int, Node]]] = defaultdict(list)
    for position, (source, target) in enumerate(pair_list):
        if source not in graph or target not in graph:
            raise nx.NodeNotFound(f"Nodo ausente: {source!r} o {target!r}")
        positions_by_source[source].append((position, target))

    rows: list[dict[str, object] | None] = [None] * len(pair_list)
    for source in sorted(positions_by_source, key=node_key):
        requested = positions_by_source[source]
        min_hops = min_hops_routes_from_source(
            graph, source, capacity_attr=capacity_attr
        )
        max_min = max_min_routes_from_source(
            graph,
            source,
            capacity_attr=capacity_attr,
            targets=(target for _, target in requested),
        )
        for position, target in requested:
            shortest = min_hops.get(target)
            widest = max_min.get(target)
            if shortest is None or widest is None:
                raise nx.NetworkXNoPath(
                    f"No hay camino entre {source!r} y {target!r}"
                )
            if shortest.bottleneck == 0:
                gain = math.nan if widest.bottleneck == 0 else math.inf
            else:
                gain = widest.bottleneck / shortest.bottleneck
            rows[position] = {
                "origen": str(source),
                "destino": str(target),
                "sp_hops": shortest.hops,
                "sp_dist_km": path_distance(
                    graph, shortest.path, distance_attr
                ),
                "sp_skr_bottleneck": shortest.bottleneck,
                "sp_path": " -> ".join(map(str, shortest.path)),
                "mqr_hops": widest.hops,
                "mqr_dist_km": path_distance(
                    graph, widest.path, distance_attr
                ),
                "mqr_skr_bottleneck": widest.bottleneck,
                "mqr_path": " -> ".join(map(str, widest.path)),
                "skr_gain": gain,
                "hop_overhead": widest.hops - shortest.hops,
            }
    return [row for row in rows if row is not None]
