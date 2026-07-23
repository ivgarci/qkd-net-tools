"""Independent audit of the thesis CyL SKR/routing claims.

This module intentionally uses only the Python standard library and does not
import the production implementation. It is a numerical cross-check, not a
unit test that repeats the same code paths.
"""

import csv
import collections
import hashlib
import heapq
import math
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
ADJACENCY = ROOT / "datos" / "cyl" / "AdjacencyMatrixNamed45.csv"
COORDINATES = ROOT / "datos" / "cyl" / "cyl_1000.csv"
ALL_PAIRS = ROOT / "datos" / "enrutamiento_qkd_allpairs.csv"
SUMMARY = ROOT / "datos" / "enrutamiento_qkd_summary.csv"
LINK_RATES = ROOT / "datos" / "skr_per_link.csv"

EXPECTED_HASHES = {
    ADJACENCY: "430072428dedb25893468b581d416fd0a45bcd9655e68d085c0855e8952e52ef",
    COORDINATES: "ab12f5d6de0c1e9773d3812604640e06e7b10658a5a6135a53b076f52e42a9eb",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_entropy(probability):
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return (
        -probability * math.log2(probability)
        - (1.0 - probability) * math.log2(1.0 - probability)
    )


def thesis_skr(distance_km):
    """Chapter 4 asymptotic ideal single-photon estimate."""
    alpha = 0.2
    eta_det = 0.10
    mu = 0.5
    background_yield = 1e-6
    background_error = 0.5
    detector_error = 0.015
    error_correction = 1.16
    sifting = 0.5

    eta = eta_det * 10 ** (-alpha * distance_km / 10)
    signal_gain = 1 - (1 - background_yield) * math.exp(-mu * eta)
    signal_qber = (
        background_error * background_yield
        + detector_error * (1 - math.exp(-mu * eta))
    ) / signal_gain
    single_yield = 1 - (1 - background_yield) * (1 - eta)
    single_gain = mu * math.exp(-mu) * single_yield
    single_qber = (
        background_error * background_yield + detector_error * eta
    ) / single_yield
    return max(
        0.0,
        sifting
        * (
            single_gain * (1 - binary_entropy(single_qber))
            - signal_gain
            * error_correction
            * binary_entropy(signal_qber)
        ),
    )


def haversine(first, second):
    radius_km = 6371.0
    latitude_1, longitude_1 = first
    latitude_2, longitude_2 = second
    phi_1 = math.radians(latitude_1)
    phi_2 = math.radians(latitude_2)
    delta_phi = math.radians(latitude_2 - latitude_1)
    delta_lambda = math.radians(longitude_2 - longitude_1)
    haversine_angle = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi_1)
        * math.cos(phi_2)
        * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(haversine_angle))


def load_graph():
    with COORDINATES.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream, delimiter=";")
        coordinates = {
            row["Población"]: (float(row["Latitud"]), float(row["Longitud"]))
            for row in rows
        }

    with ADJACENCY.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream)
        node_names = next(rows)[1:]
        matrix_rows = list(rows)

    graph = {node: [] for node in node_names}
    edge_count = 0
    for row_index, row in enumerate(matrix_rows):
        source = row[0]
        for column_index, raw_value in enumerate(row[1:]):
            if column_index <= row_index or float(raw_value or 0) == 0:
                continue
            target = node_names[column_index]
            distance = haversine(coordinates[source], coordinates[target])
            graph[source].append((target, distance))
            graph[target].append((source, distance))
            edge_count += 1
    return node_names, graph, coordinates, edge_count


def lexicographic_costs(graph, source, primary):
    """Return exact (primary, secondary) costs from one source."""
    if primary == "hops":
        initial = (0, 0.0)

        def extend(cost, edge_distance):
            return cost[0] + 1, max(cost[1], edge_distance)

    elif primary == "max_edge":
        initial = (0.0, 0)

        def extend(cost, edge_distance):
            return max(cost[0], edge_distance), cost[1] + 1

    else:
        raise ValueError(primary)

    best = {source: initial}
    queue = [(*initial, source)]
    while queue:
        first, second, node = heapq.heappop(queue)
        if best[node] != (first, second):
            continue
        for neighbour, edge_distance in graph[node]:
            candidate = extend((first, second), edge_distance)
            if candidate < best.get(neighbour, (math.inf, math.inf)):
                best[neighbour] = candidate
                heapq.heappush(queue, (*candidate, neighbour))
    return best


def max_min_hops(graph, source, target, optimum_max_edge):
    """Fewest hops among paths attaining the already computed optimum.

    This must be a second phase. A single ``(max_edge, hops)`` label at an
    intermediate node can discard a shorter prefix whose primary cost later
    becomes tied after traversing a more restrictive edge.
    """
    queue = collections.deque([(source, 0)])
    visited = {source}
    while queue:
        node, hops = queue.popleft()
        if node == target:
            return hops
        for neighbour, edge_distance in graph[node]:
            if edge_distance <= optimum_max_edge and neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, hops + 1))
    raise AssertionError("The optimum bottleneck subgraph disconnected a pair")


def audit_all_pairs():
    nodes, graph, _, _ = load_graph()
    ratios = []
    overheads = []
    minimum_hop_rates = []
    for source_index, source in enumerate(nodes):
        shortest = lexicographic_costs(graph, source, "hops")
        max_min = lexicographic_costs(graph, source, "max_edge")
        for target in nodes[source_index + 1 :]:
            shortest_hops, shortest_max_edge = shortest[target]
            max_min_max_edge, _ = max_min[target]
            max_min_hop_count = max_min_hops(
                graph, source, target, max_min_max_edge
            )
            shortest_rate = thesis_skr(shortest_max_edge)
            max_min_rate = thesis_skr(max_min_max_edge)
            ratios.append(max_min_rate / shortest_rate)
            overheads.append(max_min_hop_count - shortest_hops)
            minimum_hop_rates.append(shortest_rate)
    return ratios, overheads, minimum_hop_rates


class ThesisSkrRoutingAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.graph, cls.coordinates, cls.edge_count = load_graph()
        cls.ratios, cls.overheads, cls.minimum_hop_rates = audit_all_pairs()

    def test_canonical_input_hashes(self):
        for path, expected in EXPECTED_HASHES.items():
            with self.subTest(path=path):
                self.assertEqual(expected, sha256(path))

    def test_input_dimensions_and_coverage(self):
        self.assertEqual(100, len(self.nodes))
        self.assertEqual(254, self.edge_count)
        self.assertTrue(set(self.nodes).issubset(self.coordinates))

    def test_declared_skr_reference_points(self):
        expected = {
            10: 6.4664382800499395e-3,
            50: 1.0188456107580379e-3,
            100: 9.996046924682088e-5,
            150: 8.253730896477398e-6,
        }
        for distance, rate in expected.items():
            with self.subTest(distance=distance):
                self.assertTrue(
                    math.isclose(
                        thesis_skr(distance), rate, rel_tol=1e-12
                    )
                )

    def test_reference_cut_is_between_190_and_191_km(self):
        self.assertGreater(thesis_skr(190), 1e-8)
        self.assertLess(thesis_skr(191), 1e-8)

    def test_all_unordered_pairs_are_covered(self):
        self.assertEqual(4950, len(self.ratios))

    def test_mean_bottleneck_ratio_supports_thesis_rounding(self):
        mean_ratio = sum(self.ratios) / len(self.ratios)
        self.assertTrue(
            math.isclose(
                mean_ratio, 1.4417703475208163, rel_tol=1e-12
            )
        )
        self.assertEqual(1.442, round(mean_ratio, 3))

    def test_mean_hop_overhead_supports_thesis_rounding(self):
        mean_overhead = sum(self.overheads) / len(self.overheads)
        self.assertTrue(
            math.isclose(
                mean_overhead, 3.993939393939394, rel_tol=1e-12
            )
        )
        self.assertEqual(3.994, round(mean_overhead, 3))

    def test_minimum_shortest_route_skr(self):
        self.assertTrue(
            math.isclose(
                min(self.minimum_hop_rates),
                1.280838767155935e-3,
                rel_tol=1e-12,
            )
        )

    def test_generated_pair_level_output_matches_independent_audit(self):
        with ALL_PAIRS.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(4950, len(rows))
        self.assertEqual(
            4950,
            len({(row["origen"], row["destino"]) for row in rows}),
        )
        generated_gain = sum(float(row["skr_gain"]) for row in rows) / len(rows)
        generated_overhead = (
            sum(int(row["hop_overhead"]) for row in rows) / len(rows)
        )
        self.assertTrue(
            math.isclose(
                generated_gain, 1.4417703475208163, rel_tol=1e-12
            )
        )
        self.assertTrue(
            math.isclose(
                generated_overhead, 3.993939393939394, rel_tol=1e-12
            )
        )

    def test_generated_summary_matches_pair_level_claims(self):
        with SUMMARY.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual(100, int(row["nodes"]))
        self.assertEqual(254, int(row["edges"]))
        self.assertEqual(4950, int(row["unordered_pairs"]))
        self.assertTrue(
            math.isclose(
                float(row["mean_skr_gain"]),
                1.4417703475208163,
                rel_tol=1e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                float(row["mean_hop_overhead"]),
                3.993939393939394,
                rel_tol=1e-12,
            )
        )

    def test_generated_link_rates_support_thesis_edge_claims(self):
        with LINK_RATES.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        by_case = collections.defaultdict(list)
        for row in rows:
            by_case[row["caso"]].append(float(row["SKR_bits_pulso"]))
            self.assertEqual("True", row["viable_QKD"])

        expected = {
            "CyL": (254, 1.280838767155935e-3, 6.810578187387255e-3),
            "España": (5681, 1.2805143646304957e-3, 7.868047301151721e-3),
        }
        self.assertEqual(set(expected), set(by_case))
        for case, (count, minimum, maximum) in expected.items():
            with self.subTest(case=case):
                self.assertEqual(count, len(by_case[case]))
                self.assertTrue(
                    math.isclose(minimum, min(by_case[case]), rel_tol=1e-12)
                )
                self.assertTrue(
                    math.isclose(maximum, max(by_case[case]), rel_tol=1e-12)
                )


if __name__ == "__main__":
    unittest.main()
