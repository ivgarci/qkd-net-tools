import importlib.util
import pathlib
import tempfile
import unittest

import networkx as nx
import random

from analisis.routing_core import (
    compare_route_metrics,
    compare_routes,
    load_qkd_graph,
    max_min_metrics_from_source,
    max_min_routes_from_source,
    min_hops_metrics_from_source,
    min_hops_routes_from_source,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "enrutamiento_qkd",
    ROOT / "analisis" / "enrutamiento_qkd.py",
)
ROUTING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUTING)


def graph(edges):
    graph = nx.Graph()
    for u, v, skr in edges:
        graph.add_edge(u, v, SKR=skr, dist_km=1.0)
    return graph


class RoutingTieBreakTests(unittest.TestCase):
    def test_cyl_rejects_noncanonical_distance_factor(self):
        with self.assertRaisesRegex(ValueError, "CyL solo admite"):
            ROUTING.run_case("cyl", distance_factor=1.25)

    def test_shortest_path_prefers_larger_bottleneck(self):
        G = graph([
            ("A", "B", 0.3),
            ("B", "D", 0.3),
            ("A", "C", 0.8),
            ("C", "D", 0.8),
        ])
        bottleneck, path = ROUTING.shortest_max_bottleneck_path(G, "A", "D")
        self.assertEqual(0.8, bottleneck)
        self.assertEqual(["A", "C", "D"], path)

    def test_shortest_final_tie_is_global_lexicographic(self):
        G = graph([
            ("S", "Z", 1.0),
            ("Z", "U", 1.0),
            ("S", "A", 0.5),
            ("A", "U", 0.5),
            ("U", "T", 0.4),
        ])
        bottleneck, path = ROUTING.shortest_max_bottleneck_path(G, "S", "T")
        self.assertEqual(0.4, bottleneck)
        self.assertEqual(["S", "A", "U", "T"], path)

    def test_max_min_path_prefers_fewer_hops(self):
        G = graph([
            ("A", "B", 0.8),
            ("B", "D", 0.8),
            ("A", "C", 0.8),
            ("C", "E", 0.8),
            ("E", "D", 0.8),
        ])
        bottleneck, path = ROUTING.max_skr_path(G, "A", "D")
        self.assertEqual(0.8, bottleneck)
        self.assertEqual(["A", "B", "D"], path)

    def test_max_min_accepts_more_hops_for_larger_bottleneck(self):
        G = graph([
            ("A", "D", 0.2),
            ("A", "B", 0.7),
            ("B", "C", 0.7),
            ("C", "D", 0.7),
        ])
        bottleneck, path = ROUTING.max_skr_path(G, "A", "D")
        self.assertEqual(0.7, bottleneck)
        self.assertEqual(["A", "B", "C", "D"], path)

    def test_max_min_keeps_shorter_lower_capacity_prefix(self):
        G = graph([
            ("S", "A", 1.0),
            ("A", "B", 1.0),
            ("B", "U", 1.0),
            ("S", "U", 0.5),
            ("U", "T", 0.5),
        ])
        bottleneck, path = ROUTING.max_skr_path(G, "S", "T")
        self.assertEqual(0.5, bottleneck)
        self.assertEqual(["S", "U", "T"], path)

    def test_max_min_final_tie_is_global_lexicographic(self):
        G = graph([
            ("S", "Z", 1.0),
            ("Z", "U", 1.0),
            ("S", "A", 0.5),
            ("A", "U", 0.5),
            ("U", "T", 0.4),
        ])
        bottleneck, path = ROUTING.max_skr_path(G, "S", "T")
        self.assertEqual(0.4, bottleneck)
        self.assertEqual(["S", "A", "U", "T"], path)

    def test_final_tie_is_lexicographic_and_insertion_independent(self):
        edges = [
            ("A", "C", 0.5),
            ("C", "D", 0.5),
            ("A", "B", 0.5),
            ("B", "D", 0.5),
        ]
        for candidate in (graph(edges), graph(reversed(edges))):
            _, path = ROUTING.max_skr_path(candidate, "A", "D")
            self.assertEqual(["A", "B", "D"], path)


class RoutingCoreTests(unittest.TestCase):
    def test_single_source_apis_cover_every_reachable_node(self):
        G = graph([
            ("A", "B", 0.8),
            ("B", "C", 0.6),
            ("A", "C", 0.5),
            ("C", "D", 0.4),
        ])
        shortest = min_hops_routes_from_source(G, "A")
        widest = max_min_routes_from_source(G, "A")
        self.assertEqual(set(G), set(shortest))
        self.assertEqual(set(G), set(widest))
        self.assertEqual(("A", "C"), shortest["C"].path)
        self.assertEqual(("A", "B", "C"), widest["C"].path)

    def test_batch_comparison_preserves_requested_pair_order(self):
        G = graph([
            ("A", "B", 0.8),
            ("B", "C", 0.6),
            ("A", "C", 0.5),
        ])
        rows = compare_routes(G, [("B", "C"), ("A", "C")])
        self.assertEqual(
            [("B", "C"), ("A", "C")],
            [(row["origen"], row["destino"]) for row in rows],
        )

    def test_fast_metrics_match_exact_path_algorithms(self):
        for seed in range(20):
            rng = random.Random(seed)
            G = nx.gnp_random_graph(9, 0.3, seed=seed)
            if not nx.is_connected(G):
                continue
            for first, second in G.edges:
                G[first][second]["SKR"] = rng.choice(
                    [0.1, 0.2, 0.4, 0.8]
                )
                G[first][second]["dist_km"] = 1.0
            with self.subTest(seed=seed):
                for source in G:
                    path_shortest = min_hops_routes_from_source(G, source)
                    path_widest = max_min_routes_from_source(G, source)
                    fast_shortest = min_hops_metrics_from_source(G, source)
                    fast_widest = max_min_metrics_from_source(G, source)
                    self.assertEqual(
                        {
                            node: (route.bottleneck, route.hops)
                            for node, route in path_shortest.items()
                        },
                        {
                            node: (route.bottleneck, route.hops)
                            for node, route in fast_shortest.items()
                        },
                    )
                    self.assertEqual(
                        {
                            node: (route.bottleneck, route.hops)
                            for node, route in path_widest.items()
                        },
                        {
                            node: (route.bottleneck, route.hops)
                            for node, route in fast_widest.items()
                        },
                    )

    def test_fast_batch_matches_path_batch_metrics(self):
        G = graph([
            ("A", "B", 0.8),
            ("B", "C", 0.6),
            ("A", "C", 0.5),
        ])
        full = compare_routes(G)
        fast = compare_route_metrics(G)
        columns = set(fast[0])
        self.assertEqual(
            [{key: row[key] for key in columns} for row in full],
            fast,
        )

    def test_loader_applies_configurable_distance_factor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            adjacency = root / "adjacency.csv"
            coordinates = root / "coordinates.csv"
            adjacency.write_text(",A,B\nA,0,1\nB,1,0\n", encoding="utf-8")
            coordinates.write_text(
                "Población;Latitud;Longitud\nA;40,0;-4,0\nB;41,0;-3,0\n",
                encoding="utf-8",
            )
            G = load_qkd_graph(
                adjacency,
                coordinates,
                lambda distance: distance * 2,
                distance_factor=1.25,
                haversine=lambda *_: 10.0,
            )
        self.assertEqual(12.5, G["A"]["B"]["dist_km"])
        self.assertEqual(25.0, G["A"]["B"]["SKR"])


if __name__ == "__main__":
    unittest.main()
