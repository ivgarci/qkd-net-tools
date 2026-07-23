import importlib.util
import pathlib
import unittest

import networkx as nx


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


if __name__ == "__main__":
    unittest.main()
