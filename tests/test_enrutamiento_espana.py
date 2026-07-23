import pathlib
import unittest

import numpy as np

from analisis.enrutamiento_qkd import build_qkd_graph, compare_routing, parse_args


ROOT = pathlib.Path(__file__).resolve().parents[1]


class SpainRoutingGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = ROOT / "datos" / "espana"
        cls.graph = build_qkd_graph(
            str(data / "AdjacencyMatrixNamed45.csv"),
            str(data / "peninsula_1000.csv"),
            distance_factor=1.0,
        )
        cls.results = compare_routing(cls.graph, metrics_only=True)

    def test_canonical_dimensions(self):
        self.assertEqual(950, self.graph.number_of_nodes())
        self.assertEqual(5_681, self.graph.number_of_edges())
        self.assertEqual(450_775, len(self.results))

    def test_canonical_exact_aggregate(self):
        self.assertAlmostEqual(
            2.179604963197868,
            float(self.results["skr_gain"].mean()),
            places=14,
        )
        self.assertAlmostEqual(
            30.061651600022184,
            float(self.results["hop_overhead"].mean()),
            places=14,
        )
        self.assertEqual(
            448_157,
            int((self.results["skr_gain"] > 1.0).sum()),
        )
        self.assertTrue(
            np.array_equal(
                self.results["hop_overhead"] >= 0,
                np.ones(len(self.results), dtype=bool),
            )
        )

    def test_cli_defaults_remain_cyl_and_geodesic(self):
        args = parse_args([])
        self.assertEqual("cyl", args.case)
        self.assertEqual(1.0, args.distance_factor)


if __name__ == "__main__":
    unittest.main()
