import json
import pathlib
import sys
import tempfile
import unittest

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analisis"))

import delta_sensitivity_espana as delta  # noqa: E402
import enrutamiento_adif_completo as adif  # noqa: E402


class DeltaSensitivityScenarioTests(unittest.TestCase):
    def test_records_edge_set_diff_and_writes_only_own_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            adjacency = directory / "adj.csv"
            coordinates = directory / "coords.csv"
            output = directory / "delta.json"

            pd.DataFrame(
                [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
                index=["A", "B", "C"],
                columns=["A", "B", "C"],
            ).to_csv(adjacency)
            pd.DataFrame([
                {"Población": "A", "Latitud": 0.0, "Longitud": 0.0},
                {"Población": "B", "Latitud": 0.0, "Longitud": 0.1},
                {"Población": "C", "Latitud": 0.0, "Longitud": 0.2},
            ]).to_csv(coordinates, sep=";", index=False)

            result = delta.run_sensitivity(
                (15.0,), 1.0, adjacency, coordinates
            )
            scenario = result["scenarios"][0]
            diff = scenario["edge_diff_vs_snapshot"]
            self.assertEqual(1, diff["added_edges"])
            self.assertEqual(0, diff["removed_edges"])
            self.assertEqual(3, scenario["routing"]["unordered_pairs"])
            self.assertIn("alternativo", result["warning"])

            delta.write_json_deterministic(result, output)
            first = output.read_bytes()
            delta.write_json_deterministic(result, output)
            self.assertEqual(first, output.read_bytes())
            self.assertNotIn("tablas_skr_routing", output.name)
            json.loads(first)

    def test_rho_defaults_to_one_and_changes_only_model_distance(self):
        coordinates = {"A": (0.0, 0.0), "B": (0.0, 0.1)}
        graph = delta.build_alternative_graph(coordinates, 20.0)
        edge = graph["A"]["B"]
        self.assertAlmostEqual(edge["haversine_km"], edge["dist_km"])


class AdifProxyScenarioTests(unittest.TestCase):
    def test_preserves_zero_skr_and_separates_infinite_gain_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            nodes = directory / "nodes.csv"
            edges = directory / "edges.csv"
            pd.DataFrame([
                {
                    "cod": "A",
                    "nombre": "A",
                    "lat": 0.0,
                    "lon": 0.0,
                    "conectado": "SI",
                },
                {
                    "cod": "B",
                    "nombre": "B",
                    "lat": 0.0,
                    "lon": 0.1,
                    "conectado": "SI",
                },
                {
                    "cod": "C",
                    "nombre": "C",
                    "lat": 0.0,
                    "lon": 0.2,
                    "conectado": "SI",
                },
            ]).to_csv(nodes, index=False)
            pd.DataFrame([
                {"cod": "A", "vecino_cod": "B", "dist_km": 250.0},
                {"cod": "A", "vecino_cod": "C", "dist_km": 10.0},
                {"cod": "C", "vecino_cod": "B", "dist_km": 10.0},
            ]).to_csv(edges, index=False)

            graph, coordinates, info = adif.build_adif_proxy_graph(nodes, edges)
            self.assertEqual(1, info["zero_skr_edges"])
            self.assertEqual(0.0, graph["A"]["B"]["SKR"])
            summary = adif.summarize_routing(graph, coordinates)
            self.assertEqual(
                {"finite": 2, "rescued_from_zero": 1, "both_zero": 0},
                summary["gain_status_counts"],
            )
            self.assertEqual(2, summary["finite_skr_gain"]["count"])
            # JSON estricto: no aparecen valores numéricos Infinity/NaN.
            json.dumps(summary, allow_nan=False)

    def test_conflicting_duplicate_distance_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            nodes = directory / "nodes.csv"
            edges = directory / "edges.csv"
            pd.DataFrame([
                {
                    "cod": "A",
                    "nombre": "A",
                    "lat": 0,
                    "lon": 0,
                    "conectado": "SI",
                },
                {
                    "cod": "B",
                    "nombre": "B",
                    "lat": 0,
                    "lon": 1,
                    "conectado": "SI",
                },
            ]).to_csv(nodes, index=False)
            pd.DataFrame([
                {"cod": "A", "vecino_cod": "B", "dist_km": 10},
                {"cod": "B", "vecino_cod": "A", "dist_km": 11},
            ]).to_csv(edges, index=False)
            with self.assertRaisesRegex(ValueError, "contradictorias"):
                adif.build_adif_proxy_graph(nodes, edges)


if __name__ == "__main__":
    unittest.main()
