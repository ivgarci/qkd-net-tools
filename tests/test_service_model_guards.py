import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analisis"))

import capacidad_servicio_ataques as capacity  # noqa: E402
import fallos_adversariales as adversarial  # noqa: E402
import fallos_dispositivo as device  # noqa: E402


def _write_partial_link_table(directory: pathlib.Path) -> None:
    pd.DataFrame([{
        "caso": "CyL",
        "nodo_u": "A",
        "nodo_v": "B",
        "dist_km": 10.0,
        "SKR_bits_pulso": 0.1,
    }]).to_csv(directory / "skr_per_link.csv", index=False)


def _two_edge_graph(with_skr: bool = False) -> nx.Graph:
    graph = nx.Graph()
    attrs = {"skr": 0.1, "SKR": 0.1} if with_skr else {}
    graph.add_edge("A", "B", **attrs)
    graph.add_edge("B", "C", **attrs)
    return graph


class MissingPhysicalInputsTests(unittest.TestCase):
    def test_capacity_rejects_missing_skr_instead_of_imputing_45_km(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = pathlib.Path(tmp)
            _write_partial_link_table(data)
            with patch.object(capacity, "DATA", str(data)):
                with self.assertRaisesRegex(ValueError, "1 aristas sin SKR"):
                    capacity._asignar_skr_desde_csv(_two_edge_graph(), "CyL")

    def test_device_faults_reject_missing_edge_distance(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = pathlib.Path(tmp)
            _write_partial_link_table(data)
            loaders = {"cyl": lambda: _two_edge_graph(with_skr=True)}
            with (
                patch.object(device, "DATA", str(data)),
                patch.object(device, "CARGADORES", loaders),
            ):
                with self.assertRaisesRegex(
                    ValueError, "1 aristas sin distancia"
                ):
                    device.cargar_red("cyl")

    def test_adversarial_faults_reject_missing_edge_distance(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = pathlib.Path(tmp)
            _write_partial_link_table(data)
            with patch.object(adversarial, "DATA", str(data)):
                with self.assertRaisesRegex(
                    ValueError, "1 aristas sin distancia"
                ):
                    adversarial.dist_por_arista(
                        _two_edge_graph(with_skr=True), "cyl"
                    )


class PortabilityTests(unittest.TestCase):
    def test_default_log_directories_are_inside_repository(self):
        for module in (capacity, device, adversarial):
            self.assertTrue(
                pathlib.Path(module.LOG_DIR).is_relative_to(ROOT),
                module.LOG_DIR,
            )


class FrozenRoutePolicyTests(unittest.TestCase):
    def test_frozen_routes_are_insertion_order_independent(self):
        edges = [
            ("A", "C", 0.5),
            ("C", "D", 0.5),
            ("A", "B", 0.5),
            ("B", "D", 0.5),
        ]
        results = []
        for ordered_edges in (edges, list(reversed(edges))):
            graph = nx.Graph()
            for first, second, skr in ordered_edges:
                graph.add_edge(first, second, skr=skr, SKR=skr)
            widest, shortest = adversarial.rutas_congeladas(
                graph, {}, [("A", "D")]
            )
            results.append((widest, shortest))

        expected = [
            frozenset(("A", "B")),
            frozenset(("B", "D")),
        ]
        for widest, shortest in results:
            self.assertEqual(expected, widest[0])
            self.assertEqual(expected, shortest[0])


if __name__ == "__main__":
    unittest.main()
