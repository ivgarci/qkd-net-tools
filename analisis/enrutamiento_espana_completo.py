"""Ejecuta el análisis all-pairs canónico de la red QKD de España.

Este módulo es un punto de entrada compatible para el caso peninsular. Toda la
lógica científica vive en :mod:`analisis.routing_core` y el runner compartido
de :mod:`analisis.enrutamiento_qkd`.

Se comparan exactamente los 450.775 pares no ordenados de los 950 nodos:

* mínimos saltos; entre empates, mayor cuello de botella SKR;
* máximo cuello de botella SKR; entre empates, mínimos saltos.

Por defecto se usa la distancia geodésica almacenada por las coordenadas
(``--distance-factor 1.0``), que es el escenario canónico documentado en la
tesis. ``--distance-factor 1.25`` permite ejecutar, de forma explícita y
etiquetada, una hipótesis alternativa de longitud de fibra. No se imputan
coordenadas, no se recortan tasas y no se escriben ficheros fuera del repo.

Salidas canónicas:

* ``datos/resultados_papers/enrutamiento_espana_allpairs.csv``
* ``datos/resultados_papers/enrutamiento_espana_summary.csv``
* ``figuras/comparacion_rutas_qkd_espana.pdf``
* ``figuras/comparacion_rutas_qkd_espana.png``
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from analisis.enrutamiento_qkd import run_case


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrutamiento QKD all-pairs de España."
    )
    parser.add_argument(
        "--distance-factor",
        type=float,
        default=1.0,
        help=(
            "Multiplicador explícito de la distancia geodésica "
            "(canónico: 1.0; escenario hipotético de fibra: 1.25)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_case("espana", distance_factor=args.distance_factor)


if __name__ == "__main__":
    main()
