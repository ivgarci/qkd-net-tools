"""Guard against reintroducing an independent SKR formula in figure scripts."""

import ast
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "analisis"
    / "generar_figuras_skr_routing.py"
)


def test_figure_generator_imports_canonical_skr_without_redefining_it():
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "protocols.skr_bb84"
        for alias in node.names
    }
    local_functions = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    assert "skr_bb84_asymptotic" in imported
    assert "skr_bb84" not in local_functions
    assert "h2" not in local_functions


def test_default_figure_output_stays_inside_repository():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "'figuras', 'qkd_skr_routing'" in source
    assert "../../../articulos" not in source
    assert "'..', '..', '..', 'articulos'" not in source
