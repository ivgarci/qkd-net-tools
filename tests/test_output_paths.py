"""Guards against silently writing analysis artefacts outside the repository."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_figure_generators_default_to_repository_paths():
    scripts = (
        "analisis/generar_figuras_skr_routing.py",
        "analisis/generar_figuras_service_resilience.py",
        "analisis/generar_figuras_fault_aware.py",
        "analisis/generar_figuras_pam_generation.py",
    )
    for relative_path in scripts:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "'..', '..', '..', 'articulos'" not in source
        assert "/Users/igarcia" not in source


def test_analysis_logs_do_not_document_machine_specific_paths():
    scripts = (
        "analisis/capacidad_servicio_ataques.py",
        "analisis/fallos_dispositivo.py",
        "analisis/fallos_adversariales.py",
        "analisis/betweenness_ponderada.py",
        "analisis/null_model_er.py",
    )
    for relative_path in scripts:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "/Users/igarcia" not in source
