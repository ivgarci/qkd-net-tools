"""
Utilidades compartidas para scripts de análisis de redes QKD.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx

BASE_REPO = os.path.dirname(os.path.abspath(__file__))


def load_graph(adj_csv: str) -> nx.Graph:
    """Carga un grafo NetworkX desde una matriz de adyacencia CSV."""
    adj = pd.read_csv(adj_csv, index_col=0)
    validate_adjacency_matrix(adj)
    return nx.from_pandas_adjacency(adj)


def relative_gcc(G: nx.Graph) -> float:
    """Fracción de nodos en la componente gigante (GCC) respecto al grafo original."""
    if G.number_of_nodes() == 0:
        return 0.0
    if nx.is_connected(G):
        return 1.0
    return len(max(nx.connected_components(G), key=len)) / G.number_of_nodes()


def robustness_index(s_values) -> float:
    """R ≈ integral de S(p) dp, suma de Riemann con Δp=0.01."""
    return round(float(np.sum(s_values) * 0.01), 4)


def p_star(p_values, s_values, threshold: float = 0.5):
    """Mínimo p (%) tal que S_rel < threshold. Devuelve None si no se alcanza."""
    for p, s in zip(p_values, s_values):
        if s < threshold:
            return int(p)
    return None


def validate_adjacency_matrix(df: pd.DataFrame) -> None:
    """Valida simetría, no negatividad y consistencia de índices."""
    arr = df.values.astype(float)
    if not np.allclose(arr, arr.T, atol=1e-8, equal_nan=False):
        raise ValueError("La matriz de adyacencia no es simétrica")
    if (arr < 0).any():
        raise ValueError("La matriz de adyacencia contiene valores negativos")
    if list(df.index) != list(df.columns):
        raise ValueError("Los índices de fila y columna no coinciden")


def get_thesis_style() -> dict:
    """Devuelve rcParams estándar para figuras de tesis."""
    return {
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 150,
        'text.usetex': False,
    }
