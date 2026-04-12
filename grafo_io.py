# SPDX-License-Identifier: GPL-3.0-only
"""Load NetworkX graphs from adjacency matrices on disk."""
from __future__ import annotations

import pandas as pd
import networkx as nx

DEFAULT_NAMED_ADJACENCY = "AdjacencyMatrixNamed45.csv"


def load_named_adjacency(path: str = DEFAULT_NAMED_ADJACENCY) -> nx.Graph:
    """Build an undirected graph from a square CSV with row/column node names.

    Non-binary entries are preserved as edge weights (suited for weighted QKD-style links).
    """
    adjacency_matrix = pd.read_csv(path, index_col=0)
    return nx.from_pandas_adjacency(adjacency_matrix)
