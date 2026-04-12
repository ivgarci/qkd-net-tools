# SPDX-License-Identifier: GPL-3.0-only
"""Clustering coefficient distribution for the named adjacency graph."""
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from grafo_io import DEFAULT_NAMED_ADJACENCY, load_named_adjacency


def main() -> None:
    G = load_named_adjacency(DEFAULT_NAMED_ADJACENCY)

    avg_clustering_coefficient = nx.average_clustering(G)
    node_clustering = nx.clustering(G)

    plt.figure(figsize=(10, 6))
    plt.hist(list(node_clustering.values()), bins=np.linspace(0, 1, 20), color="skyblue", edgecolor="black")
    plt.title("Clustering coefficient distribution")
    plt.xlabel("Clustering coefficient")
    plt.ylabel("Number of nodes")
    plt.grid(axis="y", alpha=0.75)

    avg_clustering_coefficient_text = f"Average clustering coefficient: {avg_clustering_coefficient:.2f}"
    plt.figtext(
        0.5,
        -0.05,
        avg_clustering_coefficient_text,
        ha="center",
        fontsize=12,
        bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5},
    )

    plt.tight_layout()
    plt.savefig("clustering_coefficient_distribution.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
