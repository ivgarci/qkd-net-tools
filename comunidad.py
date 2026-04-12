# SPDX-License-Identifier: GPL-3.0-only
"""Louvain communities on the named adjacency graph."""
from __future__ import annotations

import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from grafo_io import DEFAULT_NAMED_ADJACENCY, load_named_adjacency


def main() -> None:
    G = load_named_adjacency(DEFAULT_NAMED_ADJACENCY)

    partition = community_louvain.best_partition(G)
    n_comm = len(set(partition.values()))

    plt.figure(figsize=(12, 10))
    palette = plt.cm.tab20(np.linspace(0, 1, max(n_comm, 1)))
    node_color = [palette[partition[node] % len(palette)] for node in G.nodes()]

    nx.draw_networkx(
        G,
        node_color=node_color,
        node_size=50,
        with_labels=False,
        edge_color="lightgray",
    )
    plt.title("Community structure (Louvain)")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig("community_structure_visualization.png", dpi=200)
    plt.show()

    print(f"Detected communities: {n_comm}")


if __name__ == "__main__":
    main()
