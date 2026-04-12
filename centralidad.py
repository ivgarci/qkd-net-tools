# SPDX-License-Identifier: GPL-3.0-only
"""Top-k centrality bar charts from the named adjacency matrix."""
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from grafo_io import DEFAULT_NAMED_ADJACENCY, load_named_adjacency

TOP_N = 10


def main() -> None:
    G = load_named_adjacency(DEFAULT_NAMED_ADJACENCY)

    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)

    centrality_measures_df = pd.DataFrame(
        {
            "Degree Centrality": degree_centrality,
            "Betweenness Centrality": betweenness_centrality,
            "Closeness Centrality": closeness_centrality,
        }
    )

    top_degree = centrality_measures_df["Degree Centrality"].sort_values(ascending=False).head(TOP_N)
    top_betweenness = centrality_measures_df["Betweenness Centrality"].sort_values(ascending=False).head(TOP_N)
    top_closeness = centrality_measures_df["Closeness Centrality"].sort_values(ascending=False).head(TOP_N)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    top_degree.plot(kind="bar", ax=axes[0], color="skyblue")
    axes[0].set_title("Top nodes by degree centrality")
    axes[0].set_ylabel("Centrality score")
    axes[0].set_xlabel("Node")

    top_betweenness.plot(kind="bar", ax=axes[1], color="lightgreen")
    axes[1].set_title("Top nodes by betweenness centrality")
    axes[1].set_xlabel("Node")

    top_closeness.plot(kind="bar", ax=axes[2], color="salmon")
    axes[2].set_title("Top nodes by closeness centrality")
    axes[2].set_xlabel("Node")

    plt.tight_layout()
    plt.savefig("centrality_measures_visualization.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
