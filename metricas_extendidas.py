# SPDX-License-Identifier: GPL-3.0-only
"""
Extended structural metrics: giant-component paths, global efficiency,
degree assortativity, Louvain modularity. Works with weighted graphs if the
adjacency CSV contains non-binary weights.
"""
from __future__ import annotations

import json

import community as community_louvain
import networkx as nx
import pandas as pd

from grafo_io import DEFAULT_NAMED_ADJACENCY, load_named_adjacency


def _partition_dict_to_sets(partition: dict) -> list[set]:
    buckets: dict[int, set] = {}
    for node, cid in partition.items():
        buckets.setdefault(cid, set()).add(node)
    return list(buckets.values())


def giant_component_subgraph(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G.copy()
    nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(nodes).copy()


def metrics_for_graph(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    out: dict = {
        "n_nodes": n,
        "n_edges": m,
        "density": nx.density(G) if n > 1 else 0.0,
        "is_connected": nx.is_connected(G) if n else False,
        "is_weighted": nx.is_weighted(G),
    }
    if n == 0:
        out.update(
            {
                "n_giant": 0,
                "giant_fraction": 0.0,
                "diameter_giant": None,
                "avg_shortest_path_length_giant": None,
                "global_efficiency_giant": 0.0,
                "avg_clustering": 0.0,
                "degree_assortativity": None,
                "modularity_louvain": None,
                "n_communities_louvain": None,
            }
        )
        return out

    giant = giant_component_subgraph(G)
    ng = giant.number_of_nodes()
    out["n_giant"] = ng
    out["giant_fraction"] = ng / n

    if ng < 2:
        out["diameter_giant"] = 0
        out["avg_shortest_path_length_giant"] = 0.0
        out["global_efficiency_giant"] = 0.0
    else:
        out["diameter_giant"] = nx.diameter(giant)
        out["avg_shortest_path_length_giant"] = nx.average_shortest_path_length(giant)
        out["global_efficiency_giant"] = nx.global_efficiency(giant)

    out["avg_clustering"] = nx.average_clustering(G)

    try:
        out["degree_assortativity"] = nx.degree_assortativity_coefficient(G)
    except Exception:
        out["degree_assortativity"] = None

    partition = community_louvain.best_partition(G)
    communities = _partition_dict_to_sets(partition)
    out["modularity_louvain"] = nx.algorithms.community.modularity(G, communities)
    out["n_communities_louvain"] = len(communities)

    return out


def main() -> None:
    path = DEFAULT_NAMED_ADJACENCY
    G = load_named_adjacency(path)
    row = metrics_for_graph(G)
    row["adjacency_file"] = path

    df = pd.DataFrame([row])
    out_csv = "extended_network_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print(json.dumps({k: row[k] for k in sorted(row) if k != "adjacency_file"}, indent=2, default=str))


if __name__ == "__main__":
    main()
