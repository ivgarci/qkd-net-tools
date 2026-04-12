# SPDX-License-Identifier: GPL-3.0-only
"""
Null models: G(n,m) Erdős–Rényi and configuration-model graphs with the same
degree sequence as the observed network. Compares summary metrics to the original graph.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd

from grafo_io import DEFAULT_NAMED_ADJACENCY, load_named_adjacency
from metricas_extendidas import metrics_for_graph


def configuration_simple_graph(deg_sequence: list[int], seed: int) -> nx.Graph:
    """Configuration model collapsed to a simple graph (self-loops dropped)."""
    rng = np.random.default_rng(seed)
    stub_seed = int(rng.integers(0, 2**31 - 1))
    H = nx.configuration_model(deg_sequence, seed=stub_seed)
    G = nx.Graph()
    G.add_edges_from((u, v) for u, v in H.edges() if u != v)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def sample_null_metrics(
    n_nodes: int,
    n_edges: int,
    deg_sequence: list[int],
    n_samples: int,
    base_seed: int,
) -> pd.DataFrame:
    rows = []
    for i in range(n_samples):
        seed_er = base_seed + 10_000 + i
        seed_cm = base_seed + 20_000 + i

        er = nx.gnm_random_graph(n_nodes, n_edges, seed=seed_er)
        rows.append({"model": "gnm", "sample": i, **metrics_for_graph(er)})

        cm = configuration_simple_graph(deg_sequence, seed_cm)
        rows.append({"model": "configuration", "sample": i, **metrics_for_graph(cm)})

    return pd.DataFrame(rows)


def main() -> None:
    base_seed = 42
    n_samples = 15
    path = DEFAULT_NAMED_ADJACENCY
    G = load_named_adjacency(path)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    deg_sequence = [d for _, d in G.degree()]

    observed = metrics_for_graph(G)
    obs_row = {"model": "observed", "sample": 0, **observed}
    nulls = sample_null_metrics(n, m, deg_sequence, n_samples=n_samples, base_seed=base_seed)
    summary = pd.concat(
        [pd.DataFrame([obs_row]), nulls],
        ignore_index=True,
    )
    out_csv = "null_model_comparison.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(summary)} rows)")

    numeric_cols = [
        "global_efficiency_giant",
        "avg_shortest_path_length_giant",
        "avg_clustering",
        "modularity_louvain",
        "n_communities_louvain",
    ]
    for model in ("gnm", "configuration"):
        sub = nulls[nulls["model"] == model]
        print(f"\n--- {model} (mean ± std over {n_samples} samples) ---")
        for col in numeric_cols:
            if col not in sub.columns:
                continue
            s = sub[col].dropna()
            if len(s) == 0:
                continue
            print(f"  {col}: {s.mean():.4f} ± {s.std():.4f}")
    print("\n--- observed ---")
    for col in numeric_cols:
        if col in obs_row:
            print(f"  {col}: {obs_row[col]}")


if __name__ == "__main__":
    main()
