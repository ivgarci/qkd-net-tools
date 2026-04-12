# SPDX-License-Identifier: GPL-3.0-only
"""
Advanced robustness: S(p) curves for random / static / adaptive node attacks,
random / static / adaptive edge attacks, and Schneider-style robustness R.

Schneider reference (conceptual): robustness as average normalized size of the
largest connected component along an attack sequence — see e.g. Schneider et al.,
"Mitigation of malicious attacks on networks" (2011).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from grafo_io import DEFAULT_NAMED_ADJACENCY, load_named_adjacency


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def relative_lcc_size(H: nx.Graph, n_ref: int) -> float:
    if n_ref <= 0 or H.number_of_nodes() == 0:
        return 0.0
    lcc = len(max(nx.connected_components(H), key=len))
    return lcc / n_ref


def curve_random_nodes(
    G: nx.Graph, fractions: np.ndarray, n_trials: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n_ref = G.number_of_nodes()
    nodes = list(G.nodes())
    means = []
    stds = []
    for f in fractions:
        k = int(round(float(f) * n_ref))
        vals = []
        for _ in range(n_trials):
            H = G.copy()
            if k > 0:
                remove = rng.choice(nodes, size=k, replace=False)
                H.remove_nodes_from(remove)
            vals.append(relative_lcc_size(H, n_ref))
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1)) if n_trials > 1 else 0.0)
    return np.array(means), np.array(stds)


def curve_static_nodes(G: nx.Graph, fractions: np.ndarray, metric: str) -> np.ndarray:
    n_ref = G.number_of_nodes()
    if metric == "degree":
        cent = nx.degree_centrality(G)
    elif metric == "closeness":
        cent = nx.closeness_centrality(G)
    else:
        cent = nx.betweenness_centrality(G)
    order = sorted(cent, key=cent.get, reverse=True)
    out = []
    for f in fractions:
        k = int(round(float(f) * n_ref))
        H = G.copy()
        H.remove_nodes_from(order[:k])
        out.append(relative_lcc_size(H, n_ref))
    return np.array(out)


def curve_adaptive_nodes(G: nx.Graph, fractions: np.ndarray, metric: str) -> np.ndarray:
    n_ref = G.number_of_nodes()
    out = []
    for f in fractions:
        k = int(round(float(f) * n_ref))
        H = G.copy()
        removed = 0
        while removed < k and H.number_of_nodes() > 0:
            if metric == "degree":
                c = nx.degree_centrality(H)
            elif metric == "closeness":
                c = nx.closeness_centrality(H)
            else:
                c = nx.betweenness_centrality(H)
            target = max(c, key=c.get)
            H.remove_node(target)
            removed += 1
        out.append(relative_lcc_size(H, n_ref))
    return np.array(out)


def curve_random_edges(
    G: nx.Graph, fractions: np.ndarray, n_trials: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n_ref = G.number_of_nodes()
    edges = list(G.edges())
    m = len(edges)
    means = []
    stds = []
    for f in fractions:
        k = min(int(round(float(f) * m)), m)
        vals = []
        for _ in range(n_trials):
            H = G.copy()
            if k > 0:
                idx = rng.choice(len(edges), size=k, replace=False)
                to_remove = [edges[int(i)] for i in np.atleast_1d(idx)]
                H.remove_edges_from(to_remove)
            vals.append(relative_lcc_size(H, n_ref))
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1)) if n_trials > 1 else 0.0)
    return np.array(means), np.array(stds)


def curve_static_edges(G: nx.Graph, fractions: np.ndarray) -> np.ndarray:
    n_ref = G.number_of_nodes()
    eb = nx.edge_betweenness_centrality(G)
    order = sorted(eb, key=eb.get, reverse=True)
    m = len(order)
    out = []
    for f in fractions:
        k = min(int(round(float(f) * m)), m)
        H = G.copy()
        H.remove_edges_from(order[:k])
        out.append(relative_lcc_size(H, n_ref))
    return np.array(out)


def curve_adaptive_edges(G: nx.Graph, fractions: np.ndarray) -> np.ndarray:
    n_ref = G.number_of_nodes()
    m = G.number_of_edges()
    out = []
    for f in fractions:
        k = min(int(round(float(f) * m)), m)
        H = G.copy()
        removed = 0
        while removed < k and H.number_of_edges() > 0:
            eb = nx.edge_betweenness_centrality(H)
            edge = max(eb, key=eb.get)
            H.remove_edge(*edge)
            removed += 1
        out.append(relative_lcc_size(H, n_ref))
    return np.array(out)


def robustness_R_mean(S: np.ndarray) -> float:
    """Mean normalized LCC over the sampled removal fractions."""
    return float(np.mean(S))


def robustness_R_integral(S: np.ndarray, fractions: np.ndarray) -> float:
    """Trapezoidal integral of S(f) over the fraction grid (unnormalized)."""
    return _trapz(S, fractions)


def run_all(
    G: nx.Graph,
    max_node_fraction: float,
    max_edge_fraction: float,
    n_trials: int,
    seed: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    frac_nodes = np.linspace(0.0, max_node_fraction, 26)
    frac_edges = np.linspace(0.0, max_edge_fraction, 26)

    records = []
    r_rows = []

    # --- Nodes ---
    m_r, s_r = curve_random_nodes(G, frac_nodes, n_trials=n_trials, rng=rng)
    for i, f in enumerate(frac_nodes):
        records.append(
            {
                "attack_family": "nodes",
                "scenario": "random",
                "fraction": float(f),
                "mean_S": m_r[i],
                "std_S": s_r[i],
            }
        )
    r_rows.append(
        {
            "attack_family": "nodes",
            "scenario": "random",
            "R_mean": robustness_R_mean(m_r),
            "R_integral": robustness_R_integral(m_r, frac_nodes),
        }
    )

    for metric in ("degree", "closeness", "betweenness"):
        tag = f"static_{metric}"
        S = curve_static_nodes(G, frac_nodes, metric=metric)
        for i, f in enumerate(frac_nodes):
            records.append(
                {
                    "attack_family": "nodes",
                    "scenario": tag,
                    "fraction": float(f),
                    "mean_S": float(S[i]),
                    "std_S": 0.0,
                }
            )
        r_rows.append(
            {
                "attack_family": "nodes",
                "scenario": tag,
                "R_mean": robustness_R_mean(S),
                "R_integral": robustness_R_integral(S, frac_nodes),
            }
        )

    for metric in ("degree", "closeness", "betweenness"):
        tag = f"adaptive_{metric}"
        S = curve_adaptive_nodes(G, frac_nodes, metric=metric)
        for i, f in enumerate(frac_nodes):
            records.append(
                {
                    "attack_family": "nodes",
                    "scenario": tag,
                    "fraction": float(f),
                    "mean_S": float(S[i]),
                    "std_S": 0.0,
                }
            )
        r_rows.append(
            {
                "attack_family": "nodes",
                "scenario": tag,
                "R_mean": robustness_R_mean(S),
                "R_integral": robustness_R_integral(S, frac_nodes),
            }
        )

    # --- Edges ---
    m_e, s_e = curve_random_edges(G, frac_edges, n_trials=n_trials, rng=rng)
    for i, f in enumerate(frac_edges):
        records.append(
            {
                "attack_family": "edges",
                "scenario": "random",
                "fraction": float(f),
                "mean_S": m_e[i],
                "std_S": s_e[i],
            }
        )
    r_rows.append(
        {
            "attack_family": "edges",
            "scenario": "random",
            "R_mean": robustness_R_mean(m_e),
            "R_integral": robustness_R_integral(m_e, frac_edges),
        }
    )

    S_se = curve_static_edges(G, frac_edges)
    for i, f in enumerate(frac_edges):
        records.append(
            {
                "attack_family": "edges",
                "scenario": "static_edge_betweenness",
                "fraction": float(f),
                "mean_S": float(S_se[i]),
                "std_S": 0.0,
            }
        )
    r_rows.append(
        {
            "attack_family": "edges",
            "scenario": "static_edge_betweenness",
            "R_mean": robustness_R_mean(S_se),
            "R_integral": robustness_R_integral(S_se, frac_edges),
        }
    )

    S_ae = curve_adaptive_edges(G, frac_edges)
    for i, f in enumerate(frac_edges):
        records.append(
            {
                "attack_family": "edges",
                "scenario": "adaptive_edge_betweenness",
                "fraction": float(f),
                "mean_S": float(S_ae[i]),
                "std_S": 0.0,
            }
        )
    r_rows.append(
        {
            "attack_family": "edges",
            "scenario": "adaptive_edge_betweenness",
            "R_mean": robustness_R_mean(S_ae),
            "R_integral": robustness_R_integral(S_ae, frac_edges),
        }
    )

    curves_df = pd.DataFrame.from_records(records)
    r_df = pd.DataFrame.from_records(r_rows)
    curves_df.to_csv(out_dir / "robustness_curves.csv", index=False)
    r_df.to_csv(out_dir / "robustness_R_metrics.csv", index=False)

    # Plots
    def plot_family(family: str, fractions_key: np.ndarray, fname: str) -> None:
        sub = curves_df[curves_df["attack_family"] == family]
        plt.figure(figsize=(10, 6))
        for scenario in sub["scenario"].unique():
            row = sub[sub["scenario"] == scenario]
            plt.plot(
                row["fraction"],
                row["mean_S"],
                label=scenario,
            )
            if row["std_S"].max() > 0:
                plt.fill_between(
                    row["fraction"],
                    row["mean_S"] - row["std_S"],
                    row["mean_S"] + row["std_S"],
                    alpha=0.15,
                )
        plt.xlabel("Removed fraction (nodes or edges)")
        plt.ylabel("Relative largest connected component |LCC| / N₀")
        plt.title(f"Robustness curves — {family}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=200)
        plt.close()

    plot_family("nodes", frac_nodes, "robustness_nodes.png")
    plot_family("edges", frac_edges, "robustness_edges.png")

    print(f"Wrote CSV and figures under {out_dir.resolve()}")
    print(r_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Advanced robustness simulations.")
    p.add_argument("--adjacency", default=DEFAULT_NAMED_ADJACENCY, help="Named adjacency CSV")
    p.add_argument("--out-dir", default="robustez_output", help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (document for the thesis)")
    p.add_argument("--trials", type=int, default=80, help="Monte Carlo trials for random attacks")
    p.add_argument("--max-node-frac", type=float, default=0.5, help="Max node removal fraction")
    p.add_argument("--max-edge-frac", type=float, default=0.6, help="Max edge removal fraction")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    G = load_named_adjacency(args.adjacency)
    run_all(
        G,
        max_node_fraction=args.max_node_frac,
        max_edge_fraction=args.max_edge_frac,
        n_trials=args.trials,
        seed=args.seed,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
