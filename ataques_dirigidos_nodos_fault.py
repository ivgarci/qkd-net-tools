# Para analizar qué sucede en la red si se eliminan nodos con mayor grado de centralidad. Cuando tiene más de un componente, la red no permite la conexión de todos los nodos
import networkx as nx
import pandas as pd

from grafo_io import DEFAULT_NAMED_ADJACENCY, load_named_adjacency

G = load_named_adjacency(DEFAULT_NAMED_ADJACENCY)


def targeted_attack_simulation_incremental(G, centrality_metric="degree"):
    results = []

    if centrality_metric == "degree":
        centrality = nx.degree_centrality(G)
    elif centrality_metric == "closeness":
        centrality = nx.closeness_centrality(G)
    elif centrality_metric == "betweenness":
        centrality = nx.betweenness_centrality(G)
    else:
        raise ValueError("centrality_metric must be degree, closeness, or betweenness")

    sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)

    for removal_fraction in range(0, 50, 1):
        G_copy = G.copy()
        num_nodes_to_remove = int((removal_fraction / 100) * G.number_of_nodes())
        nodes_to_remove = sorted_nodes[:num_nodes_to_remove]
        G_copy.remove_nodes_from(nodes_to_remove)

        if G_copy.number_of_nodes() == 0:
            largest_cc_size = 0
            diameter = 0
            num_components = 0
        elif nx.is_connected(G_copy):
            largest_cc_size = G_copy.number_of_nodes()
            diameter = nx.diameter(G_copy)
            num_components = 1
        else:
            largest_cc = max(nx.connected_components(G_copy), key=len)
            largest_cc_size = len(largest_cc)
            diameter = nx.diameter(G_copy.subgraph(largest_cc))
            num_components = nx.number_connected_components(G_copy)

        results.append((removal_fraction, largest_cc_size, num_components, diameter))

    return results


attack_results = targeted_attack_simulation_incremental(G, centrality_metric="degree")

attack_df = pd.DataFrame(
    attack_results,
    columns=[
        "Removal Fraction (%)",
        "Largest Connected Component Size",
        "Number of Components",
        "Diameter of Largest Component",
    ],
)

attack_df.to_csv("incremental_targeted_attack_results.csv", index=False)

print(f"Adjacency file: {DEFAULT_NAMED_ADJACENCY}")
print("Incremental Targeted Attack Simulation Results:")
print(attack_df)

disconnection_threshold = attack_df[attack_df["Number of Components"] > 1]["Removal Fraction (%)"].min()

print(f"\nThe network becomes disconnected at {disconnection_threshold}% of nodes removed.")
