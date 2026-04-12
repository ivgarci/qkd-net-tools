# Para analizar qué sucede en la red si se eliminan nodos de forma aleatoria (13%). Cuando tiene más de un componente, la red no permite la conexión de todos los nodos

import random

import networkx as nx
import pandas as pd

from grafo_io import DEFAULT_NAMED_ADJACENCY, load_named_adjacency

# Document this seed in the thesis when reporting confidence intervals / reproducibility.
RNG_SEED = 42
random.seed(RNG_SEED)

G = load_named_adjacency(DEFAULT_NAMED_ADJACENCY)

# Function to simulate random failures
def random_failure_simulation(G, removal_fraction=0.13):
    G_copy = G.copy()
    num_nodes_to_remove = int(removal_fraction * G.number_of_nodes())
    nodes_to_remove = random.sample(list(G.nodes()), num_nodes_to_remove)
    G_copy.remove_nodes_from(nodes_to_remove)
    largest_cc = max(nx.connected_components(G_copy), key=len)
    largest_cc_size = len(largest_cc)
    num_components = nx.number_connected_components(G_copy)
    diameter = nx.diameter(G_copy.subgraph(largest_cc))
    return largest_cc_size, num_components, diameter


num_simulations = 300

# Data storage
random_failure_data = []

# Run simulations
for _ in range(num_simulations):
    random_failure_results = random_failure_simulation(G, removal_fraction=0.13)
    random_failure_data.append(random_failure_results)

# Convert results to DataFrame
random_failure_df = pd.DataFrame(random_failure_data, columns=['Largest Connected Component Size', 'Number of Components', 'Diameter of Largest Component'])

# Save results to CSV
random_failure_df.to_csv("random_failure_results.csv", index=False)

# Print summaries
print(f"RNG_SEED = {RNG_SEED} (set in script for reproducibility)")
print(f"Adjacency file: {DEFAULT_NAMED_ADJACENCY}")
print("Random Failure Simulation Results Summary:")
print(random_failure_df.describe())


