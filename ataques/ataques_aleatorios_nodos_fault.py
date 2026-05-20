"""
Simulación de fallos aleatorios de nodos — R=300 trials, p=13 %.
Cuando la red queda con más de un componente, no todos los nodos pueden
comunicarse (requisito crítico para redes QKD de relé de confianza).
"""

import os
import random
import numpy as np
import pandas as pd
import networkx as nx

random.seed(42)
np.random.seed(42)

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_CYL = os.path.join(BASE, '..', 'datos', 'cyl')

G = nx.from_pandas_adjacency(
    pd.read_csv(os.path.join(DATA_CYL, 'AdjacencyMatrixNamed45.csv'), index_col=0)
)


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
random_failure_data = []

for _ in range(num_simulations):
    random_failure_data.append(random_failure_simulation(G, removal_fraction=0.13))

random_failure_df = pd.DataFrame(
    random_failure_data,
    columns=['Largest Connected Component Size', 'Number of Components',
             'Diameter of Largest Component']
)

out_csv = os.path.join(DATA_CYL, 'random_failure_results.csv')
random_failure_df.to_csv(out_csv, index=False)

print("Random Failure Simulation Results Summary:")
print(random_failure_df.describe())
print(f"\nGuardado: {out_csv}")
