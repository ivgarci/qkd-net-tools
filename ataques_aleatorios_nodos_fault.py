# Para analizar qué sucede en la red si se eliminan nodos de forma aleatoria (13%). Cuando tiene más de un componente, la red no permite la conexión de todos los nodos

import pandas as pd
import networkx as nx
import random
import matplotlib.pyplot as plt

# Load the adjacency matrix from the uploaded CSV file
file_path = 'AdjacencyMatrixNamed45.csv'
adjacency_matrix = pd.read_csv(file_path, index_col=0)

# Convert the adjacency matrix to a NetworkX graph
G = nx.from_pandas_adjacency(adjacency_matrix)

# Load the node coordinates from the uploaded CSV file
coordinates_file_path = 'cyl_1000.csv'
coordinates_df = pd.read_csv(coordinates_file_path, delimiter=';')

# Create a dictionary of node positions using the coordinates
node_positions = {row['Población']: (row['Longitud'], row['Latitud']) for idx, row in coordinates_df.iterrows()}

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
random_failure_df.to_csv('random_failure_results.csv', index=False)

# Print summaries
print("Random Failure Simulation Results Summary:")
print(random_failure_df.describe())


