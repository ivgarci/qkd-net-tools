# Para analizar qué sucede en la red si se eliminan nodos con mayor grado de centralidad. Cuando tiene más de un componente, la red no permite la conexión de todos los nodos
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

# Function to simulate targeted attacks
def targeted_attack_simulation_incremental(G, centrality_metric='degree'):
    results = []
    
    if centrality_metric == 'degree':
        centrality = nx.degree_centrality(G)
    elif centrality_metric == 'closeness':
        centrality = nx.closeness_centrality(G)
    elif centrality_metric == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    
    # Sort nodes by centrality
    sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)
    
    for removal_fraction in range(0, 50, 1):  # Increment by 1 from 0 to 50
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

# Run incremental attack simulations
attack_results = targeted_attack_simulation_incremental(G, centrality_metric='degree')

# Convert results to DataFrame
attack_df = pd.DataFrame(attack_results, columns=['Removal Fraction (%)', 'Largest Connected Component Size', 'Number of Components', 'Diameter of Largest Component'])

# Save results to CSV
attack_df.to_csv('incremental_targeted_attack_results.csv', index=False)

# Print summary of results
print("Incremental Targeted Attack Simulation Results:")
print(attack_df)

# Determine the percentage at which the network becomes disconnected
disconnection_threshold = attack_df[attack_df['Number of Components'] > 1]['Removal Fraction (%)'].min()

print(f"\nThe network becomes disconnected at {disconnection_threshold}% of nodes removed.")
