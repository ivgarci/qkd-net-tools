import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the adjacency matrix
adjacency_matrix_path = 'adjacency_matrix.80.csv'
adj_matrix = pd.read_csv(adjacency_matrix_path, header=None)

# Create a graph from the adjacency matrix
G = nx.from_pandas_adjacency(adj_matrix)

# Node Connectivity Analysis
# Generate a basic plot of the network
plt.figure(figsize=(10, 8))
nx.draw_networkx(G, with_labels=True, node_color='skyblue', edge_color='gray')
plt.title('Node Connectivity Visualization')
plt.axis('off')  # Turn off the axis
plt.tight_layout()

plt.savefig('node_connectivity_visualization.png')
plt.show()


