import pandas as pd
import networkx as nx
from networkx.algorithms.community import girvan_newman
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

# Run the Girvan-Newman algorithm for more levels
communities_generator = girvan_newman(G)
k = 8  # Specify the number of communities you want to create
limited = [next(communities_generator) for _ in range(k - 1)]
top_communities = sorted(map(sorted, next(communities_generator)))

# Create a color map for the communities
color_map = []
colors = ['skyblue', 'lightgreen', 'coral', 'gold', 'lightpink', 'lightgrey', 'lightblue', 'orange', 'purple', 'brown']
for node in G:
    for idx, community in enumerate(top_communities):
        if node in community:
            color_map.append(colors[idx % len(colors)])

# Plot the network using the provided coordinates and color-coded by community
plt.figure(figsize=(12, 12))
nx.draw(G, pos=node_positions, with_labels=True, node_size=500, node_color=color_map, font_size=8, font_weight='bold', edge_color='gray')
plt.title("Network Visualization with Multiple Communities")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Guardar el gráfico en diferentes formatos
plt.savefig("girvan_newman_cyl_9.png", format='png', dpi=300)  # Guardar como PNG
plt.savefig("girvan_newman_cyl_9.pdf", format='pdf')  # Guardar como PDF
plt.savefig("girvan_newman_cyl_9.svg", format='svg')  # Guardar como SVG

plt.show()

# Convert the communities to a DataFrame for better display
communities_df = pd.DataFrame({"Community": range(len(top_communities)), "Nodes": top_communities})

# Display the DataFrame in a readable format (for example, print to console)
print(communities_df)
