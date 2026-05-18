import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the adjacency matrix from the uploaded CSV file
file_path = 'AdjacencyMatrixNamed45.csv'
adjacency_matrix = pd.read_csv(file_path, index_col=0)

# Convert the adjacency matrix to a NetworkX graph
G = nx.from_pandas_adjacency(adjacency_matrix)

# Calculate the degree of each node
degree_sequence = [d for n, d in G.degree()]

# Create a histogram of the degree distribution
plt.figure(figsize=(10, 6))
plt.hist(degree_sequence, bins=30, color='skyblue', edgecolor='black')
plt.title("Distribución del grado del grafo - 100 nodos CyL")
plt.xlabel("Grado")
plt.ylabel("Frecuencia")
plt.grid(True)


# Guardar el gráfico en diferentes formatos
plt.savefig("distribucion_grado_grafo.png", format='png', dpi=300)  # Guardar como PNG
plt.savefig("distribucion_grado_grafo.pdf", format='pdf')  # Guardar como PDF
plt.savefig("distribucion_grado_grafo.svg", format='svg')  # Guardar como SVG

# Mostrar por pantalla el gráfico
plt.show()
