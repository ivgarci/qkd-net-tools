import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import f_oneway

# Cargar la matriz de adyacencia
file_path = 'AdjacencyMatrixNamed45.csv'  
adjacency_matrix = pd.read_csv(file_path, index_col=0)

# Crear el grafo
G = nx.from_pandas_adjacency(adjacency_matrix)

# Calcular número de aristas
number_of_edges = G.number_of_edges()

# Medidas de centralidad
degree_centrality = nx.degree_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# Ajustar parámetros para la centralidad de vector propio
try:
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
except nx.PowerIterationFailedConvergence:
    eigenvector_centrality = {node: None for node in G.nodes()}
    print("Eigenvector centrality did not converge within the given iterations")

# Crear DataFrame para medidas de centralidad
centralities_df = pd.DataFrame({
    'Degree Centrality': degree_centrality,
    'Closeness Centrality': closeness_centrality,
    'Betweenness Centrality': betweenness_centrality,
    'Eigenvector Centrality': eigenvector_centrality
})

# Calcular coeficiente de agrupamiento
clustering_coefficient = nx.clustering(G)

# Calcular el diámetro y la densidad de la red
diameter = nx.diameter(G) if nx.is_connected(G) else None
density = nx.density(G)

# Añadir el coeficiente de agrupamiento al DataFrame
centralities_df['Clustering Coefficient'] = centralities_df.index.map(clustering_coefficient)

# HJ-Biplot (utilizando PCA como aproximación)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(centralities_df.fillna(0))  # Rellenar posibles valores NaN

# Análisis de Clúster
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(pca_result)
centralities_df['Cluster'] = clusters

# ANOVA entre clústeres para la centralidad de grado 
anova_results = f_oneway(centralities_df[centralities_df['Cluster'] == 0]['Degree Centrality'],
                         centralities_df[centralities_df['Cluster'] == 1]['Degree Centrality'],
                         centralities_df[centralities_df['Cluster'] == 2]['Degree Centrality'])

# Visualización
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('HJ-Biplot Clusters')

# Guardar el gráfico en diferentes formatos
plt.savefig("hj_biplot_clusters.pdf", format='pdf')  # Guardar como PDF

# Mostrar por pantalla el gráfico
plt.show()

# Guardar los resultados en un archivo CSV
centralities_df.to_csv('Node_Specific_Network_Measures.csv', index=True)

# Imprimir resultados globales
print(f"Número de aristas de la red: {number_of_edges}")
print(f"Diámetro de la red: {diameter}")
print(f"Densidad de la red: {density}")
print("\nANOVA results:", anova_results)

# Imprimir resumen de medidas de centralidad
print("\nResumen de Medidas de Centralidad:")
print(centralities_df.describe())

