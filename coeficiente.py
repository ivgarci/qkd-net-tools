# Calculate the average clustering coefficient
avg_clustering_coefficient = nx.average_clustering(G)

# Calculate the clustering coefficient for each node
node_clustering = nx.clustering(G)

# Visualization of Clustering Coefficient Distribution
plt.figure(figsize=(10, 6))
plt.hist(list(node_clustering.values()), bins=np.linspace(0, 1, 20), color='skyblue', edgecolor='black')
plt.title('Clustering Coefficient Distribution')
plt.xlabel('Clustering Coefficient')
plt.ylabel('Number of Nodes')
plt.grid(axis='y', alpha=0.75)

avg_clustering_coefficient_text = f"Average Clustering Coefficient: {avg_clustering_coefficient:.2f}"
plt.figtext(0.5, -0.05, avg_clustering_coefficient_text, ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

plt.savefig('clustering_coefficient_distribution.png')
plt.show(), avg_clustering_coefficient


