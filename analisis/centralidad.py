import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Define graph G here
#G = nx.karate_club_graph()  # Example graph, replace with correct

# Calculate centrality measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

# Convert centrality measures to DataFrame for easier handling
centrality_measures_df = pd.DataFrame({
	'Degree Centrality': degree_centrality,
	'Betweenness Centrality': betweenness_centrality,
	'Closeness Centrality': closeness_centrality
})

# Visualize the top nodes based on each centrality measure
top_n = 10  # Number of top nodes to display

# Sort and select top nodes for each centrality measure
top_degree = centrality_measures_df['Degree Centrality'].sort_values(ascending=False).head(top_n)
top_betweenness = centrality_measures_df['Betweenness Centrality'].sort_values(ascending=False).head(top_n)
top_closeness = centrality_measures_df['Closeness Centrality'].sort_values(ascending=False).head(top_n)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Degree Centrality
top_degree.plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Top Nodes by Degree Centrality')
axes[0].set_ylabel('Centrality Score')
axes[0].set_xlabel('Node')

# Betweenness Centrality
top_betweenness.plot(kind='bar', ax=axes[1], color='lightgreen')
axes[1].set_title('Top Nodes by Betweenness Centrality')
axes[1].set_xlabel('Node')

# Closeness Centrality
top_closeness.plot(kind='bar', ax=axes[2], color='salmon')
axes[2].set_title('Top Nodes by Closeness Centrality')
axes[2].set_xlabel('Node')

plt.tight_layout()
plt.savefig('centrality_measures_visualization.png')
plt.show()


