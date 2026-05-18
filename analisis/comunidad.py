import community as community_louvain

# Use the Louvain method to find the best partition
partition = community_louvain.best_partition(G)

# Visualization of the community structure
plt.figure(figsize=(12, 10))
# Generate a color palette with enough colors for each community
cmap = plt.cm.get_cmap('hsv', max(partition.values()) + 1)
# Draw the nodes with colors according to their partition
nx.draw_networkx(G, node_color=[cmap(partition[node]) for node in G], node_size=50, with_labels=False, edge_color='lightgray')
plt.title('Community Structure in the Network')
plt.axis('off')
plt.tight_layout()

plt.savefig('community_structure_visualization.png')
plt.show()

# Count the number of communities detected
num_communities = len(set(partition.values()))
num_communities


