import numpy as np
import networkx as nx
num_points = 100
pc = np.random.rand(num_points, 3)

G = nx.Graph()

for i, point in enumerate(pc):
    G.add_node(i, pos=(point[0], point[1], point[2]))

edge_threshold = 0.2
for i in range(num_points):
    for j in range(i+1, num_points):
        jarak = np.linalg.norm(pc[i] - pc[j])
        if jarak < edge_threshold:
            G.add_edge(i, j)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Extract node positions from the graph
node_positions = nx.get_node_attributes(G, 'pos')

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot nodes
for node, pos in node_positions.items():
    ax.scatter(pos[0], pos[1], pos[2], label=str(node))

# Plot edges
for edge in G.edges():
    node1, node2 = edge
    pos1 = node_positions[node1]
    pos2 = node_positions[node2]
    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2],])

# Customize the plot if needed
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('3D Graph from Point Cloud')

# Show the plot
plt.show()