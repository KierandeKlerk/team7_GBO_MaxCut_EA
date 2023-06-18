import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations

# Create the graph
G = nx.Graph()

# Add nodes with color attributes
G.add_node(1, color='pink')
G.add_node(2, color='gray')
G.add_node(3, color='gray')
G.add_node(4, color='gray')

# Add edges with increasing weights
weights = [1, 2, 5, 5, 3, 4]
for i, (u, v) in enumerate(combinations(G.nodes, 2)):
    G.add_edge(u, v, weight=weights[i])

# Define the 2D layout
pos = {
    1: [0, 0],
    2: [1, 0],
    3: [0, 1],
    4: [1, 1]
}

# Draw the graph
fig = plt.figure()
ax = fig.add_subplot(111)

# nodes
nx.draw_networkx_nodes(G, pos, node_color=[node[1]['color'] for node in G.nodes(data=True)], ax=ax, node_size=3000)

# Get the cut edges depending on the color of the nodes
cut_edges = [(u, v) for (u, v, d) in G.edges(data=True) if G.nodes[u]['color'] != G.nodes[v]['color']]
remaining_edges = [(u, v) for (u, v, d) in G.edges(data=True) if G.nodes[u]['color'] == G.nodes[v]['color']]
# Make all the edges connected to node 1 purple
purple_edges = [(u, v) for (u, v, d) in G.edges(data=True) if u == 1 or v == 1]

# edges
nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color='black', width=3, ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=remaining_edges, edge_color='black', width=3, ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=purple_edges, edge_color='#FF69B4', width=3, ax=ax)


# edge labels
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=30)

# node labels, but bold

nx.draw_networkx_labels(G, pos, font_size=30, font_family='sans-serif',font_weight='bold', ax=ax)

ax.set_axis_off()
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
plt.show()
