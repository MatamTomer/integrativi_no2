import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

print("Loading D_ij matrix...")
with open('dij_matrix.json', 'r') as f:
    data = json.load(f)

D_ij = np.array(data['D_ij_matrix'])
neuron_ids = data['neuron_ids']
N = len(neuron_ids)

print(f"Number of neurons: {N}")

# Create directed graph with threshold D_ij < -10
threshold = -10
print(f"\nCreating directed graph with threshold D_ij < {threshold}")

G = nx.DiGraph()

# Add all neurons as nodes
for neuron_id in neuron_ids:
    G.add_node(neuron_id)

# Add edges where D_ij < -10 (strong inhibition)
edge_weights = {}
for i in range(N):
    for j in range(N):
        if i == j:
            continue

        d_val = D_ij[i][j]
        if not np.isnan(d_val) and d_val < threshold:
            # i inhibits j
            G.add_edge(neuron_ids[i], neuron_ids[j])
            edge_weights[(neuron_ids[i], neuron_ids[j])] = d_val

print(f"\nGraph statistics:")
print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")
print(f"  Density: {nx.density(G):.4f}")

# Analyze network structure
in_degrees = dict(G.in_degree())
out_degrees = dict(G.out_degree())

# Find most inhibited neurons (highest in-degree)
sorted_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
print(f"\nTop 5 most inhibited neurons (highest in-degree):")
for neuron_id, degree in sorted_in[:5]:
    print(f"  Neuron {neuron_id}: {degree} incoming inhibitory connections")

# Find strongest inhibitors (highest out-degree)
sorted_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
print(f"\nTop 5 strongest inhibitors (highest out-degree):")
for neuron_id, degree in sorted_out[:5]:
    print(f"  Neuron {neuron_id}: {degree} outgoing inhibitory connections")

# Find strongest inhibitory connections
print(f"\nTop 10 strongest inhibitory connections (most negative D_ij):")
sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1])
for (i, j), weight in sorted_edges[:10]:
    print(f"  Neuron {i} -> {j}: D_ij = {weight:.3f}")

# Create visualization
print("\nCreating network visualization...")
fig, ax = plt.subplots(figsize=(16, 16))

# Use spring layout for better visualization
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# Node sizes based on total degree (in + out)
total_degrees = [in_degrees[n] + out_degrees[n] for n in G.nodes()]
node_sizes = [500 + 100 * deg for deg in total_degrees]

# Node colors based on net inhibition (in-degree - out-degree)
net_inhibition = [in_degrees[n] - out_degrees[n] for n in G.nodes()]
node_colors = net_inhibition

# Draw nodes
nodes = nx.draw_networkx_nodes(G, pos,
                                node_size=node_sizes,
                                node_color=node_colors,
                                cmap='RdYlBu',  # Red = inhibited, Blue = inhibitor
                                vmin=min(net_inhibition),
                                vmax=max(net_inhibition),
                                edgecolors='black',
                                linewidths=2,
                                ax=ax)

# Draw edges with varying thickness based on strength
edge_strengths = [abs(edge_weights[(u, v)]) for (u, v) in G.edges()]
max_strength = max(edge_strengths)
edge_widths = [1 + 3 * (strength / max_strength) for strength in edge_strengths]

nx.draw_networkx_edges(G, pos,
                        width=edge_widths,
                        edge_color='darkred',
                        alpha=0.5,
                        arrowsize=15,
                        arrowstyle='->',
                        connectionstyle='arc3,rad=0.1',
                        ax=ax)

# Draw labels
nx.draw_networkx_labels(G, pos,
                        font_size=10,
                        font_weight='bold',
                        font_color='white',
                        ax=ax)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='RdYlBu',
                           norm=plt.Normalize(vmin=min(net_inhibition),
                                             vmax=max(net_inhibition)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Net Inhibition (In-degree - Out-degree)', fontsize=12)

ax.set_title(f'Inhibitory Network Graph (D_ij < {threshold})\n'
             f'{G.number_of_nodes()} neurons, {G.number_of_edges()} inhibitory connections\n'
             f'Node size: total connectivity, Color: Red = inhibited, Blue = inhibitor',
             fontsize=16, fontweight='bold')

ax.axis('off')
plt.tight_layout()
plt.savefig('inhibitory_network_graph.png', dpi=150, bbox_inches='tight')
print("Saved network graph to: inhibitory_network_graph.png")
plt.close()

# Save network statistics to JSON
network_stats = {
    'threshold': threshold,
    'n_nodes': G.number_of_nodes(),
    'n_edges': G.number_of_edges(),
    'density': nx.density(G),
    'in_degrees': in_degrees,
    'out_degrees': out_degrees,
    'net_inhibition': {neuron_ids[i]: net_inhibition[i] for i in range(N)},
    'top_inhibited': [(int(n), int(d)) for n, d in sorted_in[:10]],
    'top_inhibitors': [(int(n), int(d)) for n, d in sorted_out[:10]],
    'strongest_connections': [{'from': int(i), 'to': int(j), 'D_ij': float(w)}
                              for (i, j), w in sorted_edges[:20]]
}

with open('inhibitory_network_stats.json', 'w') as f:
    json.dump(network_stats, f, indent=2)

print("Saved network statistics to: inhibitory_network_stats.json")
print("\nDone!")
