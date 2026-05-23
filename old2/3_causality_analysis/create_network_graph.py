import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def load_sij_matrix():
    """Load the S_ij matrix."""
    with open('significance_analysis/sij_matrix.json', 'r') as f:
        data = json.load(f)

    S_ij = np.array(data['S_ij'])
    neuron_ids = data['neuron_ids']

    return S_ij, neuron_ids


def create_directed_graph(S_ij, neuron_ids, threshold=50):
    """
    Create a directed graph where edge i->j exists if S_ij > threshold.

    Args:
        S_ij: 29x29 matrix of S values
        neuron_ids: List of neuron IDs
        threshold: Minimum S_ij value for an edge

    Returns:
        G: NetworkX directed graph
        edge_weights: Dictionary of edge weights
    """
    N = len(neuron_ids)
    G = nx.DiGraph()

    # Add all nodes
    for neuron_id in neuron_ids:
        G.add_node(neuron_id)

    # Add edges where S_ij > threshold
    edge_weights = {}
    for i in range(N):
        for j in range(N):
            if i == j:  # Skip self-loops
                continue

            s_val = S_ij[i][j]

            # Only add edge if S_ij > threshold and not NaN
            if not np.isnan(s_val) and s_val > threshold:
                G.add_edge(neuron_ids[i], neuron_ids[j])
                edge_weights[(neuron_ids[i], neuron_ids[j])] = s_val

    return G, edge_weights


def visualize_graph(G, edge_weights, neuron_ids, threshold, output_path):
    """Create a visualization of the directed graph."""

    # Calculate node sizes based on in-degree and out-degree
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    # Node size proportional to total degree
    node_sizes = [300 + 100 * (in_degrees.get(n, 0) + out_degrees.get(n, 0)) for n in neuron_ids]

    # Node color based on out-degree (how many neurons this one influences)
    node_colors = [out_degrees.get(n, 0) for n in neuron_ids]

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14))

    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Draw nodes
    nodes = nx.draw_networkx_nodes(G, pos,
                                   nodelist=neuron_ids,
                                   node_size=node_sizes,
                                   node_color=node_colors,
                                   cmap='YlOrRd',
                                   alpha=0.9,
                                   ax=ax)

    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos,
                          edge_color='gray',
                          alpha=0.5,
                          arrows=True,
                          arrowsize=15,
                          arrowstyle='->',
                          width=1.5,
                          connectionstyle='arc3,rad=0.1',
                          ax=ax)

    # Draw labels
    nx.draw_networkx_labels(G, pos,
                           font_size=8,
                           font_weight='bold',
                           ax=ax)

    # Add colorbar for out-degree
    sm = plt.cm.ScalarMappable(cmap='YlOrRd',
                               norm=plt.Normalize(vmin=0,
                                                 vmax=max(node_colors) if node_colors else 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Out-degree (# neurons influenced)', fontsize=10)

    ax.set_title(f'Directed Network Graph (S_ij > {threshold})\n' +
                f'Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}',
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved graph visualization to: {output_path}")


def save_graph_statistics(G, edge_weights, neuron_ids, threshold, output_path):
    """Save graph statistics and adjacency information."""

    # Calculate statistics
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    # Find nodes sorted by out-degree (most influential)
    sorted_by_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)

    # Find nodes sorted by in-degree (most influenced)
    sorted_by_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)

    # Get edge list with weights
    edge_list = []
    for (source, target), weight in edge_weights.items():
        edge_list.append({
            'source': int(source),
            'target': int(target),
            'S_ij': float(weight)
        })

    # Sort edges by weight
    edge_list.sort(key=lambda x: x['S_ij'], reverse=True)

    # Create adjacency matrix (as list of lists for JSON)
    N = len(neuron_ids)
    adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
    for i, source_id in enumerate(neuron_ids):
        for j, target_id in enumerate(neuron_ids):
            if G.has_edge(source_id, target_id):
                adj_matrix[i][j] = 1

    data = {
        'threshold': threshold,
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'neuron_ids': neuron_ids,
        'in_degrees': {int(k): v for k, v in in_degrees.items()},
        'out_degrees': {int(k): v for k, v in out_degrees.items()},
        'top_influencers': [(int(k), v) for k, v in sorted_by_out[:10]],
        'top_influenced': [(int(k), v) for k, v in sorted_by_in[:10]],
        'edges': edge_list,
        'adjacency_matrix': adj_matrix,
        'description': f'Directed graph where edge i->j exists if S_ij > {threshold}'
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved graph data to: {output_path}")


def main():
    output_dir = Path('significance_analysis')
    graph_viz_path = output_dir / 'network_graph.png'
    graph_data_path = output_dir / 'network_graph_data.json'

    threshold = 50

    print("="*70)
    print("CREATING DIRECTED NETWORK GRAPH")
    print("="*70)
    print(f"Threshold: S_ij > {threshold}")
    print()

    # Load S_ij matrix
    print("Loading S_ij matrix...")
    S_ij, neuron_ids = load_sij_matrix()
    N = len(neuron_ids)
    print(f"  {N} neurons loaded")
    print()

    # Create graph
    print("Creating directed graph...")
    G, edge_weights = create_directed_graph(S_ij, neuron_ids, threshold)

    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.4f}")
    print()

    # Calculate statistics
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    # Top influencers (highest out-degree)
    sorted_by_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
    print("Top 5 influencers (by out-degree):")
    for rank, (neuron_id, degree) in enumerate(sorted_by_out[:5], 1):
        print(f"  {rank}. Neuron {neuron_id}: influences {degree} neurons")
    print()

    # Top influenced (highest in-degree)
    sorted_by_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
    print("Top 5 most influenced (by in-degree):")
    for rank, (neuron_id, degree) in enumerate(sorted_by_in[:5], 1):
        print(f"  {rank}. Neuron {neuron_id}: influenced by {degree} neurons")
    print()

    # Strongest connections
    sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)
    print("Top 5 strongest connections:")
    for rank, ((source, target), weight) in enumerate(sorted_edges[:5], 1):
        print(f"  {rank}. Neuron {source} -> {target}: S_ij = {weight:.2f}")
    print()

    # Visualize
    print("Creating visualization...")
    visualize_graph(G, edge_weights, neuron_ids, threshold, graph_viz_path)

    # Save data
    print("Saving graph data...")
    save_graph_statistics(G, edge_weights, neuron_ids, threshold, graph_data_path)

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Created directed graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Threshold: S_ij > {threshold}")
    print(f"Output files:")
    print(f"  - Visualization: {graph_viz_path}")
    print(f"  - Data (JSON): {graph_data_path}")
    print()
    print("Interpretation:")
    print(f"  - Edge i->j exists if neuron j has strong early peak (1-5ms)")
    print(f"    when neuron i fires, with S_ij > {threshold}")
    print(f"  - Node size: proportional to total connections")
    print(f"  - Node color: proportional to out-degree (influence)")
    print("="*70)


if __name__ == "__main__":
    main()
