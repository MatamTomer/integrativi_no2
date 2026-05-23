import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def load_activity_conditioned_data(path: Path) -> Dict:
    """Load the activity-conditioned pairwise vectors JSON file."""
    with path.open("r") as f:
        data = json.load(f)
    required_keys = {"activity_conditioned_vectors", "num_neurons", "neuron_ids"}
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"Missing keys in activity-conditioned file: {missing}")
    return data


def create_single_graph(
    i: int,
    j: int,
    vector: List,
    neuron_ids: List[int],
    output_dir: Path,
) -> Path:
    """
    Create a single graph for pair (i, j) showing P(i|j) vs network activity.

    Returns:
        Path to the saved PNG file
    """
    # Convert None to NaN for plotting
    y_values = [v if v is not None else np.nan for v in vector]
    x_values = list(range(len(y_values)))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot line with markers
    ax.plot(x_values, y_values, marker='o', markersize=4, linewidth=1.5, color='steelblue')

    # Labels and title
    neuron_i_id = neuron_ids[i]
    neuron_j_id = neuron_ids[j]
    ax.set_xlabel('Network Activity (number of active neurons)', fontsize=11)
    ax.set_ylabel(f'P(neuron {neuron_i_id} fires | neuron {neuron_j_id} fires)', fontsize=11)
    ax.set_title(f'Conditional Probability: P({neuron_i_id}|{neuron_j_id}) vs Network Activity', fontsize=12, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set y-axis limits
    ax.set_ylim(-0.05, 1.05)

    # Set x-axis to show all activity levels
    ax.set_xlim(-0.5, len(y_values) - 0.5)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f"pair_{neuron_i_id}_{neuron_j_id}.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    return output_path


def create_compact_page(
    j: int,
    neuron_ids: List[int],
    vectors_for_j: List[List],
    page_num: int,
    total_pages: int,
) -> plt.Figure:
    """
    Create a compact page with 29 subplots showing all P(i|j) for a given j.

    Args:
        j: Index of the "given" neuron
        neuron_ids: List of neuron IDs
        vectors_for_j: List of 29 vectors, one for each i
        page_num: Current page number (1-indexed)
        total_pages: Total number of pages

    Returns:
        Matplotlib figure object
    """
    num_neurons = len(neuron_ids)
    neuron_j_id = neuron_ids[j]

    # Create a grid layout: 5 rows x 6 columns = 30 slots (29 used)
    fig = plt.figure(figsize=(17, 11))
    fig.suptitle(
        f'Page {page_num}/{total_pages}: All P(i | neuron {neuron_j_id}) vs Network Activity',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    rows, cols = 5, 6

    for i in range(num_neurons):
        vector = vectors_for_j[i]
        neuron_i_id = neuron_ids[i]

        # Convert None to NaN
        y_values = [v if v is not None else np.nan for v in vector]
        x_values = list(range(len(y_values)))

        # Create subplot
        ax = fig.add_subplot(rows, cols, i + 1)

        # Plot
        ax.plot(x_values, y_values, marker='o', markersize=2, linewidth=1, color='steelblue')

        # Title
        ax.set_title(f'P({neuron_i_id}|{neuron_j_id})', fontsize=9, fontweight='bold')

        # Labels (only on edges to save space)
        if i >= (rows - 1) * cols:  # Bottom row
            ax.set_xlabel('Activity', fontsize=7)
        if i % cols == 0:  # Left column
            ax.set_ylabel('P(i|j)', fontsize=7)

        # Tick labels smaller
        ax.tick_params(axis='both', which='major', labelsize=6)

        # Grid
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

        # Y-axis limits
        ax.set_ylim(-0.05, 1.05)

        # X-axis limits
        ax.set_xlim(-0.5, len(y_values) - 0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize pairwise conditional probabilities as a function of network activity."
    )
    parser.add_argument(
        "--input",
        default="pairwise_activity_by_network_state.json",
        help="Path to activity-conditioned vectors JSON (default: pairwise_activity_by_network_state.json)"
    )
    parser.add_argument(
        "--output-dir",
        default="pairwise_graphs",
        help="Directory to save individual PNG files (default: pairwise_graphs)"
    )
    parser.add_argument(
        "--pdf-output",
        default="pairwise_by_network_activity.pdf",
        help="Output PDF file with all graphs (default: pairwise_by_network_activity.pdf)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    pdf_path = Path(args.pdf_output)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {input_path}")
    data = load_activity_conditioned_data(input_path)

    activity_vectors = data["activity_conditioned_vectors"]
    neuron_ids = data["neuron_ids"]
    num_neurons = data["num_neurons"]

    print(f"Generating {num_neurons}x{num_neurons} = {num_neurons**2} individual graphs...")

    # Generate all individual graphs
    graph_paths = []
    for i in range(num_neurons):
        for j in range(num_neurons):
            vector = activity_vectors[i][j]
            graph_path = create_single_graph(i, j, vector, neuron_ids, output_dir)
            graph_paths.append(graph_path)

            total_graphs = num_neurons * num_neurons
            current = i * num_neurons + j + 1
            print(f"  Generated graph {current}/{total_graphs}", end='\r')

    print(f"\n\nAll individual graphs saved to: {output_dir}/")

    # Create PDF with compact pages
    print(f"\nCreating compact PDF: {pdf_path}")
    with PdfPages(pdf_path) as pdf:
        for j in range(num_neurons):
            print(f"  Creating page {j + 1}/{num_neurons} (neuron {neuron_ids[j]})", end='\r')

            # Collect all vectors for this j (all i values)
            vectors_for_j = [activity_vectors[i][j] for i in range(num_neurons)]

            # Create compact page
            fig = create_compact_page(j, neuron_ids, vectors_for_j, j + 1, num_neurons)

            # Save to PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"\n\nPDF created successfully: {pdf_path}")
    print(f"Total individual graphs: {len(graph_paths)}")
    print(f"PDF pages: {num_neurons} (one per 'given' neuron)")


if __name__ == "__main__":
    main()
