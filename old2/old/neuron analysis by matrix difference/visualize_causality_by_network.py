import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import UnivariateSpline


def load_causality_vector(path: Path) -> Dict:
    """Load a causality vector JSON file."""
    with path.open('r') as f:
        data = json.load(f)
    return data


def create_compact_page(
    j: int,
    neuron_ids: List[int],
    all_vectors: List[np.ndarray],
    page_num: int,
    total_pages: int,
) -> plt.Figure:
    """
    Create a compact page with 29 subplots showing all P(i|j) for a given j.

    Args:
        j: Index of the "given" neuron
        neuron_ids: List of neuron IDs
        all_vectors: List of 29 vectors, one for each i
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
        neuron_i_id = neuron_ids[i]
        vector = all_vectors[i]

        # Create x-axis (activity levels where we have data)
        x_values = []
        y_values = []
        for activity_level in range(len(vector)):
            if not np.isnan(vector[activity_level]):
                x_values.append(activity_level)
                y_values.append(vector[activity_level])

        x_values = np.array(x_values)
        y_values = np.array(y_values)

        # Create subplot
        ax = fig.add_subplot(rows, cols, i + 1)

        # Plot
        if len(x_values) > 0:
            # Plot original data points
            ax.plot(x_values, y_values, 'o', markersize=3, color='lightsteelblue', alpha=0.5, label='Data')

            # Apply spline interpolation if we have enough points
            if len(x_values) >= 4:  # Need at least 4 points for cubic spline
                try:
                    # Create spline with smoothing
                    # s parameter controls smoothing: 0=exact fit, higher=more smoothing
                    spline = UnivariateSpline(x_values, y_values, s=len(x_values)*0.01, k=3)

                    # Generate smooth curve
                    x_smooth = np.linspace(x_values.min(), x_values.max(), 200)
                    y_smooth = spline(x_smooth)

                    # Plot smooth curve
                    ax.plot(x_smooth, y_smooth, linewidth=1.5, color='steelblue', label='Smoothed')
                except:
                    # If spline fails, just plot the line
                    ax.plot(x_values, y_values, linewidth=1, color='steelblue')
            else:
                # Not enough points for spline, plot simple line
                ax.plot(x_values, y_values, linewidth=1, color='steelblue')

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

        # Auto-scale y-axis with margin
        if len(y_values) > 0:
            y_range = np.max(y_values) - np.min(y_values)
            if y_range > 0:
                y_margin = 0.1 * y_range
                ax.set_ylim(np.min(y_values) - y_margin, np.max(y_values) + y_margin)
            else:
                ax.set_ylim(np.min(y_values) - 0.1, np.max(y_values) + 0.1)

        # X-axis limits
        ax.set_xlim(-0.5, 29.5)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize causality probabilities as a function of network activity."
    )
    parser.add_argument(
        "--input-dir",
        default="causality_vectors",
        help="Directory containing causality vectors (default: causality_vectors)"
    )
    parser.add_argument(
        "--pdf-output",
        default="causality_by_network_activity.pdf",
        help="Output PDF file (default: causality_by_network_activity.pdf)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    pdf_path = Path(args.pdf_output)

    print(f"Loading causality vectors from: {input_dir}")

    # Find all causality vector files
    causality_files = sorted(input_dir.glob("causality_*.json"))

    if len(causality_files) == 0:
        print(f"Error: No causality vector files found in {input_dir}")
        return

    # Extract neuron IDs from filenames
    neuron_ids = set()
    for file in causality_files:
        # Parse filename: causality_i_j.json
        parts = file.stem.split('_')
        if len(parts) >= 3:
            neuron_ids.add(int(parts[1]))
            neuron_ids.add(int(parts[2]))

    neuron_ids = sorted(list(neuron_ids))
    num_neurons = len(neuron_ids)

    print(f"Found {num_neurons} neurons: {neuron_ids}")

    # Load all vectors into memory
    print("Loading all vectors...")
    all_data = {}  # (i, j) -> vector

    for i_idx, neuron_i_id in enumerate(neuron_ids):
        for j_idx, neuron_j_id in enumerate(neuron_ids):
            filename = f"causality_{neuron_i_id}_{neuron_j_id}.json"
            filepath = input_dir / filename

            if not filepath.exists():
                print(f"Warning: File not found: {filename}")
                all_data[(i_idx, j_idx)] = np.full(30, np.nan)
                continue

            data = load_causality_vector(filepath)
            vector = data['causality_vector']

            # Convert None to NaN
            vector_array = np.array([np.nan if v is None else v for v in vector])
            all_data[(i_idx, j_idx)] = vector_array

    print(f"Loaded {len(all_data)} vectors")

    # Create PDF with compact pages
    print(f"\nCreating PDF: {pdf_path}")
    with PdfPages(pdf_path) as pdf:
        for j_idx, neuron_j_id in enumerate(neuron_ids):
            print(f"  Creating page {j_idx + 1}/{num_neurons} (neuron {neuron_j_id})", end='\r')

            # Collect all vectors for this j
            all_vectors = []
            for i_idx in range(num_neurons):
                if (i_idx, j_idx) in all_data:
                    all_vectors.append(all_data[(i_idx, j_idx)])
                else:
                    all_vectors.append(np.full(30, np.nan))

            # Create compact page
            fig = create_compact_page(
                j_idx,
                neuron_ids,
                all_vectors,
                j_idx + 1,
                num_neurons
            )

            # Save to PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"\n\nPDF created successfully: {pdf_path}")
    print(f"Total pages: {num_neurons} (one per 'given' neuron)")


if __name__ == "__main__":
    main()
