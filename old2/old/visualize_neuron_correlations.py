import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def load_correlation_data(json_path: Path):
    """Load the correlation vectors from JSON."""
    with json_path.open('r') as f:
        data = json.load(f)
    return np.array(data)


def load_neuron_data(json_path: Path):
    """Load neuron data to get neuron IDs."""
    with json_path.open('r') as f:
        data = json.load(f)

    neuron_ids = []
    for neuron_idx, neuron_spikes in enumerate(data):
        neuron_id = neuron_spikes[0].get('unit') if neuron_spikes else neuron_idx + 1
        neuron_ids.append(neuron_id)

    return neuron_ids


def create_histogram(neuron_idx, neuron_id, correlation_vector, neuron_ids, output_path):
    """Create histogram for a single neuron excluding self-correlation."""
    # Remove the self-correlation (diagonal element)
    correlations = []
    labels = []

    for other_idx, count in enumerate(correlation_vector):
        if other_idx != neuron_idx:  # Exclude self
            correlations.append(count)
            labels.append(str(neuron_ids[other_idx]))

    correlations = np.array(correlations)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bar chart
    x_pos = np.arange(len(correlations))
    bars = ax.bar(x_pos, correlations, color='steelblue', alpha=0.8)

    # Highlight significant correlations
    threshold = np.mean(correlations) + np.std(correlations)
    for i, (bar, val) in enumerate(zip(bars, correlations)):
        if val > threshold:
            bar.set_color('coral')

    # Labels and title
    ax.set_xlabel('Other Neuron ID', fontsize=12)
    ax.set_ylabel('Correlation Count', fontsize=12)
    ax.set_title(f'Neuron {neuron_id} Correlations with Other Neurons', fontsize=14, fontweight='bold')

    # X-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Add statistics text
    stats_text = f'Mean: {np.mean(correlations):.1f}\nStd: {np.std(correlations):.1f}\nMax: {np.max(correlations)}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_compact_histogram(ax, neuron_idx, neuron_id, correlation_vector, neuron_ids):
    """Create a compact histogram for PDF layout."""
    # Remove the self-correlation
    correlations = []
    labels = []

    for other_idx, count in enumerate(correlation_vector):
        if other_idx != neuron_idx:
            correlations.append(count)
            labels.append(str(neuron_ids[other_idx]))

    correlations = np.array(correlations)

    # Create bar chart
    x_pos = np.arange(len(correlations))
    bars = ax.bar(x_pos, correlations, color='steelblue', alpha=0.8)

    # Highlight significant correlations
    if len(correlations) > 0 and np.std(correlations) > 0:
        threshold = np.mean(correlations) + np.std(correlations)
        for i, (bar, val) in enumerate(zip(bars, correlations)):
            if val > threshold:
                bar.set_color('coral')

    # Title
    ax.set_title(f'Neuron {neuron_id}', fontsize=10, fontweight='bold')

    # Labels
    ax.set_xlabel('Other Neuron ID', fontsize=8)
    ax.set_ylabel('Count', fontsize=8)

    # X-axis labels (smaller and rotated)
    ax.set_xticks(x_pos[::2])  # Show every other label to avoid crowding
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), 2)], rotation=45, ha='right', fontsize=6)

    # Tick labels
    ax.tick_params(axis='y', labelsize=7)

    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')


def main():
    # Paths
    correlation_path = Path('neuron_histograms/correlation_vectors.json')
    neuron_data_path = Path('neuron_data.json')
    output_dir = Path('neuron_histograms')
    pdf_path = Path('neuron_histograms/neuron_correlations.pdf')

    print("Loading data...")
    correlation_matrix = load_correlation_data(correlation_path)
    neuron_ids = load_neuron_data(neuron_data_path)

    num_neurons = len(neuron_ids)
    print(f"Found {num_neurons} neurons")

    print("\nGenerating individual histograms (excluding self-correlations)...")

    for neuron_idx in range(num_neurons):
        neuron_id = neuron_ids[neuron_idx]
        correlation_vector = correlation_matrix[neuron_idx]

        output_filename = f"neuron_{neuron_id}_histogram.png"
        output_path = output_dir / output_filename

        create_histogram(neuron_idx, neuron_id, correlation_vector, neuron_ids, output_path)

        print(f"  Created histogram {neuron_idx + 1}/{num_neurons}: {output_filename}", end='\r')

    print(f"\n\nAll {num_neurons} individual histograms saved to: {output_dir}/")

    # Create PDF with all histograms
    print(f"\nCreating PDF with all histograms: {pdf_path}")

    with PdfPages(pdf_path) as pdf:
        # Calculate pages needed (4 histograms per page in 2x2 grid)
        histograms_per_page = 4
        num_pages = (num_neurons + histograms_per_page - 1) // histograms_per_page

        for page_num in range(num_pages):
            fig = plt.figure(figsize=(17, 11))
            fig.suptitle(f'Neuron Correlation Histograms - Page {page_num + 1}/{num_pages}',
                        fontsize=16, fontweight='bold')

            # 2x2 grid
            for subplot_idx in range(histograms_per_page):
                neuron_idx = page_num * histograms_per_page + subplot_idx

                if neuron_idx >= num_neurons:
                    break

                neuron_id = neuron_ids[neuron_idx]
                correlation_vector = correlation_matrix[neuron_idx]

                ax = fig.add_subplot(2, 2, subplot_idx + 1)
                create_compact_histogram(ax, neuron_idx, neuron_id, correlation_vector, neuron_ids)

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            print(f"  Created PDF page {page_num + 1}/{num_pages}", end='\r')

    print(f"\n\nPDF created successfully: {pdf_path}")
    print(f"Total pages: {num_pages} (4 histograms per page)")


if __name__ == "__main__":
    main()
