import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def load_neuron_data(json_path: Path):
    """Load neuron spike data from JSON file."""
    with json_path.open('r') as f:
        data = json.load(f)
    return data


def extract_spike_times(neuron_spikes):
    """Extract spike times from a neuron's spike list."""
    if not neuron_spikes:
        return np.array([], dtype=float)
    spike_times = np.array([float(spike['spikeTime']) for spike in neuron_spikes], dtype=float)
    spike_times.sort()
    return spike_times


def compute_cross_correlogram(target_spikes, other_spikes, time_window=0.04, bin_size=0.001):
    """
    Compute cross-correlogram between two neurons.

    Args:
        target_spikes: Spike times of target neuron
        other_spikes: Spike times of other neuron
        time_window: Total time window in seconds (e.g., 0.04 = ±20ms)
        bin_size: Size of each bin in seconds (e.g., 0.001 = 1ms)

    Returns:
        bin_centers: Time values for x-axis
        counts: Spike counts for each bin
    """
    if len(target_spikes) == 0 or len(other_spikes) == 0:
        return np.array([]), np.array([])

    half_window = time_window / 2
    num_bins = int(time_window / bin_size)
    bin_edges = np.linspace(-half_window, half_window, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Accumulator for all time differences
    all_diffs = []

    # For each target spike, find other neuron spikes within window
    for target_time in target_spikes:
        # Find other spikes within the window
        time_diffs = other_spikes - target_time

        # Keep only diffs within our window
        valid_diffs = time_diffs[(time_diffs >= -half_window) & (time_diffs <= half_window)]
        all_diffs.extend(valid_diffs)

    # Create histogram
    counts, _ = np.histogram(all_diffs, bins=bin_edges)

    return bin_centers, counts


def create_cross_correlogram_plot(neuron_idx, neuron_id, all_spike_times, neuron_ids, output_path):
    """Create cross-correlograms for one target neuron vs all others."""
    num_neurons = len(neuron_ids)
    target_spikes = all_spike_times[neuron_idx]

    # Calculate grid size (exclude self)
    num_others = num_neurons - 1
    cols = 7  # 7 columns
    rows = (num_others + cols - 1) // cols

    fig = plt.figure(figsize=(20, rows * 2.5))
    fig.suptitle(f'Cross-Correlograms for Neuron {neuron_id}', fontsize=16, fontweight='bold')

    plot_idx = 0
    for other_idx in range(num_neurons):
        if other_idx == neuron_idx:  # Skip self
            continue

        other_id = neuron_ids[other_idx]
        other_spikes = all_spike_times[other_idx]

        # Compute cross-correlogram
        bin_centers, counts = compute_cross_correlogram(target_spikes, other_spikes)

        # Create subplot
        ax = fig.add_subplot(rows, cols, plot_idx + 1)

        if len(bin_centers) > 0:
            # Convert to milliseconds for display
            bin_centers_ms = bin_centers * 1000

            # Plot as bar chart
            ax.bar(bin_centers_ms, counts, width=0.8, color='steelblue', alpha=0.7, edgecolor='none')

            # Add vertical line at zero
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

            # Labels
            ax.set_title(f'vs Neuron {other_id}', fontsize=8, fontweight='bold')
            ax.set_xlabel('Time (ms)', fontsize=7)
            ax.set_ylabel('Count', fontsize=7)
            ax.tick_params(axis='both', labelsize=6)
            ax.grid(True, alpha=0.2, axis='y')

        plot_idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_compact_correlogram_page(neuron_idx, neuron_id, all_spike_times, neuron_ids):
    """Create a single page with all cross-correlograms for one neuron (compact for PDF)."""
    num_neurons = len(neuron_ids)
    target_spikes = all_spike_times[neuron_idx]

    # Grid: 7 columns x 4 rows = 28 plots (perfect for 28 other neurons)
    cols = 7
    rows = 4

    fig = plt.figure(figsize=(17, 11))
    fig.suptitle(f'Cross-Correlograms: Neuron {neuron_id} (target)', fontsize=14, fontweight='bold')

    plot_idx = 0
    for other_idx in range(num_neurons):
        if other_idx == neuron_idx:  # Skip self
            continue

        if plot_idx >= rows * cols:  # Safety check
            break

        other_id = neuron_ids[other_idx]
        other_spikes = all_spike_times[other_idx]

        # Compute cross-correlogram
        bin_centers, counts = compute_cross_correlogram(target_spikes, other_spikes)

        # Create subplot
        ax = fig.add_subplot(rows, cols, plot_idx + 1)

        if len(bin_centers) > 0:
            bin_centers_ms = bin_centers * 1000
            ax.bar(bin_centers_ms, counts, width=0.8, color='steelblue', alpha=0.7, edgecolor='none')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)

            ax.set_title(f'vs {other_id}', fontsize=7, fontweight='bold')
            if plot_idx >= (rows - 1) * cols:  # Bottom row
                ax.set_xlabel('Time (ms)', fontsize=5)
            ax.tick_params(axis='both', labelsize=5)
            ax.grid(True, alpha=0.2, axis='y')

        plot_idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def split_spike_times_by_time(all_spike_times):
    """Split spike times into first half and second half based on time."""
    # Find the overall time range across all neurons
    min_time = float('inf')
    max_time = float('-inf')

    for spike_times in all_spike_times:
        if len(spike_times) > 0:
            min_time = min(min_time, spike_times[0])
            max_time = max(max_time, spike_times[-1])

    # Calculate midpoint
    mid_time = (min_time + max_time) / 2

    print(f"  Time range: {min_time:.3f}s to {max_time:.3f}s")
    print(f"  Midpoint: {mid_time:.3f}s")

    # Split each neuron's spike times
    first_half = []
    second_half = []

    for spike_times in all_spike_times:
        first = spike_times[spike_times < mid_time]
        second = spike_times[spike_times >= mid_time]
        first_half.append(first)
        second_half.append(second)

    return first_half, second_half, mid_time


def main():
    neuron_data_path = Path('neuron_data.json')
    output_dir = Path('neuron_histograms')
    pdf_path = Path('neuron_histograms/cross_correlograms_20ms_window.pdf')

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading neuron data...")
    neuron_data = load_neuron_data(neuron_data_path)

    # Extract spike times and IDs
    all_spike_times = []
    neuron_ids = []
    for neuron_idx, neuron_spikes in enumerate(neuron_data):
        spike_times = extract_spike_times(neuron_spikes)
        all_spike_times.append(spike_times)
        neuron_id = neuron_spikes[0].get('unit') if neuron_spikes else neuron_idx + 1
        neuron_ids.append(neuron_id)

    num_neurons = len(neuron_ids)
    total_spikes = sum(len(spikes) for spikes in all_spike_times)
    print(f"Found {num_neurons} neurons")
    print(f"Total spikes: {total_spikes:,}")

    # Create PDF with all data
    print(f"\n{'='*60}")
    print(f"Creating PDF with ±20ms window: {pdf_path}")
    print(f"{'='*60}")

    with PdfPages(pdf_path) as pdf:
        for neuron_idx in range(num_neurons):
            neuron_id = neuron_ids[neuron_idx]

            fig = create_compact_correlogram_page(neuron_idx, neuron_id, all_spike_times, neuron_ids)
            fig.suptitle(f'Cross-Correlograms (±20ms window): Neuron {neuron_id} (target)', fontsize=14, fontweight='bold')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            print(f"  Created PDF page {neuron_idx + 1}/{num_neurons} (neuron {neuron_id})", end='\r')

    print(f"\n\nPDF created successfully: {pdf_path}")

    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"{'='*60}")
    print(f"Total pages: {num_neurons} (one per target neuron)")
    print(f"Total spikes analyzed: {total_spikes:,}")
    print(f"Time window: ±20ms (40ms total)")
    print(f"Bin size: 1ms")
    print("\nInterpretation:")
    print("  - X-axis: Time relative to target neuron spike (ms)")
    print("  - Y-axis: Count of other neuron spikes")
    print("  - Red dashed line: Time of target spike (t=0)")
    print("  - Peak at positive time: Other neuron fires AFTER target")
    print("  - Peak at negative time: Other neuron fires BEFORE target")
    print("  - Trough: Suppression/inhibition")


if __name__ == "__main__":
    main()
