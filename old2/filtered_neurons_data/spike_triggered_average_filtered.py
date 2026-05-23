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


def compute_spike_triggered_histograms(all_spike_times, causality_window_ms=10):
    """
    Compute spike-triggered histograms for all neuron pairs.

    Algorithm:
    - For each spike of neuron i (target)
    - Count spikes of neuron j (other) within ±C ms
    - Bin them by their time offset (excluding K=0)

    Args:
        all_spike_times: List of spike time arrays for each neuron
        causality_window_ms: Causality window in milliseconds (C)

    Returns:
        histograms: 3D array of shape (N, N, 2*C) where:
                   histograms[i][j][k] = count of neuron j spikes at bin k
                   relative to neuron i spikes
        bins: Array of bin values [-C, ..., -1, 1, ..., C]
    """
    N = len(all_spike_times)
    C = causality_window_ms

    # Bins: [-C, ..., -1, 1, ..., C] (excluding 0)
    bins = np.concatenate([np.arange(-C, 0), np.arange(1, C + 1)])
    num_bins = len(bins)  # Should be 2*C

    # Initialize histogram matrix: N x N x (2*C)
    histograms = np.zeros((N, N, num_bins), dtype=int)

    print(f"Computing spike-triggered histograms...")
    print(f"  Causality window: ±{C}ms")
    print(f"  Bins: {bins.tolist()}")
    print(f"  Number of neuron pairs: {N * (N-1)} (excluding self)")

    # Convert causality window to seconds
    C_sec = C / 1000.0

    # For each neuron pair (i, j) where i != j
    for i in range(N):
        target_spikes = all_spike_times[i]

        if len(target_spikes) == 0:
            continue

        for j in range(N):
            if i == j:  # Skip self-correlation
                continue

            other_spikes = all_spike_times[j]

            if len(other_spikes) == 0:
                continue

            # Vectorized approach: compute all time differences at once
            # For each target spike, find all other spikes within window
            all_diffs = []

            for target_time in target_spikes:
                # Use searchsorted to find spikes in window efficiently
                left_idx = np.searchsorted(other_spikes, target_time - C_sec, side='left')
                right_idx = np.searchsorted(other_spikes, target_time + C_sec, side='right')

                # Get spikes within window
                nearby_spikes = other_spikes[left_idx:right_idx]

                if len(nearby_spikes) > 0:
                    diffs = nearby_spikes - target_time
                    all_diffs.append(diffs)

            if len(all_diffs) == 0:
                continue

            # Concatenate all differences
            all_diffs = np.concatenate(all_diffs)

            # Convert to milliseconds and round
            all_diffs_ms = np.round(all_diffs * 1000).astype(int)

            # Exclude zero bin
            all_diffs_ms = all_diffs_ms[all_diffs_ms != 0]

            # Vectorized binning using np.bincount
            # Negative values: -C to -1 -> indices 0 to C-1
            # Positive values: 1 to C -> indices C to 2C-1
            negative_mask = all_diffs_ms < 0
            positive_mask = all_diffs_ms > 0

            # Calculate bin indices for all valid differences
            bin_indices = np.zeros(len(all_diffs_ms), dtype=int)

            # Negative bins
            bin_indices[negative_mask] = all_diffs_ms[negative_mask] + C

            # Positive bins
            bin_indices[positive_mask] = all_diffs_ms[positive_mask] + C - 1

            # Filter valid indices (within [0, num_bins))
            valid_mask = (bin_indices >= 0) & (bin_indices < num_bins)
            valid_indices = bin_indices[valid_mask]

            # Use bincount for fast counting
            if len(valid_indices) > 0:
                counts = np.bincount(valid_indices, minlength=num_bins)
                histograms[i][j] += counts[:num_bins]

        if (i + 1) % 5 == 0 or i == N - 1:
            print(f"  Processed {i + 1}/{N} target neurons", end='\r')

    print(f"\n  Done!")

    return histograms, bins


def create_histogram_page(neuron_i, neuron_id_i, histograms, bins, neuron_ids):
    """
    Create a single PDF page showing all histograms for neuron i as target.

    Args:
        neuron_i: Index of target neuron
        neuron_id_i: ID/name of target neuron
        histograms: Full 3D histogram array
        bins: Bin values
        neuron_ids: List of all neuron IDs
    """
    N = len(neuron_ids)
    num_others = N - 1  # Excluding self

    # Grid layout: 7 columns x 4 rows
    cols = 7
    rows = (num_others + cols - 1) // cols

    fig = plt.figure(figsize=(17, 2.5 * rows))
    fig.suptitle(f'Spike-Triggered Histograms: Neuron {neuron_id_i} (trigger)',
                 fontsize=14, fontweight='bold')

    plot_idx = 0
    for j in range(N):
        if j == neuron_i:  # Skip self
            continue

        neuron_id_j = neuron_ids[j]
        counts = histograms[neuron_i][j]

        # Create subplot
        ax = fig.add_subplot(rows, cols, plot_idx + 1)

        # Plot histogram
        ax.bar(bins, counts, width=0.8, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.3)

        # Add vertical line at zero (gap between -1 and 1)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8, alpha=0.6)

        # Labels
        ax.set_title(f'Neuron {neuron_id_j}', fontsize=7, fontweight='bold')
        if plot_idx >= (rows - 1) * cols:  # Bottom row
            ax.set_xlabel('Time offset (ms)', fontsize=6)
        if plot_idx % cols == 0:  # Left column
            ax.set_ylabel('Count', fontsize=6)
        ax.tick_params(axis='both', labelsize=5)
        ax.grid(True, alpha=0.2, axis='y')

        # Set x-axis to show the gap at zero
        ax.set_xlim(bins[0] - 1, bins[-1] + 1)

        plot_idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def save_histograms_to_file(histograms, bins, neuron_ids, output_path):
    """Save histogram data to JSON file."""
    data = {
        'causality_window_ms': int((len(bins) / 2)),
        'bins': bins.tolist(),
        'neuron_ids': neuron_ids,
        'histograms': histograms.tolist(),  # Convert to list for JSON serialization
        'shape': histograms.shape,
        'description': 'histograms[i][j][k] = count of neuron j spikes at bin k relative to neuron i spikes'
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved histogram data to: {output_path}")


def main():
    neuron_data_path = Path('../data/neuron_data_filtered_activity_lt15.json')
    output_dir = Path('spike_triggered_histograms_filtered')
    pdf_path = output_dir / 'spike_triggered_histograms.pdf'
    data_path = output_dir / 'histogram_data.json'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("SPIKE-TRIGGERED AVERAGE ANALYSIS - FILTERED DATA")
    print("="*70)

    # Load data
    print("\nLoading neuron data...")
    neuron_data = load_neuron_data(neuron_data_path)

    # Extract spike times and IDs
    all_spike_times = []
    neuron_ids = []
    for neuron_idx, neuron_spikes in enumerate(neuron_data):
        spike_times = extract_spike_times(neuron_spikes)
        all_spike_times.append(spike_times)
        neuron_id = neuron_spikes[0].get('unit') if neuron_spikes else neuron_idx + 1
        neuron_ids.append(neuron_id)

    N = len(neuron_ids)
    total_spikes = sum(len(spikes) for spikes in all_spike_times)

    print(f"  Found {N} neurons")
    print(f"  Total spikes: {total_spikes:,}")

    # Compute histograms
    print()
    C = 10  # Causality window in ms
    histograms, bins = compute_spike_triggered_histograms(all_spike_times, causality_window_ms=C)

    # Save histogram data
    print("\nSaving histogram data...")
    save_histograms_to_file(histograms, bins, neuron_ids, data_path)

    # Create PDF with all histograms
    print(f"\nCreating PDF: {pdf_path}")
    with PdfPages(pdf_path) as pdf:
        for i in range(N):
            neuron_id_i = neuron_ids[i]

            fig = create_histogram_page(i, neuron_id_i, histograms, bins, neuron_ids)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            print(f"  Created page {i + 1}/{N} (neuron {neuron_id_i})", end='\r')

    print(f"\n  Done!")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Causality window: ±{C}ms")
    print(f"Bins: {len(bins)} bins (excluding zero)")
    print(f"Number of neurons: {N}")
    print(f"Number of pairs (excluding self): {N * (N-1)}")
    print(f"Total histograms: {N * (N-1)}")
    print(f"Output PDF: {pdf_path}")
    print(f"Output data: {data_path}")
    print()
    print("Interpretation:")
    print("  - Each page shows one target (trigger) neuron")
    print("  - Each subplot shows activity of another neuron relative to trigger")
    print("  - Negative bins: other neuron fired BEFORE trigger")
    print("  - Positive bins: other neuron fired AFTER trigger")
    print("  - Bin K: other neuron fired K ms relative to trigger spike")
    print("="*70)


if __name__ == "__main__":
    main()
