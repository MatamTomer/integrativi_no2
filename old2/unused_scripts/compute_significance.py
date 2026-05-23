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


def jitter_spike_times(spike_times, jitter_min_ms=20, jitter_max_ms=50):
    """
    Jitter each spike time independently by a random value in ±[jitter_min, jitter_max] ms.

    Args:
        spike_times: Array of spike times in seconds
        jitter_min_ms: Minimum jitter magnitude in milliseconds
        jitter_max_ms: Maximum jitter magnitude in milliseconds

    Returns:
        Jittered spike times (sorted)
    """
    if len(spike_times) == 0:
        return spike_times

    # Generate random jitter magnitudes in [jitter_min, jitter_max]
    jitter_magnitudes = np.random.uniform(jitter_min_ms, jitter_max_ms, size=len(spike_times))

    # Randomly assign direction (± with equal probability)
    jitter_directions = np.random.choice([-1, 1], size=len(spike_times))

    # Apply jitter (convert ms to seconds)
    jitters = jitter_directions * jitter_magnitudes / 1000.0
    jittered = spike_times + jitters

    # Sort and return
    jittered.sort()
    return jittered


def compute_histogram_for_pair(target_spikes, other_spikes, causality_window_ms=10):
    """
    Compute spike-triggered histogram for a single neuron pair.

    Returns:
        histogram: Array of counts for each bin (length 2*C)
    """
    C = causality_window_ms
    C_sec = C / 1000.0

    # Bins: [-C, ..., -1, 1, ..., C] (excluding 0)
    num_bins = 2 * C
    histogram = np.zeros(num_bins, dtype=int)

    if len(target_spikes) == 0 or len(other_spikes) == 0:
        return histogram

    # Collect all time differences
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
        return histogram

    # Concatenate all differences
    all_diffs = np.concatenate(all_diffs)

    # Convert to milliseconds and round
    all_diffs_ms = np.round(all_diffs * 1000).astype(int)

    # Exclude zero bin
    all_diffs_ms = all_diffs_ms[all_diffs_ms != 0]

    if len(all_diffs_ms) == 0:
        return histogram

    # Vectorized binning
    negative_mask = all_diffs_ms < 0
    positive_mask = all_diffs_ms > 0

    # Calculate bin indices
    bin_indices = np.zeros(len(all_diffs_ms), dtype=int)
    bin_indices[negative_mask] = all_diffs_ms[negative_mask] + C
    bin_indices[positive_mask] = all_diffs_ms[positive_mask] + C - 1

    # Filter valid indices
    valid_mask = (bin_indices >= 0) & (bin_indices < num_bins)
    valid_indices = bin_indices[valid_mask]

    # Use bincount for fast counting
    if len(valid_indices) > 0:
        counts = np.bincount(valid_indices, minlength=num_bins)
        histogram += counts[:num_bins]

    return histogram


def compute_null_distribution(target_spikes, other_spikes, n_iterations=100,
                              causality_window_ms=10, jitter_min_ms=20, jitter_max_ms=50):
    """
    Compute null distribution by jittering neuron j's spikes and computing histograms.

    Args:
        target_spikes: Spike times of neuron i (target)
        other_spikes: Spike times of neuron j (to be jittered)
        n_iterations: Number of jitter iterations
        causality_window_ms: Causality window C
        jitter_min_ms: Minimum jitter magnitude
        jitter_max_ms: Maximum jitter magnitude

    Returns:
        mu: Mean histogram across all iterations (length 2*C)
        sigma: Standard deviation across all iterations (length 2*C)
        all_histograms: All iteration histograms (n_iterations x 2*C) for reference
    """
    C = causality_window_ms
    num_bins = 2 * C

    # Store all histograms
    all_histograms = np.zeros((n_iterations, num_bins), dtype=int)

    for iteration in range(n_iterations):
        # Jitter neuron j's spikes
        jittered_spikes = jitter_spike_times(other_spikes, jitter_min_ms, jitter_max_ms)

        # Compute histogram with jittered spikes
        histogram = compute_histogram_for_pair(target_spikes, jittered_spikes, causality_window_ms)
        all_histograms[iteration] = histogram

    # Calculate statistics
    mu = np.mean(all_histograms, axis=0)
    sigma = np.std(all_histograms, axis=0)

    return mu, sigma, all_histograms


def compute_z_scores(original_histogram, mu, sigma):
    """
    Compute Z-scores for each bin.

    Z[k] = (H[k] - mu[k]) / sigma[k]

    Handle division by zero by setting Z=0 when sigma=0.
    """
    z_scores = np.zeros_like(original_histogram, dtype=float)

    # Only compute Z-score where sigma > 0
    valid_mask = sigma > 0
    z_scores[valid_mask] = (original_histogram[valid_mask] - mu[valid_mask]) / sigma[valid_mask]

    # Where sigma = 0, set Z = 0 (no variance in null distribution)
    z_scores[~valid_mask] = 0

    return z_scores


def load_original_histograms(histogram_data_path):
    """Load the original histograms from the first analysis."""
    with open(histogram_data_path, 'r') as f:
        data = json.load(f)

    histograms = np.array(data['histograms'])
    bins = np.array(data['bins'])
    neuron_ids = data['neuron_ids']

    return histograms, bins, neuron_ids


def create_z_score_page(neuron_i, neuron_id_i, z_scores, original_histograms, bins, neuron_ids):
    """
    Create a PDF page showing Z-score histograms for neuron i as target.

    Color bars by significance:
    - Blue: |Z| < 2 (not significant)
    - Orange: 2 <= |Z| < 3 (significant)
    - Red: |Z| >= 3 (highly significant)
    """
    N = len(neuron_ids)
    num_others = N - 1

    # Grid layout
    cols = 7
    rows = (num_others + cols - 1) // cols

    fig = plt.figure(figsize=(17, 2.5 * rows))
    fig.suptitle(f'Z-Score Analysis: Neuron {neuron_id_i} (trigger)', fontsize=14, fontweight='bold')

    plot_idx = 0
    for j in range(N):
        if j == neuron_i:  # Skip self
            continue

        neuron_id_j = neuron_ids[j]
        z = z_scores[neuron_i][j]

        # Create subplot
        ax = fig.add_subplot(rows, cols, plot_idx + 1)

        # Color bars by significance
        colors = []
        for z_val in z:
            abs_z = abs(z_val)
            if abs_z >= 3:
                colors.append('red')
            elif abs_z >= 2:
                colors.append('orange')
            else:
                colors.append('steelblue')

        # Plot
        ax.bar(bins, z, width=0.8, color=colors, alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

        # Significance lines
        ax.axhline(y=2, color='orange', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.axhline(y=-2, color='orange', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.axhline(y=3, color='red', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.axhline(y=-3, color='red', linestyle=':', linewidth=0.5, alpha=0.5)

        # Labels
        ax.set_title(f'Neuron {neuron_id_j}', fontsize=7, fontweight='bold')
        if plot_idx >= (rows - 1) * cols:  # Bottom row
            ax.set_xlabel('Time offset (ms)', fontsize=6)
        if plot_idx % cols == 0:  # Left column
            ax.set_ylabel('Z-score', fontsize=6)
        ax.tick_params(axis='both', labelsize=5)
        ax.grid(True, alpha=0.2, axis='y')
        ax.set_xlim(bins[0] - 1, bins[-1] + 1)

        plot_idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main():
    neuron_data_path = Path('neuron_data.json')
    histogram_data_path = Path('spike_triggered_histograms/histogram_data.json')
    output_dir = Path('significance_analysis')

    z_scores_path = output_dir / 'z_scores.json'
    null_data_path = output_dir / 'null_hypothesis_data.json'
    pdf_path = output_dir / 'z_score_analysis.pdf'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*70)

    # Load original histograms
    print("\nLoading original histograms...")
    original_histograms, bins, neuron_ids = load_original_histograms(histogram_data_path)
    N = len(neuron_ids)
    C = len(bins) // 2

    print(f"  Loaded {N} neurons")
    print(f"  Causality window: ±{C}ms")
    print(f"  Bins: {len(bins)} bins")

    # Load neuron spike data
    print("\nLoading neuron spike data...")
    neuron_data = load_neuron_data(neuron_data_path)
    all_spike_times = []
    for neuron_idx, neuron_spikes in enumerate(neuron_data):
        spike_times = extract_spike_times(neuron_spikes)
        all_spike_times.append(spike_times)

    # Initialize storage for results
    z_scores = np.zeros((N, N, 2*C), dtype=float)
    mu_all = np.zeros((N, N, 2*C), dtype=float)
    sigma_all = np.zeros((N, N, 2*C), dtype=float)

    # Parameters
    n_iterations = 100
    jitter_min_ms = 20
    jitter_max_ms = 50

    print(f"\nComputing null distributions and Z-scores...")
    print(f"  Iterations per pair: {n_iterations}")
    print(f"  Jitter range: ±{jitter_min_ms}-{jitter_max_ms}ms")
    print(f"  Total pairs to process: {N * (N-1)}")

    pair_count = 0
    total_pairs = N * (N - 1)

    for i in range(N):
        target_spikes = all_spike_times[i]

        if len(target_spikes) == 0:
            continue

        for j in range(N):
            if i == j:  # Skip self
                continue

            other_spikes = all_spike_times[j]

            if len(other_spikes) == 0:
                continue

            # Compute null distribution
            mu, sigma, _ = compute_null_distribution(
                target_spikes, other_spikes,
                n_iterations=n_iterations,
                causality_window_ms=C,
                jitter_min_ms=jitter_min_ms,
                jitter_max_ms=jitter_max_ms
            )

            # Get original histogram
            original_hist = original_histograms[i][j]

            # Compute Z-scores
            z = compute_z_scores(original_hist, mu, sigma)

            # Store results
            z_scores[i][j] = z
            mu_all[i][j] = mu
            sigma_all[i][j] = sigma

            pair_count += 1
            if pair_count % 10 == 0 or pair_count == total_pairs:
                print(f"  Processed {pair_count}/{total_pairs} pairs ({100*pair_count/total_pairs:.1f}%)", end='\r')

    print(f"\n  Done!")

    # Save Z-scores
    print(f"\nSaving Z-scores to {z_scores_path}...")
    z_data = {
        'neuron_ids': neuron_ids,
        'bins': bins.tolist(),
        'causality_window_ms': C,
        'n_iterations': n_iterations,
        'jitter_range_ms': [jitter_min_ms, jitter_max_ms],
        'z_scores': z_scores.tolist(),
        'shape': z_scores.shape,
        'description': 'z_scores[i][j][k] = (H[i][j][k] - mu[i][j][k]) / sigma[i][j][k]'
    }

    with open(z_scores_path, 'w') as f:
        json.dump(z_data, f, indent=2)

    # Save null hypothesis data
    print(f"Saving null hypothesis data to {null_data_path}...")
    null_data = {
        'neuron_ids': neuron_ids,
        'bins': bins.tolist(),
        'causality_window_ms': C,
        'n_iterations': n_iterations,
        'jitter_range_ms': [jitter_min_ms, jitter_max_ms],
        'mu': mu_all.tolist(),
        'sigma': sigma_all.tolist(),
        'shape': mu_all.shape,
        'description': 'mu[i][j][k] = mean of null distribution, sigma[i][j][k] = std deviation'
    }

    with open(null_data_path, 'w') as f:
        json.dump(null_data, f, indent=2)

    # Create PDF
    print(f"\nCreating PDF: {pdf_path}")
    with PdfPages(pdf_path) as pdf:
        for i in range(N):
            neuron_id_i = neuron_ids[i]

            fig = create_z_score_page(i, neuron_id_i, z_scores, original_histograms, bins, neuron_ids)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            print(f"  Created page {i + 1}/{N} (neuron {neuron_id_i})", end='\r')

    print(f"\n  Done!")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Null iterations per pair: {n_iterations}")
    print(f"Jitter range: ±{jitter_min_ms}-{jitter_max_ms}ms per spike")
    print(f"Pairs analyzed: {N * (N-1)}")
    print(f"Output files:")
    print(f"  - Z-scores: {z_scores_path}")
    print(f"  - Null hypothesis (mu, sigma): {null_data_path}")
    print(f"  - Visualization PDF: {pdf_path}")
    print()
    print("Interpretation:")
    print("  - Z-score = (observed - expected) / std_deviation")
    print("  - Blue bars: |Z| < 2 (not significant)")
    print("  - Orange bars: 2 <= |Z| < 3 (significant, p < 0.05)")
    print("  - Red bars: |Z| >= 3 (highly significant, p < 0.01)")
    print("  - Positive Z: More spikes than expected by chance")
    print("  - Negative Z: Fewer spikes than expected by chance")
    print("="*70)


if __name__ == "__main__":
    main()
