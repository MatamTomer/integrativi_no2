import json
from pathlib import Path
from multiprocessing import Pool, cpu_count

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


def compute_histogram_vectorized(target_spikes, other_spikes_list, causality_window_ms=10):
    """
    Compute histograms for multiple versions of other_spikes in one pass.

    Args:
        target_spikes: Spike times of neuron i
        other_spikes_list: List of spike time arrays (jittered versions)
        causality_window_ms: Causality window C

    Returns:
        histograms: Array of shape (n_versions, 2*C) with histogram for each version
    """
    C = causality_window_ms
    C_sec = C / 1000.0
    num_bins = 2 * C
    n_versions = len(other_spikes_list)

    histograms = np.zeros((n_versions, num_bins), dtype=int)

    if len(target_spikes) == 0:
        return histograms

    # Process each version
    for v_idx, other_spikes in enumerate(other_spikes_list):
        if len(other_spikes) == 0:
            continue

        # Collect all time differences
        all_diffs = []

        for target_time in target_spikes:
            # Use searchsorted for efficiency
            left_idx = np.searchsorted(other_spikes, target_time - C_sec, side='left')
            right_idx = np.searchsorted(other_spikes, target_time + C_sec, side='right')

            nearby_spikes = other_spikes[left_idx:right_idx]

            if len(nearby_spikes) > 0:
                diffs = nearby_spikes - target_time
                all_diffs.append(diffs)

        if len(all_diffs) == 0:
            continue

        # Concatenate and process
        all_diffs = np.concatenate(all_diffs)
        all_diffs_ms = np.round(all_diffs * 1000).astype(int)
        all_diffs_ms = all_diffs_ms[all_diffs_ms != 0]

        if len(all_diffs_ms) == 0:
            continue

        # Vectorized binning
        negative_mask = all_diffs_ms < 0
        positive_mask = all_diffs_ms > 0

        bin_indices = np.zeros(len(all_diffs_ms), dtype=int)
        bin_indices[negative_mask] = all_diffs_ms[negative_mask] + C
        bin_indices[positive_mask] = all_diffs_ms[positive_mask] + C - 1

        valid_mask = (bin_indices >= 0) & (bin_indices < num_bins)
        valid_indices = bin_indices[valid_mask]

        if len(valid_indices) > 0:
            counts = np.bincount(valid_indices, minlength=num_bins)
            histograms[v_idx] = counts[:num_bins]

    return histograms


def generate_jittered_versions(spike_times, n_iterations, jitter_min_ms=20, jitter_max_ms=50):
    """
    Generate multiple jittered versions of spike times at once.

    Returns:
        List of n_iterations jittered spike time arrays
    """
    if len(spike_times) == 0:
        return [spike_times.copy() for _ in range(n_iterations)]

    jittered_versions = []

    for _ in range(n_iterations):
        # Random jitter magnitudes
        jitter_mags = np.random.uniform(jitter_min_ms, jitter_max_ms, size=len(spike_times))
        # Random directions
        jitter_dirs = np.random.choice([-1, 1], size=len(spike_times))
        # Apply jitter
        jitters = jitter_dirs * jitter_mags / 1000.0
        jittered = spike_times + jitters
        jittered.sort()
        jittered_versions.append(jittered)

    return jittered_versions


def compute_two_sided_pvalue(observed_count, bootstrap_counts):
    """
    Compute two-sided p-value using the formula:

    p_plus = (1 + count(bootstrap >= observed)) / (1 + B)
    p_minus = (1 + count(bootstrap <= observed)) / (1 + B)
    p_two_sided = 2 * min(p_plus, p_minus)

    Clip to 1 if greater than 1.

    Args:
        observed_count: The observed histogram count H_k
        bootstrap_counts: Array of B bootstrap/jittered counts H_k^(b)

    Returns:
        p_two_sided, p_plus, p_minus
    """
    B = len(bootstrap_counts)

    # Count how many bootstrap values are >= observed
    n_greater_equal = np.sum(bootstrap_counts >= observed_count)
    p_plus = (1 + n_greater_equal) / (1 + B)

    # Count how many bootstrap values are <= observed
    n_less_equal = np.sum(bootstrap_counts <= observed_count)
    p_minus = (1 + n_less_equal) / (1 + B)

    # Two-sided p-value
    p_two_sided = 2 * min(p_plus, p_minus)

    # Clip to 1
    p_two_sided = min(p_two_sided, 1.0)

    return p_two_sided, p_plus, p_minus


def compute_z_score(observed_count, mu, sigma):
    """
    Compute Z-score: Z = (observed - mu) / sigma

    Handle division by zero by setting Z=0 when sigma=0.
    """
    if sigma > 0:
        return (observed_count - mu) / sigma
    else:
        return 0.0


def process_neuron_pair(args):
    """
    Process a single neuron pair (i, j).
    Compute both Z-scores and p-values from the same bootstrap data.
    """
    i, j, target_spikes, other_spikes, original_hist, n_iterations, C, jitter_min_ms, jitter_max_ms = args

    # Generate jittered versions
    jittered_versions = generate_jittered_versions(other_spikes, n_iterations, jitter_min_ms, jitter_max_ms)

    # Compute histograms for all jittered versions (bootstrap data)
    bootstrap_histograms = compute_histogram_vectorized(target_spikes, jittered_versions, C)

    num_bins = len(original_hist)

    # Calculate statistics from bootstrap
    mu = np.mean(bootstrap_histograms, axis=0)
    sigma = np.std(bootstrap_histograms, axis=0)

    # Compute Z-scores and p-values for each bin
    z_scores = np.zeros(num_bins, dtype=float)
    p_two_sided = np.zeros(num_bins, dtype=float)
    p_plus = np.zeros(num_bins, dtype=float)
    p_minus = np.zeros(num_bins, dtype=float)

    for k in range(num_bins):
        observed = original_hist[k]
        bootstrap_values = bootstrap_histograms[:, k]

        # Z-score
        z_scores[k] = compute_z_score(observed, mu[k], sigma[k])

        # P-values
        p_two, p_pl, p_mi = compute_two_sided_pvalue(observed, bootstrap_values)
        p_two_sided[k] = p_two
        p_plus[k] = p_pl
        p_minus[k] = p_mi

    return i, j, z_scores, mu, sigma, p_two_sided, p_plus, p_minus


def load_original_histograms(histogram_data_path):
    """Load the original histograms from the first analysis."""
    with open(histogram_data_path, 'r') as f:
        data = json.load(f)

    histograms = np.array(data['histograms'])
    bins = np.array(data['bins'])
    neuron_ids = data['neuron_ids']

    return histograms, bins, neuron_ids


def create_z_score_page(neuron_i, neuron_id_i, z_scores, bins, neuron_ids):
    """Create a PDF page showing Z-score histograms for neuron i as target."""
    N = len(neuron_ids)
    num_others = N - 1

    cols = 7
    rows = (num_others + cols - 1) // cols

    fig = plt.figure(figsize=(17, 2.5 * rows))
    fig.suptitle(f'Z-Score Analysis: Neuron {neuron_id_i} (trigger)', fontsize=14, fontweight='bold')

    plot_idx = 0
    for j in range(N):
        if j == neuron_i:
            continue

        neuron_id_j = neuron_ids[j]
        z = z_scores[neuron_i][j]

        ax = fig.add_subplot(rows, cols, plot_idx + 1)

        # Check if this pair was skipped (all NaN)
        if np.all(np.isnan(z)):
            ax.text(0.5, 0.5, 'Skipped\n(<300 spikes)', ha='center', va='center',
                   fontsize=6, color='gray', transform=ax.transAxes)
            ax.set_title(f'Neuron {neuron_id_j}', fontsize=7, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            plot_idx += 1
            continue

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

        ax.bar(bins, z, width=0.8, color=colors, alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axhline(y=2, color='orange', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.axhline(y=-2, color='orange', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.axhline(y=3, color='red', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.axhline(y=-3, color='red', linestyle=':', linewidth=0.5, alpha=0.5)

        ax.set_title(f'Neuron {neuron_id_j}', fontsize=7, fontweight='bold')
        if plot_idx >= (rows - 1) * cols:
            ax.set_xlabel('Time offset (ms)', fontsize=6)
        if plot_idx % cols == 0:
            ax.set_ylabel('Z-score', fontsize=6)
        ax.tick_params(axis='both', labelsize=5)
        ax.grid(True, alpha=0.2, axis='y')
        ax.set_xlim(bins[0] - 1, bins[-1] + 1)

        plot_idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def create_pvalue_page(neuron_i, neuron_id_i, p_values, bins, neuron_ids):
    """Create a PDF page showing -log10(p-value) histograms for neuron i as target."""
    N = len(neuron_ids)
    num_others = N - 1

    cols = 7
    rows = (num_others + cols - 1) // cols

    fig = plt.figure(figsize=(17, 2.5 * rows))
    fig.suptitle(f'P-Value Analysis: Neuron {neuron_id_i} (trigger)', fontsize=14, fontweight='bold')

    plot_idx = 0
    for j in range(N):
        if j == neuron_i:
            continue

        neuron_id_j = neuron_ids[j]
        p_vals = p_values[neuron_i][j]

        ax = fig.add_subplot(rows, cols, plot_idx + 1)

        # Check if this pair was skipped (all NaN)
        if np.all(np.isnan(p_vals)):
            ax.text(0.5, 0.5, 'Skipped\n(<300 spikes)', ha='center', va='center',
                   fontsize=6, color='gray', transform=ax.transAxes)
            ax.set_title(f'Neuron {neuron_id_j}', fontsize=7, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            plot_idx += 1
            continue

        # Convert to -log10(p)
        p_vals_clipped = np.maximum(p_vals, 1e-10)
        neg_log_p = -np.log10(p_vals_clipped)

        # Color bars by significance
        colors = []
        for p_val in p_vals:
            if p_val < 0.001:
                colors.append('red')
            elif p_val < 0.05:
                colors.append('orange')
            else:
                colors.append('steelblue')

        ax.bar(bins, neg_log_p, width=0.8, color=colors, alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axhline(y=-np.log10(0.05), color='orange', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.axhline(y=-np.log10(0.01), color='red', linestyle=':', linewidth=0.5, alpha=0.5)

        ax.set_title(f'Neuron {neuron_id_j}', fontsize=7, fontweight='bold')
        if plot_idx >= (rows - 1) * cols:
            ax.set_xlabel('Time offset (ms)', fontsize=6)
        if plot_idx % cols == 0:
            ax.set_ylabel('-log10(p-value)', fontsize=6)
        ax.tick_params(axis='both', labelsize=5)
        ax.grid(True, alpha=0.2, axis='y')
        ax.set_xlim(bins[0] - 1, bins[-1] + 1)
        ax.set_ylim(bottom=0)

        plot_idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main():
    neuron_data_path = Path('../data/neuron_data_filtered_activity_lt15.json')
    histogram_data_path = Path('spike_triggered_histograms_filtered/histogram_data.json')
    output_dir = Path('significance_analysis_filtered')

    # Output file paths
    z_scores_path = output_dir / 'z_scores.json'
    null_data_path = output_dir / 'null_hypothesis_data.json'
    z_pdf_path = output_dir / 'z_score_analysis.pdf'

    p_values_path = output_dir / 'p_values.json'
    p_onesided_path = output_dir / 'p_values_onesided.json'
    p_pdf_path = output_dir / 'p_value_analysis.pdf'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("COMBINED STATISTICAL SIGNIFICANCE ANALYSIS - FILTERED DATA")
    print("Computing both Z-scores and P-values from single bootstrap run")
    print("Filtered data: activity < 15 neurons (excludes burst spikes)")
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

    # Parameters
    n_iterations = 1000
    jitter_min_ms = 20
    jitter_max_ms = 50

    # Prepare arguments for parallel processing
    print(f"\nPreparing tasks for parallel processing...")
    print(f"  Bootstrap iterations per pair: {n_iterations}")
    print(f"  Jitter range: ±{jitter_min_ms}-{jitter_max_ms}ms")
    print(f"  CPU cores available: {cpu_count()}")

    tasks = []
    skipped_pairs = 0
    for i in range(N):
        target_spikes = all_spike_times[i]
        if len(target_spikes) == 0:
            continue

        for j in range(N):
            if i == j:
                continue

            other_spikes = all_spike_times[j]
            if len(other_spikes) == 0:
                continue

            original_hist = original_histograms[i][j]

            # Skip pairs with fewer than 300 total correlated spikes
            total_correlated_spikes = np.sum(original_hist)
            if total_correlated_spikes < 300:
                skipped_pairs += 1
                continue

            tasks.append((i, j, target_spikes, other_spikes, original_hist,
                         n_iterations, C, jitter_min_ms, jitter_max_ms))

    total_pairs = len(tasks)
    print(f"  Skipped pairs with < 300 correlated spikes: {skipped_pairs}")
    print(f"  Total pairs to process: {total_pairs}")

    # Initialize storage for all results (use NaN for skipped pairs)
    z_scores_all = np.full((N, N, 2*C), np.nan, dtype=float)
    mu_all = np.full((N, N, 2*C), np.nan, dtype=float)
    sigma_all = np.full((N, N, 2*C), np.nan, dtype=float)

    p_two_sided_all = np.full((N, N, 2*C), np.nan, dtype=float)
    p_plus_all = np.full((N, N, 2*C), np.nan, dtype=float)
    p_minus_all = np.full((N, N, 2*C), np.nan, dtype=float)

    # Process in parallel
    print(f"\nProcessing pairs in parallel...")
    n_workers = max(1, cpu_count() - 1)

    with Pool(processes=n_workers) as pool:
        for idx, result in enumerate(pool.imap_unordered(process_neuron_pair, tasks), 1):
            i, j, z_scores, mu, sigma, p_two, p_pl, p_mi = result

            # Store Z-score results
            z_scores_all[i][j] = z_scores
            mu_all[i][j] = mu
            sigma_all[i][j] = sigma

            # Store p-value results
            p_two_sided_all[i][j] = p_two
            p_plus_all[i][j] = p_pl
            p_minus_all[i][j] = p_mi

            if idx % 10 == 0 or idx == total_pairs:
                print(f"  Processed {idx}/{total_pairs} pairs ({100*idx/total_pairs:.1f}%)", end='\r')

    print(f"\n  Done!")

    # Save Z-score results
    print(f"\nSaving Z-score results...")
    print(f"  {z_scores_path}")
    z_data = {
        'neuron_ids': neuron_ids,
        'bins': bins.tolist(),
        'causality_window_ms': C,
        'n_iterations': n_iterations,
        'jitter_range_ms': [jitter_min_ms, jitter_max_ms],
        'z_scores': z_scores_all.tolist(),
        'shape': list(z_scores_all.shape),
        'description': 'z_scores[i][j][k] = (H[i][j][k] - mu[i][j][k]) / sigma[i][j][k]'
    }
    with open(z_scores_path, 'w') as f:
        json.dump(z_data, f, indent=2)

    print(f"  {null_data_path}")
    null_data = {
        'neuron_ids': neuron_ids,
        'bins': bins.tolist(),
        'causality_window_ms': C,
        'n_iterations': n_iterations,
        'jitter_range_ms': [jitter_min_ms, jitter_max_ms],
        'mu': mu_all.tolist(),
        'sigma': sigma_all.tolist(),
        'shape': list(mu_all.shape),
        'description': 'mu[i][j][k] = mean of null distribution, sigma[i][j][k] = std deviation'
    }
    with open(null_data_path, 'w') as f:
        json.dump(null_data, f, indent=2)

    # Save p-value results
    print(f"\nSaving p-value results...")
    print(f"  {p_values_path}")
    p_data = {
        'neuron_ids': neuron_ids,
        'bins': bins.tolist(),
        'causality_window_ms': C,
        'n_iterations': n_iterations,
        'jitter_range_ms': [jitter_min_ms, jitter_max_ms],
        'p_values_two_sided': p_two_sided_all.tolist(),
        'shape': list(p_two_sided_all.shape),
        'description': 'p_values[i][j][k] = two-sided p-value for bin k, neuron pair (i,j)',
        'formula': 'p_two_sided = 2 * min(p_plus, p_minus), clipped to 1'
    }
    with open(p_values_path, 'w') as f:
        json.dump(p_data, f, indent=2)

    print(f"  {p_onesided_path}")
    p_onesided_data = {
        'neuron_ids': neuron_ids,
        'bins': bins.tolist(),
        'causality_window_ms': C,
        'n_iterations': n_iterations,
        'jitter_range_ms': [jitter_min_ms, jitter_max_ms],
        'p_plus': p_plus_all.tolist(),
        'p_minus': p_minus_all.tolist(),
        'shape': list(p_plus_all.shape),
        'description': 'p_plus[i][j][k] = upper tail p-value, p_minus[i][j][k] = lower tail p-value',
        'formula_p_plus': 'p_plus = (1 + count(bootstrap >= observed)) / (1 + B)',
        'formula_p_minus': 'p_minus = (1 + count(bootstrap <= observed)) / (1 + B)'
    }
    with open(p_onesided_path, 'w') as f:
        json.dump(p_onesided_data, f, indent=2)

    # Create Z-score PDF
    print(f"\nCreating Z-score PDF: {z_pdf_path}")
    with PdfPages(z_pdf_path) as pdf:
        for i in range(N):
            neuron_id_i = neuron_ids[i]
            fig = create_z_score_page(i, neuron_id_i, z_scores_all, bins, neuron_ids)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print(f"  Created Z-score page {i + 1}/{N} (neuron {neuron_id_i})", end='\r')
    print(f"\n  Done!")

    # Create p-value PDF
    print(f"\nCreating p-value PDF: {p_pdf_path}")
    with PdfPages(p_pdf_path) as pdf:
        for i in range(N):
            neuron_id_i = neuron_ids[i]
            fig = create_pvalue_page(i, neuron_id_i, p_two_sided_all, bins, neuron_ids)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print(f"  Created p-value page {i + 1}/{N} (neuron {neuron_id_i})", end='\r')
    print(f"\n  Done!")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Bootstrap iterations per pair: {n_iterations}")
    print(f"Jitter range: ±{jitter_min_ms}-{jitter_max_ms}ms per spike")
    print(f"Pairs analyzed: {total_pairs}")
    print()
    print("Z-score output files:")
    print(f"  - Z-scores: {z_scores_path}")
    print(f"  - Null hypothesis (mu, sigma): {null_data_path}")
    print(f"  - Visualization PDF: {z_pdf_path}")
    print()
    print("P-value output files:")
    print(f"  - Two-sided p-values: {p_values_path}")
    print(f"  - One-sided p-values: {p_onesided_path}")
    print(f"  - Visualization PDF: {p_pdf_path}")
    print()
    print("Interpretation:")
    print("  Z-scores:")
    print("    - Blue: |Z| < 2 (not significant)")
    print("    - Orange: 2 <= |Z| < 3 (significant, p < 0.05)")
    print("    - Red: |Z| >= 3 (highly significant, p < 0.01)")
    print("  P-values (shown as -log10(p)):")
    print("    - Blue: p >= 0.05 (not significant)")
    print("    - Orange: 0.001 < p < 0.05 (significant)")
    print("    - Red: p < 0.001 (highly significant)")
    print("="*70)


if __name__ == "__main__":
    main()
