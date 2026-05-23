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


def _compute_single_histogram(target_spikes, other_spikes, C_ms):
    """
    Compute histogram for a single (target, other) pair.
    Fully vectorized — no Python loop over target spikes.
    """
    C_sec = C_ms / 1000.0
    num_bins = 2 * C_ms

    if len(other_spikes) == 0:
        return np.zeros(num_bins, dtype=int)

    # Vectorized searchsorted over ALL target spikes at once
    left_idx = np.searchsorted(other_spikes, target_spikes - C_sec, side='left')
    right_idx = np.searchsorted(other_spikes, target_spikes + C_sec, side='right')

    counts = right_idx - left_idx
    total = int(counts.sum())

    if total == 0:
        return np.zeros(num_bins, dtype=int)

    # Build flat (target, other_spike) pairs without a Python loop:
    #   For each target i, include other_spikes[left_idx[i] .. right_idx[i])
    target_rep = np.repeat(target_spikes, counts)

    group_starts_flat = np.repeat(
        np.concatenate(([0], np.cumsum(counts[:-1]))), counts
    )
    local_offsets = np.arange(total) - group_starts_flat
    other_flat = other_spikes[np.repeat(left_idx, counts) + local_offsets]

    diffs_ms = np.round((other_flat - target_rep) * 1000).astype(int)
    diffs_ms = diffs_ms[diffs_ms != 0]

    if len(diffs_ms) == 0:
        return np.zeros(num_bins, dtype=int)

    neg = diffs_ms < 0
    pos = diffs_ms > 0
    bin_idx = np.empty_like(diffs_ms)
    bin_idx[neg] = diffs_ms[neg] + C_ms
    bin_idx[pos] = diffs_ms[pos] + C_ms - 1

    valid = (bin_idx >= 0) & (bin_idx < num_bins)
    return np.bincount(bin_idx[valid], minlength=num_bins)[:num_bins]


def compute_histogram_vectorized(target_spikes, other_spikes_list, causality_window_ms=10):
    """
    Compute histograms for multiple versions of other_spikes.

    Args:
        target_spikes: Spike times of neuron i
        other_spikes_list: List of spike time arrays (jittered versions)
        causality_window_ms: Causality window C

    Returns:
        histograms: Array of shape (n_versions, 2*C)
    """
    num_bins = 2 * causality_window_ms
    histograms = np.zeros((len(other_spikes_list), num_bins), dtype=int)

    if len(target_spikes) == 0:
        return histograms

    for v_idx, other_spikes in enumerate(other_spikes_list):
        histograms[v_idx] = _compute_single_histogram(target_spikes, other_spikes, causality_window_ms)

    return histograms


def generate_jittered_versions(spike_times, n_iterations, jitter_min_ms=20, jitter_max_ms=50):
    """
    Generate multiple jittered versions of spike times.
    Fully vectorized — generates all iterations as a 2D array at once.

    Returns:
        List of n_iterations jittered spike time arrays
    """
    if len(spike_times) == 0:
        return [spike_times.copy() for _ in range(n_iterations)]

    # Generate all magnitudes and directions at once: shape (n_iterations, n_spikes)
    magnitudes = np.random.uniform(jitter_min_ms, jitter_max_ms, size=(n_iterations, len(spike_times)))
    directions = np.random.choice([-1, 1], size=(n_iterations, len(spike_times)))
    jittered_all = spike_times[np.newaxis, :] + directions * magnitudes / 1000.0
    jittered_all.sort(axis=1)

    return list(jittered_all)  # list of 1D arrays


def process_neuron_pair(args):
    """
    Process a single neuron pair (i, j).
    Compute both Z-scores and p-values from the same bootstrap data.
    Returns None on failure so the caller can handle it gracefully.
    """
    i, j, target_spikes, other_spikes, original_hist, n_iterations, C, jitter_min_ms, jitter_max_ms = args

    try:
        jittered_versions = generate_jittered_versions(other_spikes, n_iterations, jitter_min_ms, jitter_max_ms)
        bootstrap_histograms = compute_histogram_vectorized(target_spikes, jittered_versions, C)

        mu = np.mean(bootstrap_histograms, axis=0)
        sigma = np.std(bootstrap_histograms, axis=0)

        # Vectorized Z-scores across all bins
        z_scores = np.where(sigma > 0, (original_hist - mu) / sigma, 0.0)

        # Vectorized p-values across all bins
        B = bootstrap_histograms.shape[0]
        n_ge = np.sum(bootstrap_histograms >= original_hist[np.newaxis, :], axis=0)
        n_le = np.sum(bootstrap_histograms <= original_hist[np.newaxis, :], axis=0)
        p_plus = (1 + n_ge) / (1 + B)
        p_minus = (1 + n_le) / (1 + B)
        p_two_sided = np.minimum(2 * np.minimum(p_plus, p_minus), 1.0)

        return i, j, z_scores, mu, sigma, p_two_sided, p_plus, p_minus

    except Exception as exc:
        print(f"\n  [WARNING] Pair ({i}, {j}) failed: {exc}")
        return None


def load_original_histograms(histogram_data_path):
    """Load the original histograms from the first analysis."""
    with open(histogram_data_path, 'r') as f:
        data = json.load(f)

    histograms = np.array(data['histograms'])
    bins = np.array(data['bins'])
    neuron_ids = data['neuron_ids']

    return histograms, bins, neuron_ids


def _make_colors_zscore(z_array):
    abs_z = np.abs(z_array)
    colors = np.where(abs_z >= 3, 'red', np.where(abs_z >= 2, 'orange', 'steelblue'))
    return list(colors)


def _make_colors_pvalue(p_array):
    colors = np.where(p_array < 0.001, 'red', np.where(p_array < 0.05, 'orange', 'steelblue'))
    return list(colors)


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

        if np.all(np.isnan(z)):
            ax.text(0.5, 0.5, 'Skipped\n(<300 spikes)', ha='center', va='center',
                   fontsize=6, color='gray', transform=ax.transAxes)
            ax.set_title(f'Neuron {neuron_id_j}', fontsize=7, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            plot_idx += 1
            continue

        colors = _make_colors_zscore(z)

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

        if np.all(np.isnan(p_vals)):
            ax.text(0.5, 0.5, 'Skipped\n(<300 spikes)', ha='center', va='center',
                   fontsize=6, color='gray', transform=ax.transAxes)
            ax.set_title(f'Neuron {neuron_id_j}', fontsize=7, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            plot_idx += 1
            continue

        p_vals_clipped = np.maximum(p_vals, 1e-10)
        neg_log_p = -np.log10(p_vals_clipped)
        colors = _make_colors_pvalue(p_vals)

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


def save_checkpoint(checkpoint_path, z_scores_all, mu_all, sigma_all,
                    p_two_sided_all, p_plus_all, p_minus_all, completed_pairs):
    np.savez_compressed(
        checkpoint_path,
        z_scores=z_scores_all,
        mu=mu_all,
        sigma=sigma_all,
        p_two_sided=p_two_sided_all,
        p_plus=p_plus_all,
        p_minus=p_minus_all,
        completed_pairs=np.array(list(completed_pairs)),
    )


def load_checkpoint(checkpoint_path):
    data = np.load(checkpoint_path)
    completed_pairs = set(map(tuple, data['completed_pairs'].tolist()))
    return (
        data['z_scores'], data['mu'], data['sigma'],
        data['p_two_sided'], data['p_plus'], data['p_minus'],
        completed_pairs,
    )


def main():
    neuron_data_path = Path('../data/neuron_data_interburst_lte10.json')
    histogram_data_path = Path('spike_triggered_histograms/histogram_data.json')
    output_dir = Path('significance_analysis')

    # Validate inputs before doing any work
    for p in [neuron_data_path, histogram_data_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / 'checkpoint.npz'

    z_scores_path = output_dir / 'z_scores.json'
    null_data_path = output_dir / 'null_hypothesis_data.json'
    z_pdf_path = output_dir / 'z_score_analysis.pdf'

    p_values_path = output_dir / 'p_values.json'
    p_onesided_path = output_dir / 'p_values_onesided.json'
    p_pdf_path = output_dir / 'p_value_analysis.pdf'

    print("="*70)
    print("COMBINED STATISTICAL SIGNIFICANCE ANALYSIS - FILTERED DATA")
    print("Computing both Z-scores and P-values from single bootstrap run")
    print("Filtered data: activity < 15 neurons (excludes burst spikes)")
    print("="*70)

    print("\nLoading original histograms...")
    original_histograms, bins, neuron_ids = load_original_histograms(histogram_data_path)
    N = len(neuron_ids)
    C = len(bins) // 2

    print(f"  Loaded {N} neurons")
    print(f"  Causality window: ±{C}ms")
    print(f"  Bins: {len(bins)} bins")

    print("\nLoading neuron spike data...")
    neuron_data = load_neuron_data(neuron_data_path)
    all_spike_times = [extract_spike_times(ns) for ns in neuron_data]

    n_iterations = 1000
    jitter_min_ms = 20
    jitter_max_ms = 50
    checkpoint_interval = 50

    # Build task list
    print(f"\nPreparing tasks for parallel processing...")
    print(f"  Bootstrap iterations per pair: {n_iterations}")
    print(f"  Jitter range: ±{jitter_min_ms}-{jitter_max_ms}ms")
    print(f"  CPU cores available: {cpu_count()}")

    all_tasks = []
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
            if np.sum(original_hist) < 300:
                skipped_pairs += 1
                continue
            all_tasks.append((i, j, target_spikes, other_spikes, original_hist,
                              n_iterations, C, jitter_min_ms, jitter_max_ms))

    print(f"  Skipped pairs with < 300 correlated spikes: {skipped_pairs}")
    print(f"  Total pairs to process: {len(all_tasks)}")

    # Resume from checkpoint if available
    completed_pairs = set()
    if checkpoint_path.exists():
        print(f"\n  Resuming from checkpoint ({checkpoint_path})...")
        (z_scores_all, mu_all, sigma_all,
         p_two_sided_all, p_plus_all, p_minus_all,
         completed_pairs) = load_checkpoint(checkpoint_path)
        print(f"  Already completed: {len(completed_pairs)} pairs")
    else:
        z_scores_all    = np.full((N, N, 2*C), np.nan, dtype=float)
        mu_all          = np.full((N, N, 2*C), np.nan, dtype=float)
        sigma_all       = np.full((N, N, 2*C), np.nan, dtype=float)
        p_two_sided_all = np.full((N, N, 2*C), np.nan, dtype=float)
        p_plus_all      = np.full((N, N, 2*C), np.nan, dtype=float)
        p_minus_all     = np.full((N, N, 2*C), np.nan, dtype=float)

    tasks = [(i, j, ts, os, oh, ni, c, jmin, jmax)
             for (i, j, ts, os, oh, ni, c, jmin, jmax) in all_tasks
             if (i, j) not in completed_pairs]
    total_pairs = len(all_tasks)
    remaining = len(tasks)
    print(f"  Remaining pairs: {remaining}")

    # Process in parallel
    print(f"\nProcessing pairs in parallel...")
    n_workers = max(1, cpu_count() - 1)
    failed_pairs = 0

    with Pool(processes=n_workers) as pool:
        for idx, result in enumerate(pool.imap_unordered(process_neuron_pair, tasks), 1):
            if result is None:
                failed_pairs += 1
                continue

            i, j, z_scores, mu, sigma, p_two, p_pl, p_mi = result
            z_scores_all[i][j]    = z_scores
            mu_all[i][j]          = mu
            sigma_all[i][j]       = sigma
            p_two_sided_all[i][j] = p_two
            p_plus_all[i][j]      = p_pl
            p_minus_all[i][j]     = p_mi
            completed_pairs.add((i, j))

            done_total = len(completed_pairs)
            print(f"  Processed {done_total}/{total_pairs} pairs ({100*done_total/total_pairs:.1f}%)", end='\r')

            # Periodic checkpoint
            if idx % checkpoint_interval == 0:
                save_checkpoint(checkpoint_path, z_scores_all, mu_all, sigma_all,
                                p_two_sided_all, p_plus_all, p_minus_all, completed_pairs)

    print(f"\n  Done!  (failed pairs: {failed_pairs})")

    # Final checkpoint before saving outputs
    save_checkpoint(checkpoint_path, z_scores_all, mu_all, sigma_all,
                    p_two_sided_all, p_plus_all, p_minus_all, completed_pairs)

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

    # Remove checkpoint now that outputs are saved
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Bootstrap iterations per pair: {n_iterations}")
    print(f"Jitter range: ±{jitter_min_ms}-{jitter_max_ms}ms per spike")
    print(f"Pairs analyzed: {total_pairs}  (failed: {failed_pairs})")
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
