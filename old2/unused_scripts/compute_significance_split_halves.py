import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
import shutil

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


def split_neuron_data_randomly(neuron_data, seed=42):
    """
    Split each neuron's spikes randomly into two halves (50/50).

    Args:
        neuron_data: Original neuron data (list of spike lists)
        seed: Random seed for reproducibility

    Returns:
        half1_data, half2_data: Two datasets with ~50% of spikes each
    """
    np.random.seed(seed)

    half1_data = []
    half2_data = []

    for neuron_spikes in neuron_data:
        if len(neuron_spikes) == 0:
            half1_data.append([])
            half2_data.append([])
            continue

        n_spikes = len(neuron_spikes)

        # Create random mask for 50/50 split
        mask = np.random.rand(n_spikes) < 0.5

        # Split spikes
        half1_spikes = [spike for i, spike in enumerate(neuron_spikes) if mask[i]]
        half2_spikes = [spike for i, spike in enumerate(neuron_spikes) if not mask[i]]

        half1_data.append(half1_spikes)
        half2_data.append(half2_spikes)

    return half1_data, half2_data


def save_split_neuron_data(neuron_data, output_path):
    """Save neuron data to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(neuron_data, f)
    print(f"  Saved to: {output_path}")


def run_significance_analysis_from_data(neuron_data, output_prefix, n_iterations=1000):
    """
    Run the full significance analysis pipeline on given neuron data.
    This is adapted from compute_significance_combined.py
    """
    from compute_significance_combined import (
        extract_spike_times,
        generate_jittered_versions,
        compute_histogram_vectorized,
        compute_two_sided_pvalue,
        compute_z_score,
        create_z_score_page,
        create_pvalue_page
    )

    print(f"\n{'='*70}")
    print(f"Running analysis: {output_prefix}")
    print(f"{'='*70}")

    # Extract spike times
    all_spike_times = []
    neuron_ids = []
    for neuron_idx, neuron_spikes in enumerate(neuron_data):
        spike_times = extract_spike_times(neuron_spikes)
        all_spike_times.append(spike_times)
        neuron_id = neuron_spikes[0].get('unit') if neuron_spikes else neuron_idx + 1
        neuron_ids.append(neuron_id)

    N = len(neuron_ids)
    C = 10  # Causality window
    jitter_min_ms = 20
    jitter_max_ms = 50

    print(f"  {N} neurons")
    print(f"  Total spikes: {sum(len(st) for st in all_spike_times):,}")

    # Compute original histograms
    print("  Computing spike-triggered histograms...")
    bins = np.concatenate([np.arange(-C, 0), np.arange(1, C + 1)])
    num_bins = 2 * C
    original_histograms = np.zeros((N, N, num_bins), dtype=int)

    # This is simplified - normally would call spike_triggered_average
    # For now, we'll just prepare for significance testing

    # Prepare tasks
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

            # Compute histogram for this pair (simplified)
            hist = compute_histogram_for_pair_simple(target_spikes, other_spikes, C)
            original_histograms[i][j] = hist

            # Skip pairs with < 300 spikes
            if np.sum(hist) < 300:
                skipped_pairs += 1
                continue

            tasks.append((i, j, target_spikes, other_spikes, hist,
                         n_iterations, C, jitter_min_ms, jitter_max_ms))

    print(f"  Skipped {skipped_pairs} pairs with <300 spikes")
    print(f"  Processing {len(tasks)} pairs with {n_iterations} iterations...")

    # Initialize results
    z_scores_all = np.full((N, N, 2*C), np.nan, dtype=float)
    mu_all = np.full((N, N, 2*C), np.nan, dtype=float)
    sigma_all = np.full((N, N, 2*C), np.nan, dtype=float)
    p_two_sided_all = np.full((N, N, 2*C), np.nan, dtype=float)
    p_plus_all = np.full((N, N, 2*C), np.nan, dtype=float)
    p_minus_all = np.full((N, N, 2*C), np.nan, dtype=float)

    # Process in parallel (simplified version)
    from compute_significance_combined import process_neuron_pair

    n_workers = max(1, cpu_count() - 1)
    with Pool(processes=n_workers) as pool:
        for idx, result in enumerate(pool.imap_unordered(process_neuron_pair, tasks), 1):
            i, j, z_scores, mu, sigma, p_two, p_pl, p_mi = result
            z_scores_all[i][j] = z_scores
            mu_all[i][j] = mu
            sigma_all[i][j] = sigma
            p_two_sided_all[i][j] = p_two
            p_plus_all[i][j] = p_pl
            p_minus_all[i][j] = p_mi

            if idx % 10 == 0 or idx == len(tasks):
                print(f"  Progress: {idx}/{len(tasks)} ({100*idx/len(tasks):.1f}%)", end='\\r')

    print(f"\\n  Done!")

    # Save results
    results = {
        'neuron_ids': neuron_ids,
        'bins': bins.tolist(),
        'z_scores': z_scores_all,
        'mu': mu_all,
        'sigma': sigma_all,
        'p_values': p_two_sided_all,
        'p_plus': p_plus_all,
        'p_minus': p_minus_all,
        'original_histograms': original_histograms
    }

    return results


def compute_histogram_for_pair_simple(target_spikes, other_spikes, C):
    """Simplified histogram computation for a single pair."""
    C_sec = C / 1000.0
    num_bins = 2 * C
    histogram = np.zeros(num_bins, dtype=int)

    if len(target_spikes) == 0 or len(other_spikes) == 0:
        return histogram

    all_diffs = []
    for target_time in target_spikes:
        left_idx = np.searchsorted(other_spikes, target_time - C_sec, side='left')
        right_idx = np.searchsorted(other_spikes, target_time + C_sec, side='right')
        nearby_spikes = other_spikes[left_idx:right_idx]

        if len(nearby_spikes) > 0:
            diffs = nearby_spikes - target_time
            all_diffs.append(diffs)

    if len(all_diffs) == 0:
        return histogram

    all_diffs = np.concatenate(all_diffs)
    all_diffs_ms = np.round(all_diffs * 1000).astype(int)
    all_diffs_ms = all_diffs_ms[all_diffs_ms != 0]

    if len(all_diffs_ms) == 0:
        return histogram

    negative_mask = all_diffs_ms < 0
    positive_mask = all_diffs_ms > 0

    bin_indices = np.zeros(len(all_diffs_ms), dtype=int)
    bin_indices[negative_mask] = all_diffs_ms[negative_mask] + C
    bin_indices[positive_mask] = all_diffs_ms[positive_mask] + C - 1

    valid_mask = (bin_indices >= 0) & (bin_indices < num_bins)
    valid_indices = bin_indices[valid_mask]

    if len(valid_indices) > 0:
        counts = np.bincount(valid_indices, minlength=num_bins)
        histogram = counts[:num_bins]

    return histogram


def save_analysis_results(results, output_base_path, half_name):
    """Save all analysis results to files."""
    output_dir = Path(output_base_path) / half_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Z-scores
    z_data = {
        'neuron_ids': results['neuron_ids'],
        'bins': results['bins'],
        'z_scores': results['z_scores'].tolist(),
        'shape': list(results['z_scores'].shape)
    }
    with open(output_dir / 'z_scores.json', 'w') as f:
        json.dump(z_data, f, indent=2)

    # Save null hypothesis data
    null_data = {
        'neuron_ids': results['neuron_ids'],
        'bins': results['bins'],
        'mu': results['mu'].tolist(),
        'sigma': results['sigma'].tolist(),
        'shape': list(results['mu'].shape)
    }
    with open(output_dir / 'null_hypothesis_data.json', 'w') as f:
        json.dump(null_data, f, indent=2)

    # Save p-values
    p_data = {
        'neuron_ids': results['neuron_ids'],
        'bins': results['bins'],
        'p_values_two_sided': results['p_values'].tolist(),
        'shape': list(results['p_values'].shape)
    }
    with open(output_dir / 'p_values.json', 'w') as f:
        json.dump(p_data, f, indent=2)

    p_onesided_data = {
        'neuron_ids': results['neuron_ids'],
        'bins': results['bins'],
        'p_plus': results['p_plus'].tolist(),
        'p_minus': results['p_minus'].tolist(),
        'shape': list(results['p_plus'].shape)
    }
    with open(output_dir / 'p_values_onesided.json', 'w') as f:
        json.dump(p_onesided_data, f, indent=2)

    print(f"Saved results to: {output_dir}")


def create_comparison_analysis(results_half1, results_half2, output_path):
    """Compare results between the two halves."""
    z1 = results_half1['z_scores']
    z2 = results_half2['z_scores']

    # Get valid (non-NaN) values
    valid_mask = (~np.isnan(z1)) & (~np.isnan(z2))
    z1_valid = z1[valid_mask]
    z2_valid = z2[valid_mask]

    # Compute correlation
    correlation = np.corrcoef(z1_valid, z2_valid)[0, 1]

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(z1_valid, z2_valid, alpha=0.3, s=10)
    ax1.plot([z1_valid.min(), z1_valid.max()],
             [z1_valid.min(), z1_valid.max()],
             'r--', linewidth=2, label='y=x')
    ax1.set_xlabel('Z-scores (Half 1)', fontsize=12)
    ax1.set_ylabel('Z-scores (Half 2)', fontsize=12)
    ax1.set_title(f'Z-Score Correlation: r={correlation:.3f}',
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Difference histogram
    ax2 = axes[1]
    diff = z1_valid - z2_valid
    ax2.hist(diff, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
    ax2.axvline(np.mean(diff), color='orange', linestyle='--', linewidth=2,
                label=f'Mean diff: {np.mean(diff):.3f}')
    ax2.set_xlabel('Z-score difference (Half1 - Half2)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Z-Score Differences', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\\nComparison Analysis:")
    print(f"  Correlation: {correlation:.3f}")
    print(f"  Mean difference: {np.mean(diff):.3f}")
    print(f"  Std difference: {np.std(diff):.3f}")
    print(f"  Saved comparison plot to: {output_path}")


def main():
    neuron_data_path = Path('neuron_data.json')
    output_base = Path('significance_analysis_split')

    print("="*70)
    print("SPLIT-HALF SIGNIFICANCE ANALYSIS")
    print("="*70)
    print()

    # Load original data
    print("Loading original neuron data...")
    neuron_data = load_neuron_data(neuron_data_path)
    total_spikes = sum(len(spikes) for spikes in neuron_data)
    print(f"  Neurons: {len(neuron_data)}")
    print(f"  Total spikes: {total_spikes:,}")
    print()

    # Split data randomly
    print("Splitting data into two random halves...")
    half1_data, half2_data = split_neuron_data_randomly(neuron_data)

    half1_spikes = sum(len(spikes) for spikes in half1_data)
    half2_spikes = sum(len(spikes) for spikes in half2_data)

    print(f"  Half 1: {half1_spikes:,} spikes ({100*half1_spikes/total_spikes:.1f}%)")
    print(f"  Half 2: {half2_spikes:,} spikes ({100*half2_spikes/total_spikes:.1f}%)")
    print()

    # Save split datasets
    print("Saving split datasets...")
    output_base.mkdir(parents=True, exist_ok=True)
    save_split_neuron_data(half1_data, output_base / 'neuron_data_half1.json')
    save_split_neuron_data(half2_data, output_base / 'neuron_data_half2.json')
    print()

    # NOTE: The actual analysis would require running the full compute_significance_combined
    # This is a template - you would need to integrate with the actual analysis pipeline

    print("\\nNOTE: Full significance analysis would be run here.")
    print("This requires integration with compute_significance_combined.py")
    print("\\nOutput structure created:")
    print(f"  {output_base}/neuron_data_half1.json")
    print(f"  {output_base}/neuron_data_half2.json")
    print(f"  {output_base}/half1/ (for analysis results)")
    print(f"  {output_base}/half2/ (for analysis results)")

    print("\\n" + "="*70)
    print("Split-half datasets created successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
