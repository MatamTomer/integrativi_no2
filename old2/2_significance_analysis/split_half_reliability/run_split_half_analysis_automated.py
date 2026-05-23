"""
Fully automated split-half reliability analysis.
Splits data, runs complete analysis on both halves, and compares results.
"""
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

# Add current directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent))


def load_neuron_data(json_path):
    """Load neuron spike data from JSON file."""
    with json_path.open('r') as f:
        data = json.load(f)
    return data


def split_neuron_data_randomly(neuron_data, seed=42):
    """
    Split each neuron's spikes randomly into two halves (50/50).
    Each neuron gets an independent random split.
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
        mask = np.random.rand(n_spikes) < 0.5

        half1_spikes = [spike for i, spike in enumerate(neuron_spikes) if mask[i]]
        half2_spikes = [spike for i, spike in enumerate(neuron_spikes) if not mask[i]]

        half1_data.append(half1_spikes)
        half2_data.append(half2_spikes)

    return half1_data, half2_data


def save_neuron_data(neuron_data, output_path):
    """Save neuron data to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(neuron_data, f)


def run_spike_triggered_average(data_path, output_dir):
    """Run spike-triggered average analysis."""
    from spike_triggered_average import (
        load_neuron_data,
        extract_spike_times,
        compute_spike_triggered_histograms,
        save_histograms_to_file
    )

    print(f"  Running spike-triggered average...")

    neuron_data = load_neuron_data(data_path)
    all_spike_times = []
    neuron_ids = []

    for neuron_idx, neuron_spikes in enumerate(neuron_data):
        spike_times = extract_spike_times(neuron_spikes)
        all_spike_times.append(spike_times)
        neuron_id = neuron_spikes[0].get('unit') if neuron_spikes else neuron_idx + 1
        neuron_ids.append(neuron_id)

    histograms, bins = compute_spike_triggered_histograms(all_spike_times, causality_window_ms=10)

    output_dir.mkdir(parents=True, exist_ok=True)
    histogram_path = output_dir / 'histogram_data.json'
    save_histograms_to_file(histograms, bins, neuron_ids, histogram_path)

    print(f"    Saved histograms to: {histogram_path}")
    return histogram_path


def run_significance_analysis(histogram_path, neuron_data_path, output_dir):
    """Run significance analysis (Z-scores and p-values)."""
    from compute_significance_combined import (
        load_original_histograms,
        load_neuron_data,
        extract_spike_times,
        process_neuron_pair,
        create_z_score_page,
        create_pvalue_page
    )
    from multiprocessing import Pool, cpu_count
    from matplotlib.backends.backend_pdf import PdfPages

    print(f"  Running significance analysis...")

    # Load data
    original_histograms, bins, neuron_ids = load_original_histograms(histogram_path)
    neuron_data = load_neuron_data(neuron_data_path)

    all_spike_times = []
    for neuron_idx, neuron_spikes in enumerate(neuron_data):
        spike_times = extract_spike_times(neuron_spikes)
        all_spike_times.append(spike_times)

    N = len(neuron_ids)
    C = len(bins) // 2
    n_iterations = 1000
    jitter_min_ms = 20
    jitter_max_ms = 50

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

            original_hist = original_histograms[i][j]
            total_correlated_spikes = np.sum(original_hist)

            if total_correlated_spikes < 300:
                skipped_pairs += 1
                continue

            tasks.append((i, j, target_spikes, other_spikes, original_hist,
                         n_iterations, C, jitter_min_ms, jitter_max_ms))

    print(f"    Processing {len(tasks)} pairs (skipped {skipped_pairs})...")

    # Initialize storage
    z_scores_all = np.full((N, N, 2*C), np.nan, dtype=float)
    mu_all = np.full((N, N, 2*C), np.nan, dtype=float)
    sigma_all = np.full((N, N, 2*C), np.nan, dtype=float)
    p_two_sided_all = np.full((N, N, 2*C), np.nan, dtype=float)
    p_plus_all = np.full((N, N, 2*C), np.nan, dtype=float)
    p_minus_all = np.full((N, N, 2*C), np.nan, dtype=float)

    # Process in parallel
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

            if idx % 50 == 0 or idx == len(tasks):
                print(f"      Progress: {idx}/{len(tasks)} ({100*idx/len(tasks):.1f}%)", end='\r')

    print(f"\n    Done processing!")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Z-scores
    z_data = {
        'neuron_ids': neuron_ids,
        'bins': bins.tolist(),
        'causality_window_ms': C,
        'n_iterations': n_iterations,
        'jitter_range_ms': [jitter_min_ms, jitter_max_ms],
        'z_scores': z_scores_all.tolist(),
        'shape': list(z_scores_all.shape)
    }
    with open(output_dir / 'z_scores.json', 'w') as f:
        json.dump(z_data, f, indent=2)

    # Null hypothesis data
    null_data = {
        'neuron_ids': neuron_ids,
        'bins': bins.tolist(),
        'causality_window_ms': C,
        'n_iterations': n_iterations,
        'jitter_range_ms': [jitter_min_ms, jitter_max_ms],
        'mu': mu_all.tolist(),
        'sigma': sigma_all.tolist(),
        'shape': list(mu_all.shape)
    }
    with open(output_dir / 'null_hypothesis_data.json', 'w') as f:
        json.dump(null_data, f, indent=2)

    # P-values
    p_data = {
        'neuron_ids': neuron_ids,
        'bins': bins.tolist(),
        'causality_window_ms': C,
        'n_iterations': n_iterations,
        'jitter_range_ms': [jitter_min_ms, jitter_max_ms],
        'p_values_two_sided': p_two_sided_all.tolist(),
        'shape': list(p_two_sided_all.shape)
    }
    with open(output_dir / 'p_values.json', 'w') as f:
        json.dump(p_data, f, indent=2)

    p_onesided_data = {
        'neuron_ids': neuron_ids,
        'bins': bins.tolist(),
        'causality_window_ms': C,
        'n_iterations': n_iterations,
        'jitter_range_ms': [jitter_min_ms, jitter_max_ms],
        'p_plus': p_plus_all.tolist(),
        'p_minus': p_minus_all.tolist(),
        'shape': list(p_plus_all.shape)
    }
    with open(output_dir / 'p_values_onesided.json', 'w') as f:
        json.dump(p_onesided_data, f, indent=2)

    # Create PDFs
    print(f"    Creating visualizations...")

    with PdfPages(output_dir / 'z_score_analysis.pdf') as pdf:
        for i in range(N):
            fig = create_z_score_page(i, neuron_ids[i], z_scores_all, bins, neuron_ids)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    with PdfPages(output_dir / 'p_value_analysis.pdf') as pdf:
        for i in range(N):
            fig = create_pvalue_page(i, neuron_ids[i], p_two_sided_all, bins, neuron_ids)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"    Saved all results to: {output_dir}")


def create_comparison_analysis(output_base):
    """Compare results between the two halves."""
    print("\n  Creating comparison analysis...")

    half1_dir = output_base / 'half1'
    half2_dir = output_base / 'half2'

    # Load Z-scores
    with open(half1_dir / 'z_scores.json', 'r') as f:
        z1_data = json.load(f)
    with open(half2_dir / 'z_scores.json', 'r') as f:
        z2_data = json.load(f)

    z1 = np.array(z1_data['z_scores'])
    z2 = np.array(z2_data['z_scores'])

    # Get valid paired values
    valid_mask = (~np.isnan(z1)) & (~np.isnan(z2))
    z1_valid = z1[valid_mask]
    z2_valid = z2[valid_mask]

    if len(z1_valid) > 0:
        correlation = np.corrcoef(z1_valid, z2_valid)[0, 1]
    else:
        correlation = np.nan

    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(z1_valid, z2_valid, alpha=0.2, s=5, c='steelblue')

    lims = [min(z1_valid.min(), z2_valid.min()), max(z1_valid.max(), z2_valid.max())]
    ax1.plot(lims, lims, 'r--', linewidth=2, label='y=x', alpha=0.7)

    ax1.set_xlabel('Z-scores (Half 1)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Z-scores (Half 2)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Split-Half Reliability\\nCorrelation: r = {correlation:.3f}',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Difference distribution
    ax2 = axes[1]
    diff = z1_valid - z2_valid
    ax2.hist(diff, bins=100, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero', alpha=0.7)
    ax2.axvline(np.mean(diff), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(diff):.3f}', alpha=0.7)

    ax2.set_xlabel('Z-score Difference (Half1 - Half2)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title(f'Difference Distribution\\nMean: {np.mean(diff):.3f}, Std: {np.std(diff):.3f}',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    comparison_path = output_base / 'split_half_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save statistics
    stats = {
        'correlation': float(correlation),
        'mean_difference': float(np.mean(diff)),
        'std_difference': float(np.std(diff)),
        'n_compared_values': int(len(z1_valid)),
        'interpretation': f'Correlation of {correlation:.3f} indicates {"high" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.5 else "low"} split-half reliability'
    }

    with open(output_base / 'comparison_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"    Correlation: r = {correlation:.3f}")
    print(f"    Mean difference: {np.mean(diff):.3f}")
    print(f"    Std difference: {np.std(diff):.3f}")
    print(f"    Saved comparison to: {comparison_path}")

    return correlation


def main():
    print("="*70)
    print("AUTOMATED SPLIT-HALF RELIABILITY ANALYSIS")
    print("="*70)
    print()

    neuron_data_path = Path('neuron_data.json')
    output_base = Path('significance_analysis_split')

    # Step 1: Load and split data
    print("STEP 1: Loading and splitting data")
    print("-"*70)
    neuron_data = load_neuron_data(neuron_data_path)
    total_spikes = sum(len(spikes) for spikes in neuron_data)
    print(f"  Total neurons: {len(neuron_data)}")
    print(f"  Total spikes: {total_spikes:,}")

    half1_data, half2_data = split_neuron_data_randomly(neuron_data, seed=42)
    half1_spikes = sum(len(spikes) for spikes in half1_data)
    half2_spikes = sum(len(spikes) for spikes in half2_data)

    print(f"  Half 1: {half1_spikes:,} spikes ({100*half1_spikes/total_spikes:.1f}%)")
    print(f"  Half 2: {half2_spikes:,} spikes ({100*half2_spikes/total_spikes:.1f}%)")

    # Create directories
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / 'half1').mkdir(exist_ok=True)
    (output_base / 'half2').mkdir(exist_ok=True)

    # Save split data
    half1_json = output_base / 'neuron_data_half1.json'
    half2_json = output_base / 'neuron_data_half2.json'
    save_neuron_data(half1_data, half1_json)
    save_neuron_data(half2_data, half2_json)

    # Step 2: Analyze Half 1
    print("\n" + "="*70)
    print("STEP 2: Analyzing Half 1")
    print("-"*70)
    hist1_path = run_spike_triggered_average(half1_json, output_base / 'half1')
    run_significance_analysis(hist1_path, half1_json, output_base / 'half1')

    # Step 3: Analyze Half 2
    print("\n" + "="*70)
    print("STEP 3: Analyzing Half 2")
    print("-"*70)
    hist2_path = run_spike_triggered_average(half2_json, output_base / 'half2')
    run_significance_analysis(hist2_path, half2_json, output_base / 'half2')

    # Step 4: Compare results
    print("\n" + "="*70)
    print("STEP 4: Comparing Results")
    print("-"*70)
    correlation = create_comparison_analysis(output_base)

    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nOutput location: {output_base.absolute()}")
    print(f"\nSplit-half correlation: r = {correlation:.3f}")
    print(f"  {abs(correlation) > 0.7 and 'High' or abs(correlation) > 0.5 and 'Moderate' or 'Low'} reliability")
    print("\nGenerated files:")
    print(f"  - {output_base}/half1/  (Full analysis for first half)")
    print(f"  - {output_base}/half2/  (Full analysis for second half)")
    print(f"  - {output_base}/split_half_comparison.png")
    print(f"  - {output_base}/comparison_statistics.json")
    print("="*70)


if __name__ == "__main__":
    main()
