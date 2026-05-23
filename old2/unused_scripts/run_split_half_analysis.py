"""
Split-half analysis: Split neuron data randomly in half, run significance analysis on each half separately.
"""
import json
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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

        # Create random mask for 50/50 split
        mask = np.random.rand(n_spikes) < 0.5

        # Split spikes
        half1_spikes = [spike for i, spike in enumerate(neuron_spikes) if mask[i]]
        half2_spikes = [spike for i, spike in enumerate(neuron_spikes) if not mask[i]]

        half1_data.append(half1_spikes)
        half2_data.append(half2_spikes)

    return half1_data, half2_data


def save_neuron_data(neuron_data, output_path):
    """Save neuron data to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(neuron_data, f)


def create_comparison_plots(output_base):
    """Create comparison plots between the two halves."""
    half1_dir = output_base / 'half1'
    half2_dir = output_base / 'half2'

    # Load Z-scores from both halves
    with open(half1_dir / 'z_scores.json', 'r') as f:
        z1_data = json.load(f)
    with open(half2_dir / 'z_scores.json', 'r') as f:
        z2_data = json.load(f)

    z1 = np.array(z1_data['z_scores'])
    z2 = np.array(z2_data['z_scores'])

    # Get valid (non-NaN) paired values
    valid_mask = (~np.isnan(z1)) & (~np.isnan(z2))
    z1_valid = z1[valid_mask]
    z2_valid = z2[valid_mask]

    # Compute correlation
    if len(z1_valid) > 0:
        correlation = np.corrcoef(z1_valid, z2_valid)[0, 1]
    else:
        correlation = np.nan

    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(z1_valid, z2_valid, alpha=0.2, s=5, c='steelblue')

    # Add y=x line
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
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference', alpha=0.7)
    ax2.axvline(np.mean(diff), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(diff):.3f}', alpha=0.7)

    ax2.set_xlabel('Z-score Difference (Half1 - Half2)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title(f'Distribution of Differences\\nMean: {np.mean(diff):.3f}, Std: {np.std(diff):.3f}',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    comparison_path = output_base / 'split_half_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save comparison statistics
    stats = {
        'correlation': float(correlation),
        'mean_difference': float(np.mean(diff)),
        'std_difference': float(np.std(diff)),
        'n_compared_values': int(len(z1_valid)),
        'interpretation': 'High correlation indicates reliable measurements across data splits'
    }

    stats_path = output_base / 'comparison_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    return correlation, np.mean(diff), np.std(diff)


def main():
    neuron_data_path = Path('neuron_data.json')
    output_base = Path('significance_analysis_split')

    print("="*70)
    print("SPLIT-HALF RELIABILITY ANALYSIS")
    print("="*70)
    print()
    print("This script will:")
    print("  1. Split each neuron's spikes randomly into two halves (50/50)")
    print("  2. Run full significance analysis on each half separately")
    print("  3. Compare results to assess reliability")
    print()

    # Load original data
    print("Step 1: Loading original neuron data...")
    neuron_data = load_neuron_data(neuron_data_path)
    total_spikes = sum(len(spikes) for spikes in neuron_data)
    print(f"  Neurons: {len(neuron_data)}")
    print(f"  Total spikes: {total_spikes:,}")
    print()

    # Split data randomly
    print("Step 2: Splitting data into two random halves...")
    half1_data, half2_data = split_neuron_data_randomly(neuron_data, seed=42)

    half1_spikes = sum(len(spikes) for spikes in half1_data)
    half2_spikes = sum(len(spikes) for spikes in half2_data)

    print(f"  Half 1: {half1_spikes:,} spikes ({100*half1_spikes/total_spikes:.1f}%)")
    print(f"  Half 2: {half2_spikes:,} spikes ({100*half2_spikes/total_spikes:.1f}%)")
    print()

    # Create output directories
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / 'half1').mkdir(exist_ok=True)
    (output_base / 'half2').mkdir(exist_ok=True)

    # Save split datasets
    print("Step 3: Saving split datasets...")
    half1_path = output_base / 'neuron_data_half1.json'
    half2_path = output_base / 'neuron_data_half2.json'

    save_neuron_data(half1_data, half1_path)
    save_neuron_data(half2_data, half2_path)
    print(f"  Saved: {half1_path}")
    print(f"  Saved: {half2_path}")
    print()

    print("="*70)
    print("NEXT STEPS (manual for now):")
    print("="*70)
    print()
    print("Run these commands to analyze each half:")
    print()
    print("1. Analyze Half 1:")
    print("   - Copy neuron_data_half1.json to neuron_data.json")
    print("   - Run: python spike_triggered_average.py")
    print("   - Run: python compute_significance_combined.py")
    print("   - Move results to significance_analysis_split/half1/")
    print()
    print("2. Analyze Half 2:")
    print("   - Copy neuron_data_half2.json to neuron_data.json")
    print("   - Run: python spike_triggered_average.py")
    print("   - Run: python compute_significance_combined.py")
    print("   - Move results to significance_analysis_split/half2/")
    print()
    print("3. Compare results:")
    print("   - Run: python -c \"from run_split_half_analysis import create_comparison_plots; ")
    print("          create_comparison_plots(Path('significance_analysis_split'))\"")
    print()
    print("="*70)


if __name__ == "__main__":
    main()
