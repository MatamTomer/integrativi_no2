import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_z_scores():
    """Load Z-scores from the significance analysis."""
    with open('significance_analysis_filtered/z_scores.json', 'r') as f:
        data = json.load(f)

    z_scores = np.array(data['z_scores'])
    bins = np.array(data['bins'])
    neuron_ids = data['neuron_ids']

    return z_scores, bins, neuron_ids


def compute_sij_matrix(z_scores, bins):
    """
    Compute the S_ij matrix where:
    S_ij = max_{k=1..5} Z_ij(k) - median_{k=6..10} Z_ij(k)

    where k=1..5 corresponds to bins [1, 2, 3, 4, 5]ms
    and k=6..10 corresponds to bins [6, 7, 8, 9, 10]ms

    Args:
        z_scores: 3D array (N, N, num_bins) of Z-scores
        bins: Array of bin values

    Returns:
        S_ij: 2D array (N, N) of S values
    """
    N = z_scores.shape[0]
    S_ij = np.full((N, N), np.nan)

    # Find indices for bins [1, 2, 3, 4, 5]ms and [6, 7, 8, 9, 10]ms
    bins_1_to_5 = [1, 2, 3, 4, 5]
    bins_6_to_10 = [6, 7, 8, 9, 10]

    # Get bin indices
    indices_1_to_5 = [np.where(bins == b)[0][0] for b in bins_1_to_5]
    indices_6_to_10 = [np.where(bins == b)[0][0] for b in bins_6_to_10]

    print(f"Bins [1-5]ms map to indices: {indices_1_to_5}")
    print(f"Bins [6-10]ms map to indices: {indices_6_to_10}")

    # Compute S_ij for each neuron pair
    for i in range(N):
        for j in range(N):
            z_ij = z_scores[i][j]

            # Skip if all NaN (skipped pair)
            if np.all(np.isnan(z_ij)):
                continue

            # Get Z values for the two ranges
            z_1_to_5 = z_ij[indices_1_to_5]
            z_6_to_10 = z_ij[indices_6_to_10]

            # Check if we have valid values
            if not np.all(np.isnan(z_1_to_5)) and not np.all(np.isnan(z_6_to_10)):
                max_early = np.nanmax(z_1_to_5)
                median_late = np.nanmedian(z_6_to_10)

                S_ij[i][j] = max_early - median_late

    return S_ij


def create_heatmap(S_ij, neuron_ids, output_path):
    """Create a heatmap visualization of the S_ij matrix."""
    N = len(neuron_ids)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    # Use a diverging colormap centered at 0
    vmax = np.nanmax(np.abs(S_ij))
    im = ax.imshow(S_ij, cmap='RdBu_r', aspect='auto',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('S_ij = max(Z at 1-5ms) - median(Z at 6-10ms)', fontsize=10)

    # Set ticks
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(neuron_ids, fontsize=8, rotation=90)
    ax.set_yticklabels(neuron_ids, fontsize=8)

    # Labels
    ax.set_xlabel('Neuron j (other)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Neuron i (trigger)', fontsize=12, fontweight='bold')
    ax.set_title('S_ij Matrix: Early Peak vs Late Baseline Difference',
                 fontsize=14, fontweight='bold', pad=20)

    # Add grid
    ax.set_xticks(np.arange(N) - 0.5, minor=True)
    ax.set_yticks(np.arange(N) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved heatmap to: {output_path}")


def save_sij_json(S_ij, neuron_ids, output_path):
    """Save S_ij matrix to JSON file."""
    data = {
        'neuron_ids': neuron_ids,
        'S_ij': S_ij.tolist(),
        'shape': S_ij.shape,
        'formula': 'S_ij = max(Z_ij at bins [1,2,3,4,5]ms) - median(Z_ij at bins [6,7,8,9,10]ms)',
        'description': 'S_ij[i][j] measures early peak strength relative to late baseline for neuron pair (i,j)'
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved S_ij matrix to: {output_path}")


def main():
    output_dir = Path('significance_analysis_filtered')
    json_path = output_dir / 'sij_matrix.json'
    heatmap_path = output_dir / 'sij_matrix_heatmap.png'

    print("="*70)
    print("COMPUTING S_ij MATRIX")
    print("="*70)
    print("Formula: S_ij = max(Z_ij at 1-5ms) - median(Z_ij at 6-10ms)")
    print()

    # Load Z-scores
    print("Loading Z-scores...")
    z_scores, bins, neuron_ids = load_z_scores()
    N = len(neuron_ids)

    print(f"  Loaded {N} neurons")
    print(f"  Z-score matrix shape: {z_scores.shape}")
    print(f"  Bins: {bins.tolist()}")
    print()

    # Compute S_ij matrix
    print("Computing S_ij matrix...")
    S_ij = compute_sij_matrix(z_scores, bins)
    print(f"  Done!")
    print()

    # Statistics
    valid_values = S_ij[~np.isnan(S_ij)]
    print("Statistics (excluding NaN/skipped pairs):")
    print(f"  Valid pairs: {len(valid_values)} / {N*N}")
    print(f"  Min S_ij: {np.min(valid_values):.3f}")
    print(f"  Max S_ij: {np.max(valid_values):.3f}")
    print(f"  Mean S_ij: {np.mean(valid_values):.3f}")
    print(f"  Median S_ij: {np.median(valid_values):.3f}")
    print(f"  Std S_ij: {np.std(valid_values):.3f}")
    print()

    # Find top pairs
    print("Top 5 pairs with highest S_ij (strongest early peak):")
    flat_indices = np.argsort(S_ij.ravel())[::-1][:5]
    for rank, flat_idx in enumerate(flat_indices, 1):
        i, j = np.unravel_index(flat_idx, S_ij.shape)
        if not np.isnan(S_ij[i, j]):
            print(f"  {rank}. Neuron {neuron_ids[i]} -> {neuron_ids[j]}: S_ij = {S_ij[i, j]:.3f}")
    print()

    # Save outputs
    print("Saving outputs...")
    save_sij_json(S_ij, neuron_ids, json_path)
    create_heatmap(S_ij, neuron_ids, heatmap_path)

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"S_ij matrix computed successfully!")
    print(f"Output files:")
    print(f"  - JSON: {json_path}")
    print(f"  - Heatmap: {heatmap_path}")
    print()
    print("Interpretation:")
    print("  - Positive S_ij: Strong early peak (1-5ms) relative to late baseline (6-10ms)")
    print("  - Negative S_ij: Early activity lower than late baseline")
    print("  - High S_ij: Neuron j fires strongly shortly after neuron i")
    print("="*70)


if __name__ == "__main__":
    main()
