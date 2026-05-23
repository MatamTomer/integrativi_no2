import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading Z-scores from significance analysis...")
with open('significance_analysis/z_scores.json', 'r') as f:
    data = json.load(f)

z_scores = np.array(data['z_scores'])
bins = np.array(data['bins'])

print(f"Z-scores shape: {z_scores.shape}")
print(f"Bins: {bins}")

N = z_scores.shape[0]
print(f"Number of neurons: {N}")

# Define the bins we need
bins_1_to_5 = [1, 2, 3, 4, 5]
bins_6_to_10 = [6, 7, 8, 9, 10]

# Find the indices for these bins
indices_1_to_5 = [np.where(bins == b)[0][0] for b in bins_1_to_5]
indices_6_to_10 = [np.where(bins == b)[0][0] for b in bins_6_to_10]

print(f"\nBins 1-5 indices: {indices_1_to_5}")
print(f"Bins 6-10 indices: {indices_6_to_10}")

# Compute D_ij matrix
print("\nComputing D_ij matrix...")
print("D_ij = min(Z_ij at bins 1-5) + median(Z_ij at bins 6-10)")

D_ij = np.full((N, N), np.nan)

for i in range(N):
    for j in range(N):
        z_ij = z_scores[i][j]

        # Skip if all NaN (pair was skipped)
        if np.all(np.isnan(z_ij)):
            continue

        # Get Z-scores for the relevant bins
        z_1_to_5 = z_ij[indices_1_to_5]
        z_6_to_10 = z_ij[indices_6_to_10]

        # Compute D_ij = min(early bins) + median(late bins)
        min_early = np.nanmin(z_1_to_5)
        median_late = np.nanmedian(z_6_to_10)

        D_ij[i][j] = min_early + median_late

print("D_ij matrix computed!")

# Count valid values
valid_count = np.sum(~np.isnan(D_ij))
print(f"Valid pairs: {valid_count} out of {N*N}")

# Statistics
valid_values = D_ij[~np.isnan(D_ij)]
print(f"\nD_ij statistics:")
print(f"  Min: {np.min(valid_values):.3f}")
print(f"  Max: {np.max(valid_values):.3f}")
print(f"  Mean: {np.mean(valid_values):.3f}")
print(f"  Median: {np.median(valid_values):.3f}")
print(f"  Std: {np.std(valid_values):.3f}")

# Find most inhibitory connections (most negative D_ij)
print("\nTop 10 most inhibitory connections (most negative D_ij):")
flat_indices = np.argsort(D_ij.flatten())
for idx in range(10):
    flat_idx = flat_indices[idx]
    if np.isnan(D_ij.flatten()[flat_idx]):
        break
    i = flat_idx // N
    j = flat_idx % N
    print(f"  Neuron {i+1} -> Neuron {j+1}: D_ij = {D_ij[i][j]:.3f}")

# Save to JSON
print("\nSaving D_ij matrix to JSON...")
output_data = {
    'D_ij_matrix': D_ij.tolist(),
    'neuron_ids': list(range(1, N+1)),
    'description': 'Inhibitory activity matrix: D_ij = min(Z_ij at bins 1-5ms) + median(Z_ij at bins 6-10ms)',
    'formula': 'D_ij = min_{k=1..5} Z_ij(k) + median_{k=6..10} Z_ij(k)',
    'statistics': {
        'min': float(np.min(valid_values)),
        'max': float(np.max(valid_values)),
        'mean': float(np.mean(valid_values)),
        'median': float(np.median(valid_values)),
        'std': float(np.std(valid_values)),
        'valid_pairs': int(valid_count)
    }
}

with open('dij_matrix.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print("Saved to dij_matrix.json")

# Create heatmap visualization
print("\nCreating heatmap visualization...")

fig, ax = plt.subplots(figsize=(12, 10))

# Create heatmap
neuron_labels = [str(i) for i in range(1, N+1)]
sns.heatmap(D_ij,
            xticklabels=neuron_labels,
            yticklabels=neuron_labels,
            cmap='RdBu_r',  # Red for positive (excitatory-like), Blue for negative (inhibitory)
            center=0,
            cbar_kws={'label': 'D_ij (Inhibitory Activity)'},
            ax=ax,
            vmin=np.percentile(valid_values, 5),
            vmax=np.percentile(valid_values, 95))

ax.set_xlabel('Neuron j (Post-synaptic)', fontsize=12)
ax.set_ylabel('Neuron i (Pre-synaptic)', fontsize=12)
ax.set_title('Inhibitory Activity Matrix (D_ij)\nD_ij = min(Z_ij at 1-5ms) + median(Z_ij at 6-10ms)\n'
             'Negative values indicate inhibition',
             fontsize=14)

plt.tight_layout()
plt.savefig('dij_matrix_heatmap.png', dpi=150)
print("Saved heatmap to dij_matrix_heatmap.png")
plt.close()

print("\nDone!")
