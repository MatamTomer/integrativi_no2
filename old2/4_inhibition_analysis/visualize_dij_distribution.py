import json
import numpy as np
import matplotlib.pyplot as plt

print("Loading D_ij matrix...")
with open('dij_matrix.json', 'r') as f:
    data = json.load(f)

D_ij = np.array(data['D_ij_matrix'])
stats = data['statistics']

# Get valid values (non-NaN)
valid_values = D_ij[~np.isnan(D_ij)]

print(f"Valid D_ij values: {len(valid_values)}")
print(f"Statistics:")
print(f"  Min: {stats['min']:.3f}")
print(f"  Max: {stats['max']:.3f}")
print(f"  Mean: {stats['mean']:.3f}")
print(f"  Median: {stats['median']:.3f}")
print(f"  Std: {stats['std']:.3f}")

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Full histogram
ax1 = axes[0, 0]
counts, bins, patches = ax1.hist(valid_values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero (neutral)')
ax1.axvline(stats['mean'], color='orange', linestyle='--', linewidth=2, label=f'Mean ({stats["mean"]:.2f})')
ax1.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f'Median ({stats["median"]:.2f})')
ax1.set_xlabel('D_ij Value', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Distribution of D_ij Values (Full Range)', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Cumulative distribution
ax2 = axes[0, 1]
sorted_values = np.sort(valid_values)
cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values) * 100
ax2.plot(sorted_values, cumulative, linewidth=2, color='darkblue')
ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
ax2.axhline(50, color='gray', linestyle=':', linewidth=1)
ax2.set_xlabel('D_ij Value', fontsize=12)
ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
ax2.set_title('Cumulative Distribution of D_ij', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: Percentile breakdown
ax3 = axes[1, 0]
percentiles = [10, 25, 50, 75, 90, 95, 99]
percentile_values = [np.percentile(valid_values, p) for p in percentiles]
colors = ['darkred' if v < -10 else 'red' if v < 0 else 'lightblue' if v < 10 else 'darkblue'
          for v in percentile_values]
bars = ax3.bar(range(len(percentiles)), percentile_values, color=colors, edgecolor='black', alpha=0.7)
ax3.axhline(0, color='black', linestyle='-', linewidth=1)
ax3.set_xticks(range(len(percentiles)))
ax3.set_xticklabels([f'{p}th' for p in percentiles])
ax3.set_xlabel('Percentile', fontsize=12)
ax3.set_ylabel('D_ij Value', fontsize=12)
ax3.set_title('D_ij Values at Different Percentiles', fontsize=13)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, percentile_values)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}',
             ha='center', va='bottom' if height > 0 else 'top',
             fontsize=9, fontweight='bold')

# Subplot 4: Inhibitory vs Excitatory-like breakdown
ax4 = axes[1, 1]
strong_inhibitory = np.sum(valid_values < -10)
weak_inhibitory = np.sum((valid_values >= -10) & (valid_values < 0))
neutral_weak = np.sum((valid_values >= 0) & (valid_values < 10))
strong_excitatory = np.sum(valid_values >= 10)

categories = ['Strong\nInhibitory\n(< -10)', 'Weak\nInhibitory\n(-10 to 0)',
              'Neutral/Weak\n(0 to 10)', 'Strong\nExcitatory-like\n(≥ 10)']
counts_cat = [strong_inhibitory, weak_inhibitory, neutral_weak, strong_excitatory]
percentages = [c / len(valid_values) * 100 for c in counts_cat]
colors_cat = ['darkred', 'salmon', 'lightblue', 'darkblue']

bars = ax4.bar(range(len(categories)), percentages, color=colors_cat, edgecolor='black', alpha=0.7)
ax4.set_xticks(range(len(categories)))
ax4.set_xticklabels(categories, fontsize=10)
ax4.set_ylabel('Percentage (%)', fontsize=12)
ax4.set_title('Distribution of Connection Types', fontsize=13)
ax4.grid(True, alpha=0.3, axis='y')

# Add percentage labels on bars
for i, (bar, pct, cnt) in enumerate(zip(bars, percentages, counts_cat)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{pct:.1f}%\n({cnt})',
             ha='center', va='bottom',
             fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('dij_distribution.png', dpi=150)
print("\nSaved distribution plot to: dij_distribution.png")
plt.close()

# Print summary statistics
print("\n" + "="*60)
print("D_IJ DISTRIBUTION SUMMARY")
print("="*60)
print(f"\nPercentile breakdown:")
for p, v in zip(percentiles, percentile_values):
    print(f"  {p}th percentile: {v:.3f}")

print(f"\nConnection type breakdown:")
print(f"  Strong Inhibitory (D_ij < -10):     {strong_inhibitory:3d} ({strong_inhibitory/len(valid_values)*100:5.1f}%)")
print(f"  Weak Inhibitory (-10 <= D_ij < 0):  {weak_inhibitory:3d} ({weak_inhibitory/len(valid_values)*100:5.1f}%)")
print(f"  Neutral/Weak (0 <= D_ij < 10):      {neutral_weak:3d} ({neutral_weak/len(valid_values)*100:5.1f}%)")
print(f"  Strong Excitatory-like (D_ij >= 10):{strong_excitatory:3d} ({strong_excitatory/len(valid_values)*100:5.1f}%)")

print(f"\nInhibitory vs Excitatory-like:")
inhibitory_total = strong_inhibitory + weak_inhibitory
excitatory_total = strong_excitatory
print(f"  Total Inhibitory (D_ij < 0):        {inhibitory_total:3d} ({inhibitory_total/len(valid_values)*100:5.1f}%)")
print(f"  Total Excitatory-like (D_ij >= 10): {excitatory_total:3d} ({excitatory_total/len(valid_values)*100:5.1f}%)")
