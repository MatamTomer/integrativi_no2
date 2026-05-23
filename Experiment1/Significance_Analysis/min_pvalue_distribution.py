import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
print("Loading data...")

with open('../Spike_Triggered_Average/histogram_data.json', 'r') as f:
    hist_data = json.load(f)

with open('../Null_Hypothesis/p_high.json', 'r') as f:
    p_high_data = json.load(f)

with open('../Null_Hypothesis/p_low.json', 'r') as f:
    p_low_data = json.load(f)

observed   = np.array(hist_data['histograms'])   # (N, N, 20)
p_high     = np.array(p_high_data['data'])        # (N, N, 20)
p_low      = np.array(p_low_data['data'])          # (N, N, 20)
neuron_ids = hist_data['neuron_ids']
N          = len(neuron_ids)

# ─────────────────────────────────────────────
# Collect min p-values for valid pairs (>= 300 spikes)
# ─────────────────────────────────────────────
min_p_high  = []
min_p_low   = []
pair_labels = []

for i in range(N):
    for j in range(N):
        if i == j:
            continue
        if observed[i][j].sum() < 300:
            continue
        min_p_high.append(np.min(p_high[i][j]))
        min_p_low.append(np.min(p_low[i][j]))
        pair_labels.append((neuron_ids[i], neuron_ids[j]))

min_p_high = np.array(min_p_high)
min_p_low  = np.array(min_p_low)
n_pairs    = len(min_p_high)

print(f"  Valid pairs (>= 300 spikes): {n_pairs}")

THRESHOLDS = [0.05, 0.01, 0.001]
for t in THRESHOLDS:
    n_exc = np.sum(min_p_high <= t)
    n_inh = np.sum(min_p_low  <= t)
    print(f"  p <= {t}: {n_exc} excitatory pairs, {n_inh} inhibitory pairs")

# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
BINS = 50

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    f'Distribution of Minimum p-values Across Neuron Pairs\n'
    f'({n_pairs} valid pairs with >= 300 correlated spikes)',
    fontsize=13, fontweight='bold'
)

# Top-left: p_high full range
ax = axes[0, 0]
ax.hist(min_p_high, bins=BINS, color='coral', edgecolor='black', linewidth=0.4, alpha=0.85)
for t in THRESHOLDS:
    ax.axvline(x=t, color='black', linestyle='--', linewidth=0.8, alpha=0.7, label=f'p={t}')
ax.set_xlabel('min p_high per pair', fontsize=10)
ax.set_ylabel('Number of pairs', fontsize=10)
ax.set_title('p_high (excitation) — full range', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)

# Top-right: p_low full range
ax = axes[0, 1]
ax.hist(min_p_low, bins=BINS, color='teal', edgecolor='black', linewidth=0.4, alpha=0.85)
for t in THRESHOLDS:
    ax.axvline(x=t, color='black', linestyle='--', linewidth=0.8, alpha=0.7, label=f'p={t}')
ax.set_xlabel('min p_low per pair', fontsize=10)
ax.set_ylabel('Number of pairs', fontsize=10)
ax.set_title('p_low (inhibition) — full range', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)

# Bottom-left: p_high zoomed to [0, 0.1]
ax = axes[1, 0]
ax.hist(min_p_high[min_p_high <= 0.1], bins=BINS,
        color='coral', edgecolor='black', linewidth=0.4, alpha=0.85)
for t in THRESHOLDS:
    n = np.sum(min_p_high <= t)
    ax.axvline(x=t, color='black', linestyle='--', linewidth=0.8, alpha=0.7,
               label=f'p={t}  (n={n})')
ax.set_xlim([0, 0.1])
ax.set_xlabel('min p_high per pair', fontsize=10)
ax.set_ylabel('Number of pairs', fontsize=10)
ax.set_title('p_high (excitation) — zoomed to [0, 0.1]', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)

# Bottom-right: p_low zoomed to [0, 0.1]
ax = axes[1, 1]
ax.hist(min_p_low[min_p_low <= 0.1], bins=BINS,
        color='teal', edgecolor='black', linewidth=0.4, alpha=0.85)
for t in THRESHOLDS:
    n = np.sum(min_p_low <= t)
    ax.axvline(x=t, color='black', linestyle='--', linewidth=0.8, alpha=0.7,
               label=f'p={t}  (n={n})')
ax.set_xlim([0, 0.1])
ax.set_xlabel('min p_low per pair', fontsize=10)
ax.set_ylabel('Number of pairs', fontsize=10)
ax.set_title('p_low (inhibition) — zoomed to [0, 0.1]', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('min_pvalue_distribution.pdf', bbox_inches='tight')
plt.close()
print("Saved: min_pvalue_distribution.pdf")

# ─────────────────────────────────────────────
# Save CSV
# ─────────────────────────────────────────────
with open('min_pvalues.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['trigger_neuron', 'response_neuron', 'min_p_high', 'min_p_low',
                     'sig_excitation_005', 'sig_inhibition_005',
                     'sig_excitation_001', 'sig_inhibition_001'])
    for idx, (ni, nj) in enumerate(pair_labels):
        ph = min_p_high[idx]
        pl = min_p_low[idx]
        writer.writerow([ni, nj, f'{ph:.6f}', f'{pl:.6f}',
                         int(ph <= 0.05), int(pl <= 0.05),
                         int(ph <= 0.01), int(pl <= 0.01)])
print("Saved: min_pvalues.csv")
print("\nDone.")
