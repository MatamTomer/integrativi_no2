import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
print("Loading data...")

with open('../Spike_Triggered_Average/histogram_data.json', 'r') as f:
    hist_data = json.load(f)

with open('../Null_Hypothesis/null_histograms.json', 'r') as f:
    null_mean_data = json.load(f)

with open('../Null_Hypothesis/null_std.json', 'r') as f:
    null_std_data = json.load(f)

observed   = np.array(hist_data['histograms'])       # (N, N, 21)
null_mean  = np.array(null_mean_data['data'])         # (N, N, 21)
null_std   = np.array(null_std_data['data'])           # (N, N, 21)
neuron_ids = hist_data['neuron_ids']
N          = len(neuron_ids)

# ─────────────────────────────────────────────
# Z-scores
# ─────────────────────────────────────────────
with np.errstate(invalid='ignore', divide='ignore'):
    z_scores = np.where(null_std > 0,
                        (observed - null_mean) / null_std,
                        0.0)   # (N, N, 21)

# Bin layout: indices 0–9 = lags -10..-1ms, index 10 = lag 0ms, indices 11–20 = lags +1..+10ms
# Positive lags k=1..5  (+1..+5ms)  -> indices 11..15
# Positive lags k=6..10 (+6..+10ms) -> indices 16..20
early  = z_scores[:, :, 11:16]   # (N, N, 5)
late   = z_scores[:, :, 16:21]   # (N, N, 5)

# S_ij = max over early lags − median over late lags
S = np.max(early, axis=2) - np.median(late, axis=2)   # (N, N)

# Mask self-pairs and pairs with < 300 correlated spikes
THRESHOLD = 50
for i in range(N):
    for j in range(N):
        if i == j or observed[i][j].sum() < 300:
            S[i][j] = np.nan

n_sig = int(np.sum(S > THRESHOLD))
print(f"  Pairs with S_ij > {THRESHOLD}: {n_sig}")

# ─────────────────────────────────────────────
# Heatmap
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(
    r'$S_{ij}$ = max Z(+1..+5ms) $-$ median Z(+6..+10ms)',
    fontsize=13, fontweight='bold'
)

tick_positions = np.arange(N)
tick_labels    = [str(nid) for nid in neuron_ids]

def style_ax(ax):
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel('Response neuron j', fontsize=10)
    ax.set_ylabel('Trigger neuron i', fontsize=10)

# Left: S_ij clipped to 2nd–98th percentile to avoid domination by extremes
valid_S_flat = S[~np.isnan(S)]
vmin = np.percentile(valid_S_flat, 2)
vmax = np.percentile(valid_S_flat, 98)

ax = axes[0]
im = ax.imshow(S, aspect='auto', cmap='RdYlBu_r', vmin=vmin, vmax=vmax, interpolation='none')
fig.colorbar(im, ax=ax, label='S_ij (clipped to 2nd–98th pct)')
ax.set_title('S_ij values (color clipped to 2nd–98th percentile)', fontsize=10, fontweight='bold')
style_ax(ax)

# Right: binary mask — S_ij > threshold
binary = (S > THRESHOLD).astype(float)
binary[np.isnan(S)] = np.nan

ax = axes[1]
im2 = ax.imshow(binary, aspect='auto', cmap='Reds', vmin=0, vmax=1, interpolation='none')
fig.colorbar(im2, ax=ax, label=f'S_ij > {THRESHOLD}')
ax.set_title(f'Significant pairs: S_ij > {THRESHOLD}  (n={n_sig})',
             fontsize=10, fontweight='bold')
style_ax(ax)

plt.tight_layout()
plt.savefig('sij_heatmap.pdf', bbox_inches='tight')
plt.close()
print("Saved: sij_heatmap.pdf")

# ─────────────────────────────────────────────
# Distribution of S_ij
# ─────────────────────────────────────────────
valid_S = S[~np.isnan(S)].flatten()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Distribution of S_ij values', fontsize=13, fontweight='bold')

# Left: full range
ax = axes[0]
ax.hist(valid_S, bins=60, color='steelblue', edgecolor='black', linewidth=0.4, alpha=0.85)
ax.axvline(x=THRESHOLD, color='red', linestyle='--', linewidth=1.2,
           label=f'threshold = {THRESHOLD}  (n={n_sig})')
ax.set_xlabel('S_ij', fontsize=10)
ax.set_ylabel('Number of pairs', fontsize=10)
ax.set_title('Full range', fontsize=10, fontweight='bold')
ax.legend(fontsize=9)

# Right: zoomed to positive values to see the tail
ax = axes[1]
pos_S = valid_S[valid_S > 0]
ax.hist(pos_S, bins=60, color='steelblue', edgecolor='black', linewidth=0.4, alpha=0.85)
ax.axvline(x=THRESHOLD, color='red', linestyle='--', linewidth=1.2,
           label=f'threshold = {THRESHOLD}  (n={n_sig})')
ax.set_xlabel('S_ij', fontsize=10)
ax.set_ylabel('Number of pairs', fontsize=10)
ax.set_title('Positive S_ij only', fontsize=10, fontweight='bold')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('sij_distribution.pdf', bbox_inches='tight')
plt.close()
print("Saved: sij_distribution.pdf")
print("Done.")
