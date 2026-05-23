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

observed   = np.array(hist_data['histograms'])
null_mean  = np.array(null_mean_data['data'])
null_std   = np.array(null_std_data['data'])
neuron_ids = hist_data['neuron_ids']
N          = len(neuron_ids)

# ─────────────────────────────────────────────
# Z-scores
# ─────────────────────────────────────────────
with np.errstate(invalid='ignore', divide='ignore'):
    z_scores = np.where(null_std > 0, (observed - null_mean) / null_std, 0.0)

# Bin layout: indices 0–9 = lags -10..-1ms, index 10 = lag 0ms, indices 11–20 = lags +1..+10ms
#   +1ms → index 11,  +2ms → index 12
#   +3ms → index 13,  +4ms → index 14,  +5ms → index 15
z_12 = z_scores[:, :, 11:13]   # +1ms, +2ms  (signal window)
z_35 = z_scores[:, :, 13:16]   # +3ms..+5ms  (local baseline)

# ─────────────────────────────────────────────
# New S-score (v2) — signed
#   E_S  = max(Z[+1], Z[+2])  - mean(Z[+3..+5])
#   I_S  = min(Z[+1], Z[+2])  - mean(Z[+3..+5])
#   S_ij = I_S  if |I_S| > |E_S|  else  E_S
#          positive = excitatory dominant
#          negative = inhibitory dominant
# ─────────────────────────────────────────────
baseline = np.mean(z_35, axis=2)
E_S = np.max(z_12, axis=2) - baseline
I_S = np.min(z_12, axis=2) - baseline
S   = np.where(np.abs(I_S) > np.abs(E_S), I_S, E_S)

# Mask self-pairs and pairs with < 300 correlated spikes
for i in range(N):
    for j in range(N):
        if i == j or observed[i][j].sum() < 300:
            S[i][j]   = np.nan
            E_S[i][j] = np.nan
            I_S[i][j] = np.nan

THRESHOLD = 10
n_excit = int(np.nansum(S >  THRESHOLD))
n_inhib = int(np.nansum(S < -THRESHOLD))
n_sig   = n_excit + n_inhib
print(f"  Excitatory pairs (S >  {THRESHOLD}): {n_excit}")
print(f"  Inhibitory pairs (S < -{THRESHOLD}): {n_inhib}")
print(f"  Total significant |S| > {THRESHOLD}: {n_sig}")

# ─────────────────────────────────────────────
# PDF 1 — Distribution of S, E_S, I_S
# ─────────────────────────────────────────────
valid_S  = S[~np.isnan(S)].flatten()
valid_ES = E_S[~np.isnan(E_S)].flatten()
valid_IS = I_S[~np.isnan(I_S)].flatten()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    'V2 S-score distribution\n'
    r'$E_S$ = max(Z[+1,+2ms]) − mean(Z[+3..+5ms])   '
    r'$I_S$ = min(Z[+1,+2ms]) − mean(Z[+3..+5ms])   '
    r'$S_{ij}$ = signed dominant',
    fontsize=11, fontweight='bold'
)

# Left: signed S
ax = axes[0]
ax.hist(valid_S, bins=80, color='mediumpurple', edgecolor='none', alpha=0.85)
ax.axvline(x=0,           color='black', linestyle='-',  linewidth=0.8)
ax.axvline(x= THRESHOLD,  color='red',   linestyle='--', linewidth=1.2,
           label=f'+{THRESHOLD}: n={n_excit}')
ax.axvline(x=-THRESHOLD,  color='red',   linestyle='--', linewidth=1.2,
           label=f'-{THRESHOLD}: n={n_inhib}')
ax.set_xlabel('S_ij  (+ excitatory, − inhibitory)', fontsize=10)
ax.set_ylabel('Number of pairs', fontsize=10)
ax.set_title('Signed S_ij', fontsize=10, fontweight='bold')
ax.legend(fontsize=9)

# Middle: E_S
ax = axes[1]
ax.hist(valid_ES, bins=80, color='coral', edgecolor='none', alpha=0.85)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('E_S_ij', fontsize=10)
ax.set_ylabel('Number of pairs', fontsize=10)
ax.set_title('Excitatory score E_S_ij', fontsize=10, fontweight='bold')

# Right: I_S
ax = axes[2]
ax.hist(valid_IS, bins=80, color='teal', edgecolor='none', alpha=0.85)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('I_S_ij', fontsize=10)
ax.set_ylabel('Number of pairs', fontsize=10)
ax.set_title('Inhibitory score I_S_ij', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('sij_v2_distribution.pdf', bbox_inches='tight')
plt.close()
print("Saved: sij_v2_distribution.pdf")

# ─────────────────────────────────────────────
# PDF 2 — Heatmaps
# ─────────────────────────────────────────────
tick_positions = np.arange(N)
tick_labels    = [str(nid) for nid in neuron_ids]

def style_ax(ax):
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=5)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=5)
    ax.set_xlabel('Response neuron j', fontsize=9)
    ax.set_ylabel('Trigger neuron i',  fontsize=9)

fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle('V2 S-score heatmaps  (+ excitatory, − inhibitory)',
             fontsize=13, fontweight='bold')

# Left: signed S (diverging colormap)
ax = axes[0]
valid_flat = S[~np.isnan(S)]
vlim = np.percentile(np.abs(valid_flat), 98)
im = ax.imshow(S, aspect='auto', cmap='RdBu_r',
               vmin=-vlim, vmax=vlim, interpolation='none')
fig.colorbar(im, ax=ax, label='S_ij  (+ excit., − inhib.)')
ax.set_title('Signed S_ij', fontsize=10, fontweight='bold')
style_ax(ax)

# Middle: |S| magnitude
ax = axes[1]
im2 = ax.imshow(np.abs(S), aspect='auto', cmap='YlOrRd',
                vmin=0, vmax=vlim, interpolation='none')
fig.colorbar(im2, ax=ax, label='|S_ij|')
ax.set_title('|S_ij| magnitude', fontsize=10, fontweight='bold')
style_ax(ax)

# Right: binary significant mask (|S| > threshold)
binary = np.where(np.isnan(S), np.nan,
                  np.where(np.abs(S) > THRESHOLD, 1.0, 0.0))
ax = axes[2]
im3 = ax.imshow(binary, aspect='auto', cmap='Reds', vmin=0, vmax=1,
                interpolation='none')
fig.colorbar(im3, ax=ax, label=f'|S_ij| > {THRESHOLD}')
ax.set_title(f'Significant pairs  |S| > {THRESHOLD}  (n={n_sig})',
             fontsize=10, fontweight='bold')
style_ax(ax)

plt.tight_layout()
plt.savefig('sij_v2_heatmap.pdf', bbox_inches='tight')
plt.close()
print("Saved: sij_v2_heatmap.pdf")

# ─────────────────────────────────────────────
# Save S_ij as CSV (all valid pairs)
# ─────────────────────────────────────────────
import csv
rows = []
for i in range(N):
    for j in range(N):
        if not np.isnan(S[i][j]):
            rows.append((neuron_ids[i], neuron_ids[j],
                         round(float(S[i][j]),   4),
                         round(float(E_S[i][j]), 4),
                         round(float(I_S[i][j]), 4)))

with open('sij_v2_scores.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['trigger_unit', 'response_unit', 'S_ij', 'E_S', 'I_S'])
    writer.writerows(rows)

print(f"Saved: sij_v2_scores.csv  ({len(rows)} pairs)")
print("Done.")
