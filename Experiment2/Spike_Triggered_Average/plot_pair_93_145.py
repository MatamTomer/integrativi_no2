"""
Plot the spike-triggered average histogram for the pair 93 -> 145,
with the null-hypothesis mean ± 2 std envelope overlaid.
Output: Experiment2/Presentation_Pictures/sta_93_145.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

HERE        = Path(__file__).parent
EXPERIMENT2 = HERE.parent
OUT_DIR     = EXPERIMENT2 / 'Presentation_Pictures'
OUT_DIR.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
hist_d  = json.load(open(HERE          / 'histogram_data.json'))
null_m  = json.load(open(EXPERIMENT2   / 'Null_Hypothesis' / 'null_histograms.json'))
null_s  = json.load(open(EXPERIMENT2   / 'Null_Hypothesis' / 'null_std.json'))

neuron_ids = hist_d['neuron_ids']
bins       = np.array(hist_d['bins'])
histograms = np.array(hist_d['histograms'])
null_mean  = np.array(null_m['data'])
null_std   = np.array(null_s['data'])

TRIGGER, TARGET = 93, 145
i = neuron_ids.index(TRIGGER)
j = neuron_ids.index(TARGET)

counts = histograms[i, j]
nm     = null_mean[i, j]
ns     = null_std[i, j]

# ── Derived stats ─────────────────────────────────────────────────────────────
peak_bin   = bins[np.argmax(counts)]
peak_count = counts.max()
z_scores   = (counts - nm) / np.clip(ns, 1e-9, None)
peak_z     = z_scores.max()

print(f"Pair {TRIGGER} -> {TARGET}")
print(f"  Peak count : {peak_count}  at offset {peak_bin:+d} ms")
print(f"  Peak z-score: {peak_z:.1f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(8, 7),
                         gridspec_kw={'height_ratios': [3, 1.5]})

# — Top panel: raw histogram + null envelope —
ax = axes[0]
bar_colors = ['tomato' if b == peak_bin else 'steelblue' for b in bins]
ax.bar(bins, counts, width=0.85, color=bar_colors, edgecolor='white',
       linewidth=0.4, alpha=0.9, label='Observed counts', zorder=3)

ax.fill_between(bins, nm - 2*ns, nm + 2*ns,
                color='orange', alpha=0.35, label='Null ± 2 σ', zorder=2)
ax.plot(bins, nm, color='darkorange', linewidth=1.4,
        linestyle='--', label='Null mean', zorder=4)

ax.axvline(0,  color='black',  linewidth=1.0, linestyle=':',  alpha=0.7, label='t = 0')
ax.axvline(peak_bin, color='red', linewidth=1.2, linestyle='--',
           alpha=0.8, label=f'Peak at {peak_bin:+d} ms')

ax.set_title(f'Spike-Triggered Average  —  Neuron {TRIGGER} → Neuron {TARGET}',
             fontsize=13, fontweight='bold', pad=10)
ax.set_ylabel('Spike count', fontsize=11)
ax.set_xticks(bins[::2])
ax.tick_params(labelsize=9)
ax.legend(fontsize=8, loc='upper left')

textstr = (f'Peak count: {peak_count}\n'
           f'Peak offset: {peak_bin:+d} ms\n'
           f'Peak z-score: {peak_z:.1f}')
ax.text(0.98, 0.97, textstr, transform=ax.transAxes,
        fontsize=8.5, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                  edgecolor='gray', alpha=0.8))

# — Bottom panel: z-score per bin —
ax2 = axes[1]
z_colors = ['tomato' if z > 2 else 'steelblue' for z in z_scores]
ax2.bar(bins, z_scores, width=0.85, color=z_colors, edgecolor='white',
        linewidth=0.4, alpha=0.9, zorder=3)
ax2.axhline(2,  color='darkorange', linewidth=1.1, linestyle='--',
            alpha=0.8, label='z = 2')
ax2.axhline(-2, color='darkorange', linewidth=1.1, linestyle='--', alpha=0.8)
ax2.axvline(0,  color='black', linewidth=1.0, linestyle=':', alpha=0.7)
ax2.set_xlabel('Time lag  τ  (ms)', fontsize=11)
ax2.set_ylabel('z-score', fontsize=11)
ax2.set_xticks(bins[::2])
ax2.tick_params(labelsize=9)
ax2.legend(fontsize=8, loc='upper left')

plt.tight_layout(h_pad=2.0)
out = OUT_DIR / 'sta_93_145.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
