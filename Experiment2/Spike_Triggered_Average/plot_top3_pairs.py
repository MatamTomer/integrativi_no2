"""
Find three neuron pairs with strong, distinct-lag functional connectivity
(largest |z-score| vs. the jitter null, each at a different nonzero peak
offset so the figure shows a variety of lags rather than three pairs that
all happen to peak at 0ms) and plot each one's histogram + null ±2σ envelope
as a single publication-quality 3-panel figure.

Artifact guards (both needed -- see the debugging history in this project):
  1. Reject any pair where a single bin holds >50% of all its spikes -- a
     real synapse has jitter and spreads over several adjacent ms; a
     hardware/sorting duplicate does not.
  2. Reject any unit implicated in many such pairs at once (a "hub" -- e.g.
     one high-amplitude unit bleeding into many electrodes), not just its
     individual flagged pairs.

Data: full dataset (Experiment2/Spike_Triggered_Average/histogram_data.json,
Experiment2/Null_Hypothesis/null_histograms.json + null_std.json).

Output: Experiment2/Presentation_Pictures/top3_pairs.png (+ .pdf/.svg)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

HERE        = Path(__file__).parent
EXPERIMENT2 = HERE.parent
OUT_DIR     = EXPERIMENT2 / 'Presentation_Pictures'
OUT_DIR.mkdir(exist_ok=True)

MIN_SPIKES = 300   # require this many trigger-window spikes for a reliable z-score
N_PAIRS    = 3

# ── Load data (full dataset) ───────────────────────────────────────────────────
print("Loading histograms and null hypothesis...")
hist_d = json.load(open(HERE        / 'histogram_data.json'))
null_m = json.load(open(EXPERIMENT2 / 'Null_Hypothesis' / 'null_histograms.json'))
null_s = json.load(open(EXPERIMENT2 / 'Null_Hypothesis' / 'null_std.json'))

neuron_ids = hist_d['neuron_ids']
bins       = np.array(hist_d['bins'])
histograms = np.array(hist_d['histograms'], dtype=np.float64)   # (N, N, 21)
null_mean  = np.array(null_m['data'])
null_std   = np.array(null_s['data'])
N          = len(neuron_ids)

with np.errstate(invalid='ignore', divide='ignore'):
    z_scores = np.where(null_std > 0, (histograms - null_mean) / null_std, 0.0)

totals = histograms.sum(axis=2)   # (N, N)

# Peak lag and z-score per pair, searched over the whole ±10ms window
PEAK_LAG_MIN, PEAK_LAG_MAX = 1, 8   # positive (causal) lags only; avoid the
                                     # extreme ±9/10ms edge bins (few samples)
peak_idx_full = np.argmax(np.abs(z_scores), axis=2)                       # (N, N)
peak_bin_full = bins[peak_idx_full]                                       # (N, N)
peak_z_full   = np.take_along_axis(
    z_scores, peak_idx_full[:, :, None], axis=2)[:, :, 0]                 # (N, N)

# ── Reject pairs contaminated by cross-talk / duplicate-sorted units. The
#    tell isn't WHICH bin dominates (that varies: 0ms, +1ms, +2ms, +4ms have
#    all shown up) -- it's that a single, essentially-zero-jitter bin holds
#    almost all the spikes. Real synaptic transmission has jitter and spreads
#    over several adjacent ms; a hardware/sorting duplicate does not. ──
SINGLE_BIN_FRAC_MAX = 0.5   # reject if any one bin holds >50% of all spikes
max_bin_frac = np.max(histograms, axis=2) / np.maximum(totals, 1)
artifact     = max_bin_frac > SINGLE_BIN_FRAC_MAX
n_artifact_units = len(set(np.where(artifact.any(axis=1) | artifact.any(axis=0))[0]))
print(f"Flagged {int(artifact.sum())} pairs ({n_artifact_units} units involved) as "
      f"likely cross-talk/duplicate-unit artifacts (single-bin >{int(SINGLE_BIN_FRAC_MAX*100)}% "
      f"of all spikes) -- excluding.")

# A single unit implicated in an unusually large number of these artifact
# pairs is itself a bad/hub channel (e.g. one high-amplitude unit bleeding
# into many electrodes) -- exclude it entirely, not just its individual pairs.
HUB_PARTNER_MAX = 3
partner_count = artifact.sum(axis=1) + artifact.sum(axis=0)   # per-unit degree in the artifact graph
hub_units = np.where(partner_count > HUB_PARTNER_MAX)[0]
if len(hub_units):
    print(f"Excluding {len(hub_units)} hub unit(s) implicated in >{HUB_PARTNER_MAX} "
          f"artifact pairs each: "
          + ", ".join(f"{neuron_ids[u]} ({partner_count[u]} partners)" for u in hub_units))
    artifact[hub_units, :] = True
    artifact[:, hub_units] = True

mask = np.ones((N, N), dtype=bool)
np.fill_diagonal(mask, False)
mask &= totals >= MIN_SPIKES
mask &= ~artifact
mask &= (peak_bin_full >= PEAK_LAG_MIN) & (peak_bin_full <= PEAK_LAG_MAX)

scores = np.where(mask, np.abs(peak_z_full), -np.inf)
order  = np.argsort(scores.ravel())[::-1]

top_pairs       = []
seen_undirected = set()   # dedupe mirror pairs (i,j)/(j,i)
seen_lags       = set()   # require each chosen pair to peak at a different lag
for flat_idx in order:
    i, j = np.unravel_index(flat_idx, scores.shape)
    if scores[i, j] == -np.inf:
        break
    lag = int(peak_bin_full[i, j])
    if lag in seen_lags:
        continue
    key = frozenset((i, j))
    if key in seen_undirected:
        continue
    seen_lags.add(lag)
    seen_undirected.add(key)
    top_pairs.append((i, j))
    if len(top_pairs) == N_PAIRS:
        break

print(f"\nTop {N_PAIRS} pairs, each at a different peak lag "
      f"(min {MIN_SPIKES} spikes, artifact-filtered):")
for i, j in top_pairs:
    print(f"  Neuron {neuron_ids[i]:>4} -> Neuron {neuron_ids[j]:>4}   "
          f"peak z = {peak_z_full[i, j]:+7.1f}  at offset {int(peak_bin_full[i, j]):+3d} ms   "
          f"total spikes = {int(totals[i, j]):,}")

# ── Plot: one row, one panel per pair ─────────────────────────────────────────
print("\nPlotting...")
fig, axes = plt.subplots(1, N_PAIRS, figsize=(18, 5.5), sharey=False)

panel_labels = ['A', 'B', 'C']

for ax, (i, j), label in zip(axes, top_pairs, panel_labels):
    counts = histograms[i, j]
    nm     = null_mean[i, j]
    ns     = null_std[i, j]
    z      = z_scores[i, j]

    peak_idx   = int(peak_idx_full[i, j])
    peak_bin   = int(bins[peak_idx])
    peak_count = counts[peak_idx]
    peak_z     = z[peak_idx]

    bar_colors = ['tomato' if b == peak_bin else 'steelblue' for b in bins]
    ax.bar(bins, counts, width=0.85, color=bar_colors, edgecolor='white',
           linewidth=0.4, alpha=0.9, zorder=3,
           label='Observed counts' if ax is axes[0] else None)

    # Null ± 2σ as a separate rectangle per bin -- including the mean marker,
    # which must NOT be a single connected line across bins (that reads as a
    # ribbon regardless of the boxes being discrete underneath it).
    BAR_W = 0.85
    for k, b in enumerate(bins):
        lo, hi = nm[k] - 2 * ns[k], nm[k] + 2 * ns[k]
        ax.add_patch(Rectangle(
            (b - BAR_W / 2, lo), BAR_W, hi - lo,
            facecolor='gold', edgecolor='none', alpha=0.55, zorder=2,
            label='Null ± 2σ' if (ax is axes[0] and k == 0) else None,
        ))
        ax.hlines(nm[k], b - BAR_W / 2, b + BAR_W / 2,
                  color='darkorange', linewidth=1.8, zorder=4,
                  label='Null mean' if (ax is axes[0] and k == 0) else None)

    ax.axvline(0, color='black', linewidth=1.0, linestyle=':', alpha=0.7,
               label='t = 0' if ax is axes[0] else None)
    ax.axvline(peak_bin, color='red', linewidth=1.2, linestyle='--', alpha=0.8,
               label='Peak offset' if ax is axes[0] else None)

    ax.set_title(f'Neuron {neuron_ids[i]} → Neuron {neuron_ids[j]}',
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Time lag τ (ms)', fontsize=12)
    if ax is axes[0]:
        ax.set_ylabel('Spike count', fontsize=12)
    ax.set_xticks(bins[::2])
    ax.tick_params(labelsize=10)

    ax.text(0.03, 0.97, label, transform=ax.transAxes, fontsize=16,
            fontweight='bold', va='top', ha='left')

    textstr = (f'Peak count: {int(peak_count)}\n'
               f'Peak offset: {peak_bin:+d} ms\n'
               f'Peak z-score: {peak_z:+.1f}')
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
            fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.85))

fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
           ncol=5, bbox_to_anchor=(0.5, 1.06), fontsize=10, frameon=False)

plt.tight_layout()
for ext in ('png', 'pdf', 'svg'):
    out = OUT_DIR / f'top3_pairs.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")

print("Done.")
