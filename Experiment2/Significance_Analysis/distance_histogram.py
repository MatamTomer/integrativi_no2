import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THRESHOLD = 10
BIN_SIZE  = 100   # µm

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
print("Loading...")
sij   = pd.read_csv('sij_v2_scores.csv')
dmat  = pd.read_csv('../unit_distance_matrix.csv', index_col=0)
dmat.index   = dmat.index.astype(int)
dmat.columns = dmat.columns.astype(int)

sij['distance'] = sij.apply(
    lambda r: dmat.loc[int(r['trigger_unit']), int(r['response_unit'])], axis=1
)

print(f"  {len(sij)} pairs, distance range: "
      f"{sij['distance'].min():.0f}–{sij['distance'].max():.0f} µm")

# ─────────────────────────────────────────────
# Bin into 100µm bins
# ─────────────────────────────────────────────
max_dist  = sij['distance'].max()
bin_edges = np.arange(0, max_dist + BIN_SIZE, BIN_SIZE)
bin_labels = [f"{int(b)}–{int(b+BIN_SIZE)}" for b in bin_edges[:-1]]

sij['bin'] = pd.cut(sij['distance'], bins=bin_edges, right=False, labels=bin_labels)

all_counts   = sij.groupby('bin', observed=True).size()
excit_counts = sij[sij['S_ij'] >  THRESHOLD].groupby('bin', observed=True).size()
inhib_counts = sij[sij['S_ij'] < -THRESHOLD].groupby('bin', observed=True).size()

all_counts   = all_counts.reindex(bin_labels,   fill_value=0)
excit_counts = excit_counts.reindex(bin_labels, fill_value=0)
inhib_counts = inhib_counts.reindex(bin_labels, fill_value=0)

x = np.arange(len(bin_labels))

# Probability per bin (total significant / all valid)
total   = all_counts.replace(0, np.nan)
p_excit = (excit_counts / total).values
p_inhib = (inhib_counts / total).values
p_total = ((excit_counts + inhib_counts) / total).values

# Bin centre distances for curve fitting
bin_centres = bin_edges[:-1] + BIN_SIZE / 2

# ─────────────────────────────────────────────
# Helper: fit a line and return R²
# ─────────────────────────────────────────────
def fit_line(xdata, ydata):
    mask = np.isfinite(xdata) & np.isfinite(ydata)
    if mask.sum() < 2:
        return None, None, None
    coeffs = np.polyfit(xdata[mask], ydata[mask], 1)
    y_fit  = np.polyval(coeffs, xdata[mask])
    ss_res = np.sum((ydata[mask] - y_fit) ** 2)
    ss_tot = np.sum((ydata[mask] - ydata[mask].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return coeffs, xdata, np.polyval(coeffs, xdata)

# ─────────────────────────────────────────────
# PDF 1 — Counts + probability bar chart
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                         gridspec_kw={'height_ratios': [2, 1]})
fig.suptitle(
    f'Pair counts by inter-unit distance  (bin = {BIN_SIZE} µm)\n'
    f'Threshold |S| > {THRESHOLD}: '
    f'{int(excit_counts.sum())} excitatory, {int(inhib_counts.sum())} inhibitory',
    fontsize=13, fontweight='bold'
)

# Top: counts
ax = axes[0]
ax.bar(x, all_counts,   color='lightgray', edgecolor='gray',
       linewidth=0.4, label='All valid pairs')
ax.bar(x, excit_counts, color='coral',     edgecolor='none',
       alpha=0.85, label=f'Excitatory  S > {THRESHOLD}')
ax.bar(x, inhib_counts, color='steelblue', edgecolor='none',
       alpha=0.85, label=f'Inhibitory  S < −{THRESHOLD}',
       bottom=excit_counts)
ax.set_ylabel('Number of pairs', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
ax.legend(fontsize=10)
ax.set_title('Pair counts per distance bin', fontsize=11, fontweight='bold')

# Bottom: probability side-by-side + total line
ax = axes[1]
w = 0.35
ax.bar(x - w/2, p_excit, width=w, color='coral',     edgecolor='none',
       alpha=0.85, label='P(excitatory)')
ax.bar(x + w/2, p_inhib, width=w, color='steelblue', edgecolor='none',
       alpha=0.85, label='P(inhibitory)')
ax.plot(x, p_total, color='black', marker='o', markersize=4,
        linewidth=1.2, label='P(significant total)', zorder=3)
ax.set_ylabel('P(significant)', fontsize=11)
ax.set_xlabel('Distance (µm)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylim(0, None)
ax.legend(fontsize=10)
ax.set_title('P(significant) = n_significant / n_total per bin',
             fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('distance_histogram.pdf', bbox_inches='tight')
plt.savefig('distance_histogram.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved: distance_histogram.pdf")
print("Saved: distance_histogram.png")

# ─────────────────────────────────────────────
# PDF 2 — Decay shape analysis (log transforms)
# Uses total P(significant) vs bin centre distance
# Only bins with p_total > 0 and all_counts > 0
# ─────────────────────────────────────────────
d  = bin_centres                  # x = distance
p  = p_total.copy()               # y = probability
valid = np.isfinite(p) & (p > 0) & (all_counts.values > 0)

d_v = d[valid]
p_v = p[valid]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    'Decay shape analysis — P(significant) vs distance\n'
    'Straight line on each plot indicates that functional form fits the data',
    fontsize=12, fontweight='bold'
)

transforms = [
    # (title, x_transform, y_transform, xlabel, ylabel)
    ('Log-linear  →  Exponential decay\nP ∝ e^(−λd)',
     d_v, np.log(p_v),
     'Distance (µm)', 'log P'),
    ('Log-log  →  Power law\nP ∝ d^(−n)',
     np.log(d_v), np.log(p_v),
     'log Distance', 'log P'),
    ('Linear-log  →  Logarithmic decay\nP ∝ −log(d)',
     np.log(d_v), p_v,
     'log Distance', 'P'),
]

for ax, (title, xd, yd, xl, yl) in zip(axes, transforms):
    coeffs, _, y_fit_line = fit_line(xd, yd)

    ax.scatter(xd, yd, color='black', s=30, zorder=3, label='Data')

    if coeffs is not None:
        x_line = np.linspace(xd[np.isfinite(xd)].min(),
                             xd[np.isfinite(xd)].max(), 200)
        y_line = np.polyval(coeffs, x_line)
        # R²
        mask = np.isfinite(xd) & np.isfinite(yd)
        y_pred = np.polyval(coeffs, xd[mask])
        ss_res = np.sum((yd[mask] - y_pred) ** 2)
        ss_tot = np.sum((yd[mask] - yd[mask].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        ax.plot(x_line, y_line, color='red', linewidth=1.5,
                label=f'Linear fit  R²={r2:.3f}')

    ax.set_xlabel(xl, fontsize=11)
    ax.set_ylabel(yl, fontsize=11)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('distance_decay_shape.pdf', bbox_inches='tight')
plt.savefig('distance_decay_shape.png', dpi=600, bbox_inches='tight')
plt.close()
print("Saved: distance_decay_shape.pdf")
print("Saved: distance_decay_shape.png")
print("Done.")
