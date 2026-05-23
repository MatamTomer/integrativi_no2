import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_venn import venn3, venn3_circles
from pathlib import Path

S_IJ_THRESHOLD  = 50
MIN_SPIKE_COUNT = 300

# ─────────────────────────────────────────────
# Helper: load S_ij and return set of passing (id_i, id_j) pairs
# ─────────────────────────────────────────────
def get_passing_pairs(sig_dir):
    """
    Returns a set of (neuron_id_i, neuron_id_j) tuples where:
      - the pair has >= MIN_SPIKE_COUNT correlated spikes
      - S_ij > S_IJ_THRESHOLD
    """
    sig_dir = Path(sig_dir)
    with open(sig_dir / '../Spike_Triggered_Average/histogram_data.json') as f:
        hist_data = json.load(f)
    with open(sig_dir / '../Null_Hypothesis/null_histograms.json') as f:
        null_mean_data = json.load(f)
    with open(sig_dir / '../Null_Hypothesis/null_std.json') as f:
        null_std_data = json.load(f)

    observed   = np.array(hist_data['histograms'])
    null_mean  = np.array(null_mean_data['data'])
    null_std   = np.array(null_std_data['data'])
    neuron_ids = hist_data['neuron_ids']
    N          = len(neuron_ids)

    with np.errstate(invalid='ignore', divide='ignore'):
        z = np.where(null_std > 0, (observed - null_mean) / null_std, 0.0)

    early = z[:, :, 10:15]
    late  = z[:, :, 15:20]
    S     = np.max(early, axis=2) - np.median(late, axis=2)

    passing = set()
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if observed[i][j].sum() < MIN_SPIKE_COUNT:
                continue
            if S[i][j] > S_IJ_THRESHOLD:
                passing.add((neuron_ids[i], neuron_ids[j]))

    return passing


# ─────────────────────────────────────────────
# 1. Load passing pairs for each group
# ─────────────────────────────────────────────
print("Computing passing pairs...")

all_pairs     = get_passing_pairs('Significance_Analysis')
within_pairs  = get_passing_pairs('Within_Burst/Significance_Analysis')
between_pairs = get_passing_pairs('Between_Burst/Significance_Analysis')

print(f"  All spikes:    {len(all_pairs):3d} passing pairs")
print(f"  Within burst:  {len(within_pairs):3d} passing pairs")
print(f"  Between burst: {len(between_pairs):3d} passing pairs")

# ─────────────────────────────────────────────
# 2. Set algebra
# ─────────────────────────────────────────────
A  = all_pairs
W  = within_pairs
B  = between_pairs

only_A   = A - W - B
only_W   = W - A - B
only_B   = B - A - W
AW_only  = (A & W) - B
AB_only  = (A & B) - W
WB_only  = (W & B) - A
AWB      = A & W & B
union    = A | W | B

print(f"\nSet breakdown:")
print(f"  Only All:               {len(only_A):3d}")
print(f"  Only Within:            {len(only_W):3d}")
print(f"  Only Between:           {len(only_B):3d}")
print(f"  All & Within only:      {len(AW_only):3d}")
print(f"  All & Between only:     {len(AB_only):3d}")
print(f"  Within & Between only:  {len(WB_only):3d}")
print(f"  All & Within & Between: {len(AWB):3d}")
print(f"  Union (any group):      {len(union):3d}")

# ─────────────────────────────────────────────
# 3. Figure layout:  Venn  (left)  +  membership grid (right)
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 9))
fig.patch.set_facecolor('white')

ax_venn = fig.add_axes([0.02, 0.08, 0.38, 0.84])   # left panel
ax_grid = fig.add_axes([0.44, 0.08, 0.54, 0.84])   # right panel

fig.suptitle(
    f'Comparison of S_ij > {S_IJ_THRESHOLD} passing pairs across recording conditions',
    fontsize=14, fontweight='bold', y=0.98
)

# ── Venn diagram ─────────────────────────────
# venn3 expects subsets in order:
#   (100, 010, 110, 001, 101, 011, 111)
#   = (A only, W only, A∩W, B only, A∩B, W∩B, A∩W∩B)
venn_counts = (
    len(only_A),
    len(only_W),
    len(AW_only),
    len(only_B),
    len(AB_only),
    len(WB_only),
    len(AWB),
)

COLORS = {
    'A': '#4e79a7',   # blue  — All spikes
    'W': '#59a14f',   # green — Within burst
    'B': '#f28e2b',   # orange — Between burst
}

v = venn3(
    subsets=venn_counts,
    set_labels=('All spikes', 'Within burst', 'Between burst'),
    set_colors=(COLORS['A'], COLORS['W'], COLORS['B']),
    alpha=0.55,
    ax=ax_venn,
)

# Make counts larger and bold
for lbl in v.subset_labels:
    if lbl is not None:
        lbl.set_fontsize(13)
        lbl.set_fontweight('bold')
for lbl in v.set_labels:
    if lbl is not None:
        lbl.set_fontsize(11)

ax_venn.set_title('Venn diagram of passing pairs', fontsize=12, pad=10)

# ── Membership grid ───────────────────────────
# Rows = all unique pairs (sorted), columns = 3 groups
all_union_sorted = sorted(union, key=lambda p: (p[0], p[1]))
n_pairs = len(all_union_sorted)

groups      = [A,   W,   B  ]
group_names = ['All', 'Within', 'Between']
group_cols  = [COLORS['A'], COLORS['W'], COLORS['B']]

# Build binary membership matrix  (n_pairs × 3)
membership = np.array([[1 if p in g else 0 for g in groups] for p in all_union_sorted])

# Draw grid
cell_h = 0.85 / n_pairs if n_pairs > 0 else 0.02
cell_w = 0.25

for row, pair in enumerate(all_union_sorted):
    y = 1.0 - (row + 0.5) / n_pairs   # top-to-bottom
    # Row label
    ax_grid.text(
        -0.02, y, f'{pair[0]}→{pair[1]}',
        ha='right', va='center', fontsize=7.5,
        transform=ax_grid.transAxes
    )
    for col, (g, color) in enumerate(zip(groups, group_cols)):
        x = 0.12 + col * 0.28
        if pair in g:
            ax_grid.plot(
                x, y, 'o',
                markersize=9,
                color=color,
                markeredgecolor='white',
                markeredgewidth=0.5,
                transform=ax_grid.transAxes,
                clip_on=False,
            )
        else:
            ax_grid.plot(
                x, y, 'o',
                markersize=9,
                color='#e0e0e0',
                markeredgecolor='#cccccc',
                markeredgewidth=0.5,
                transform=ax_grid.transAxes,
                clip_on=False,
            )

# Column headers
for col, (name, color) in enumerate(zip(group_names, group_cols)):
    x = 0.12 + col * 0.28
    ax_grid.text(
        x, 1.03, name,
        ha='center', va='bottom', fontsize=10, fontweight='bold',
        color=color, transform=ax_grid.transAxes
    )

ax_grid.set_xlim(0, 1)
ax_grid.set_ylim(0, 1)
ax_grid.axis('off')
ax_grid.set_title('Pair membership (filled = passed)', fontsize=12, pad=10)

# ─────────────────────────────────────────────
# 4. Save
# ─────────────────────────────────────────────
out_path = 'passing_pairs_comparison.pdf'
plt.savefig(out_path, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved: {out_path}")
print("Done.")
