import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle

THRESHOLD = 10

# ─────────────────────────────────────────────
# Load significant pairs from each group
# ─────────────────────────────────────────────
def load_sig(path, threshold):
    df = pd.read_csv(path)
    sig = df[df['S_ij'].abs() > threshold]
    return set(zip(sig['trigger_unit'].astype(int), sig['response_unit'].astype(int)))

print("Loading...")
sig_all     = load_sig('Significance_Analysis/sij_v2_scores.csv',               THRESHOLD)
sig_within  = load_sig('Within_Burst/Significance_Analysis/sij_v2_scores.csv',  THRESHOLD)
sig_between = load_sig('Between_Burst/Significance_Analysis/sij_v2_scores.csv', THRESHOLD)

print(f"  Full dataset:   {len(sig_all)} significant pairs")
print(f"  Within Burst:   {len(sig_within)} significant pairs")
print(f"  Between Burst:  {len(sig_between)} significant pairs")

# Set intersections
only_all     = sig_all - sig_within - sig_between
only_within  = sig_within - sig_all - sig_between
only_between = sig_between - sig_all - sig_within
all_and_w    = (sig_all & sig_within) - sig_between
all_and_b    = (sig_all & sig_between) - sig_within
w_and_b      = (sig_within & sig_between) - sig_all
all_three    = sig_all & sig_within & sig_between
union        = sig_all | sig_within | sig_between

print(f"\n  Only Full:              {len(only_all)}")
print(f"  Only Within:            {len(only_within)}")
print(f"  Only Between:           {len(only_between)}")
print(f"  Full & Within only:     {len(all_and_w)}")
print(f"  Full & Between only:    {len(all_and_b)}")
print(f"  Within & Between only:  {len(w_and_b)}")
print(f"  All three:              {len(all_three)}")
print(f"  Union:                  {len(union)}")

# ─────────────────────────────────────────────
# Figure 1 — Venn diagram
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect('equal')
ax.axis('off')
ax.set_xlim(-3, 3)
ax.set_ylim(-2.5, 3)

fig.suptitle(
    f'Significant pairs comparison  (|S_ij| > {THRESHOLD})\n'
    f'Full={len(sig_all)},  Within={len(sig_within)},  Between={len(sig_between)}',
    fontsize=13, fontweight='bold'
)

# Three overlapping circles
r = 1.5
centers = [(-0.8, 0.6), (0.8, 0.6), (0.0, -0.8)]
colors  = ['#4e79a7', '#59a14f', '#f28e2b']
labels  = ['Full dataset', 'Within Burst', 'Between Burst']

for (cx, cy), color, label in zip(centers, colors, labels):
    circle = Circle((cx, cy), r, color=color, alpha=0.25, linewidth=2,
                    edgecolor=color)
    ax.add_patch(circle)

# Label circles
ax.text(-1.8,  1.8, 'Full\ndataset',  ha='center', va='center',
        fontsize=12, fontweight='bold', color='#4e79a7')
ax.text( 1.8,  1.8, 'Within\nBurst',  ha='center', va='center',
        fontsize=12, fontweight='bold', color='#59a14f')
ax.text( 0.0, -2.0, 'Between\nBurst', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#f28e2b')

# Counts in each region
regions = [
    (-1.4,  0.2, len(only_all)),
    ( 1.4,  0.2, len(only_within)),
    ( 0.0, -1.2, len(only_between)),
    ( 0.0,  1.1, len(all_and_w)),
    (-0.7, -0.4, len(all_and_b)),
    ( 0.7, -0.4, len(w_and_b)),
    ( 0.0,  0.1, len(all_three)),
]
for (rx, ry, count) in regions:
    ax.text(rx, ry, str(count), ha='center', va='center',
            fontsize=14, fontweight='bold', color='black')

plt.tight_layout()
plt.savefig('Significance_Analysis/compare_venn_v2.pdf', bbox_inches='tight')
plt.close()
print("Saved: Significance_Analysis/compare_venn_v2.pdf")

# ─────────────────────────────────────────────
# Figure 2 — Membership dot grid
# Shows every pair in the union; columns = groups
# ─────────────────────────────────────────────
union_list = sorted(union)
n = len(union_list)

membership = np.array([
    [1 if p in sig_all     else 0,
     1 if p in sig_within  else 0,
     1 if p in sig_between else 0]
    for p in union_list
], dtype=float)

# Sort rows by membership pattern for readability
sort_key = membership @ np.array([4, 2, 1])
order = np.argsort(-sort_key)
membership = membership[order]
union_sorted = [union_list[i] for i in order]

fig, ax = plt.subplots(figsize=(6, max(6, n * 0.12)))
ax.set_aspect('auto')

col_colors = ['#4e79a7', '#59a14f', '#f28e2b']
col_labels = ['Full', 'Within', 'Between']

for col, (color, lbl) in enumerate(zip(col_colors, col_labels)):
    for row in range(n):
        val = membership[row, col]
        ax.scatter(col, row, s=30, color=color if val else 'lightgray',
                   alpha=0.9 if val else 0.3, linewidths=0)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(col_labels, fontsize=11, fontweight='bold')
ax.set_yticks([])
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-1, n)
ax.set_title(
    f'Pair membership across groups\n({n} pairs in union)',
    fontsize=11, fontweight='bold'
)

# Add count annotations per column
for col, lbl in enumerate(col_labels):
    cnt = int(membership[:, col].sum())
    ax.text(col, -0.7, f'n={cnt}', ha='center', va='top',
            fontsize=9, color=col_colors[col], fontweight='bold')

plt.tight_layout()
plt.savefig('Significance_Analysis/compare_membership_v2.pdf', bbox_inches='tight')
plt.close()
print("Saved: Significance_Analysis/compare_membership_v2.pdf")
print("Done.")
