import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────
THRESHOLD      = 10
N_CLUSTERS     = 4
CLUSTER_COLORS = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759']

# ─────────────────────────────────────────────
# Load positions
# ─────────────────────────────────────────────
print("Loading unit positions...")
pos_df = pd.read_csv('../unit_positions.csv')
pos_df = pos_df.set_index('unit')

# ─────────────────────────────────────────────
# Load data and compute S_ij (v2)
# ─────────────────────────────────────────────
print("Loading spike data...")
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

with np.errstate(invalid='ignore', divide='ignore'):
    z_scores = np.where(null_std > 0, (observed - null_mean) / null_std, 0.0)

z_12     = z_scores[:, :, 10:12]
z_35     = z_scores[:, :, 12:15]
baseline = np.mean(z_35, axis=2)
E_S = np.max(z_12, axis=2) - baseline
I_S = np.min(z_12, axis=2) - baseline
S   = np.where(np.abs(I_S) > np.abs(E_S), I_S, E_S)

for i in range(N):
    for j in range(N):
        if i == j or observed[i][j].sum() < 300:
            S[i][j] = np.nan

# ─────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────
print("Clustering...")
features       = np.nan_to_num(S, nan=0.0)
distances      = pdist(features, metric='euclidean')
linkage_tree   = linkage(distances, method='ward')
cluster_labels = fcluster(linkage_tree, N_CLUSTERS, criterion='maxclust')

# ─────────────────────────────────────────────
# Build XY arrays aligned to neuron_ids order
# ─────────────────────────────────────────────
xs = np.array([pos_df.loc[uid, 'x'] if uid in pos_df.index else np.nan
               for uid in neuron_ids])
ys = np.array([pos_df.loc[uid, 'y'] if uid in pos_df.index else np.nan
               for uid in neuron_ids])

n_missing = int(np.sum(np.isnan(xs)))
if n_missing:
    print(f"  Warning: {n_missing} units have no position data")

# ─────────────────────────────────────────────
# Collect significant edges
# ─────────────────────────────────────────────
excit_edges = []  # (i, j, S_val)
inhib_edges = []

for i in range(N):
    for j in range(N):
        if i == j or np.isnan(S[i][j]):
            continue
        if abs(S[i][j]) > THRESHOLD:
            if S[i][j] > 0:
                excit_edges.append((i, j, S[i][j]))
            else:
                inhib_edges.append((i, j, S[i][j]))

print(f"  Excitatory edges: {len(excit_edges)}")
print(f"  Inhibitory edges: {len(inhib_edges)}")

all_s_abs = [abs(s) for _, _, s in excit_edges + inhib_edges]
s_max = np.percentile(all_s_abs, 95) if all_s_abs else THRESHOLD + 1

def edge_alpha(s_val):
    return 0.08 + 0.5 * min(abs(s_val) / s_max, 1.0)

def edge_lw(s_val):
    return 0.3 + 1.2 * min(abs(s_val) / s_max, 1.0)

# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
print("Plotting...")
fig, ax = plt.subplots(figsize=(14, 12))

# Draw edges first (behind nodes)
for i, j, s_val in excit_edges:
    ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
            color='coral', alpha=edge_alpha(s_val),
            linewidth=edge_lw(s_val), zorder=1)

for i, j, s_val in inhib_edges:
    ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
            color='steelblue', alpha=edge_alpha(s_val),
            linewidth=edge_lw(s_val), zorder=1)

# Draw nodes
node_colors = [CLUSTER_COLORS[cluster_labels[i] - 1] for i in range(N)]
ax.scatter(xs, ys, c=node_colors, s=120, zorder=3,
           edgecolors='black', linewidths=0.6)

# Label each unit
for i, uid in enumerate(neuron_ids):
    if not np.isnan(xs[i]):
        ax.text(xs[i], ys[i], str(uid),
                fontsize=4.5, ha='center', va='center',
                color='white', fontweight='bold', zorder=4)

ax.set_xlabel('X position (µm)', fontsize=12)
ax.set_ylabel('Y position (µm)', fontsize=12)
ax.set_title(
    f'Unit positions — Experiment 2  (n={N} units)\n'
    f'Node color = cluster (k={N_CLUSTERS})   '
    f'Coral edges = excitatory (S > {THRESHOLD}), n={len(excit_edges)}   '
    f'Blue edges = inhibitory (S < −{THRESHOLD}), n={len(inhib_edges)}',
    fontsize=11, fontweight='bold'
)

legend_handles = [mpatches.Patch(color=CLUSTER_COLORS[c - 1], label=f'Cluster {c}')
                  for c in range(1, N_CLUSTERS + 1)]
legend_handles += [
    mpatches.Patch(color='coral',      label=f'Excitatory edge (S > {THRESHOLD})'),
    mpatches.Patch(color='steelblue',  label=f'Inhibitory edge (S < −{THRESHOLD})'),
]
ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
          framealpha=0.9)

ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('unit_position_map.pdf', bbox_inches='tight')
plt.close()
print("Saved: unit_position_map.pdf")
print("Done.")
