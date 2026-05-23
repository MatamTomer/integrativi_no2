import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────
N_CLUSTERS     = 4
CLUSTER_COLORS = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759']

# ─────────────────────────────────────────────
# Load data and compute S_ij (v2)
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
labels     = [str(nid) for nid in neuron_ids]

with np.errstate(invalid='ignore', divide='ignore'):
    z_scores = np.where(null_std > 0, (observed - null_mean) / null_std, 0.0)

z_12   = z_scores[:, :, 11:13]
z_35   = z_scores[:, :, 13:16]
baseline = np.mean(z_35, axis=2)
E_S = np.max(z_12, axis=2) - baseline
I_S = np.min(z_12, axis=2) - baseline
S   = np.where(np.abs(I_S) > np.abs(E_S), I_S, E_S)

for i in range(N):
    for j in range(N):
        if i == j or observed[i][j].sum() < 300:
            S[i][j] = np.nan

# ─────────────────────────────────────────────
# Hierarchical clustering
# ─────────────────────────────────────────────
print("Clustering...")

features       = np.nan_to_num(S, nan=0.0)
distances      = pdist(features, metric='euclidean')
linkage_tree   = linkage(distances, method='ward')
cluster_labels = fcluster(linkage_tree, N_CLUSTERS, criterion='maxclust')

dend         = dendrogram(linkage_tree, no_plot=True)
leaf_order   = dend['leaves']
S_reordered  = S[np.ix_(leaf_order, leaf_order)]
labels_ord   = [labels[i] for i in leaf_order]
clusters_ord = cluster_labels[leaf_order]

print(f"  Cluster sizes: { {c: int(np.sum(cluster_labels == c)) for c in range(1, N_CLUSTERS+1)} }")

# ─────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────
print("Plotting...")

fig = plt.figure(figsize=(18, 16))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    width_ratios  = [0.12, 0.04, 1.0],
    height_ratios = [0.15, 0.04, 1.0],
    hspace=0.01,
    wspace=0.01,
)

ax_col_dend  = fig.add_subplot(gs[0, 2])
ax_row_dend  = fig.add_subplot(gs[2, 0])
ax_col_strip = fig.add_subplot(gs[1, 2])
ax_row_strip = fig.add_subplot(gs[2, 1])
ax_heat      = fig.add_subplot(gs[2, 2])

dendrogram(linkage_tree, ax=ax_row_dend, orientation='left',
           color_threshold=linkage_tree[-(N_CLUSTERS - 1), 2],
           above_threshold_color='gray', no_labels=True)
ax_row_dend.axis('off')

dendrogram(linkage_tree, ax=ax_col_dend, orientation='top',
           color_threshold=linkage_tree[-(N_CLUSTERS - 1), 2],
           above_threshold_color='gray', no_labels=True)
ax_col_dend.axis('off')
ax_col_dend.set_title(
    'V2 S_ij matrix — hierarchical clustering (Ward linkage, k=4)\n'
    r'$S_{ij}$ = max(|$E_S$|, |$I_S$|)   where   '
    r'$E_S$ = max Z[+1,+2ms] − mean Z[+3..+5ms]',
    fontsize=12, fontweight='bold', pad=12
)

for pos, c in enumerate(clusters_ord):
    color = CLUSTER_COLORS[c - 1]
    ax_col_strip.add_patch(mpatches.Rectangle((pos, 0), 1, 1, color=color, linewidth=0))
    ax_row_strip.add_patch(mpatches.Rectangle((0, N - pos - 1), 1, 1, color=color, linewidth=0))

for ax in [ax_col_strip, ax_row_strip]:
    ax.set_xlim(0, N); ax.set_ylim(0, N); ax.axis('off')

valid_vals = S_reordered[~np.isnan(S_reordered)]
vmin = np.percentile(valid_vals, 2)
vmax = np.percentile(valid_vals, 98)
vcenter = 0.0
if vmin >= 0:
    vcenter = vmin + (vmax - vmin) / 2
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

im = ax_heat.imshow(S_reordered, aspect='auto', cmap='YlOrRd',
                    vmin=vmin, vmax=vmax, interpolation='none')

ax_heat.set_xticks(np.arange(N))
ax_heat.set_xticklabels(labels_ord, rotation=90, fontsize=6)
ax_heat.set_yticks(np.arange(N))
ax_heat.set_yticklabels(labels_ord, fontsize=6)
ax_heat.set_xlabel('Response neuron j', fontsize=11, labelpad=8)
ax_heat.set_ylabel('Trigger neuron i',  fontsize=11, labelpad=8)
ax_heat.tick_params(axis='x', bottom=True, top=False)

current = clusters_ord[0]
for pos in range(1, N):
    if clusters_ord[pos] != current:
        ax_heat.axhline(y=pos - 0.5, color='black', linewidth=2.0)
        ax_heat.axvline(x=pos - 0.5, color='black', linewidth=2.0)
        current = clusters_ord[pos]

cbar = fig.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02, shrink=0.8)
cbar.set_label('S_ij (v2)', fontsize=11)
cbar.ax.tick_params(labelsize=9)

handles = [mpatches.Patch(color=CLUSTER_COLORS[c - 1], label=f'Cluster {c}')
           for c in range(1, N_CLUSTERS + 1)]
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.01, 0.88),
           fontsize=10, title='Clusters', title_fontsize=10, framealpha=0.9)

plt.savefig('sij_v2_clustering.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: sij_v2_clustering.pdf")
print("Done.")
