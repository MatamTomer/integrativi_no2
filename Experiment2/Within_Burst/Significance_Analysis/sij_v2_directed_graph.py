import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import networkx as nx

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
# Clustering for node colours
# ─────────────────────────────────────────────
N_CLUSTERS     = 4
EDGE_THRESHOLD = 10

features       = np.nan_to_num(S, nan=0.0)
distances      = pdist(features, metric='euclidean')
linkage_tree   = linkage(distances, method='ward')
cluster_labels = fcluster(linkage_tree, N_CLUSTERS, criterion='maxclust')

print(f"  Cluster sizes: { {c: int(np.sum(cluster_labels == c)) for c in range(1, N_CLUSTERS+1)} }")

# ─────────────────────────────────────────────
# Build directed graph
# Excitatory edges: coral/red
# Inhibitory edges: teal/blue
# ─────────────────────────────────────────────
G = nx.DiGraph()
for i in range(N):
    G.add_node(i, label=str(neuron_ids[i]), cluster=int(cluster_labels[i]))

excit_edges = []
inhib_edges = []

for i in range(N):
    for j in range(N):
        if i == j or np.isnan(S[i][j]):
            continue
        if abs(S[i][j]) > EDGE_THRESHOLD:
            G.add_edge(i, j, weight=float(abs(S[i][j])))
            if S[i][j] > 0:
                excit_edges.append((i, j, float(S[i][j])))
            else:
                inhib_edges.append((i, j, float(S[i][j])))

n_excit = len(excit_edges)
n_inhib = len(inhib_edges)
print(f"  Excitatory edges (S > {EDGE_THRESHOLD}, E_S dominant): {n_excit}")
print(f"  Inhibitory edges (S > {EDGE_THRESHOLD}, I_S dominant): {n_inhib}")

# ─────────────────────────────────────────────
# Layout and drawing
# ─────────────────────────────────────────────
pos = nx.spring_layout(G, weight='weight', k=2.5, seed=42, iterations=100)

CLUSTER_COLORS = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759']
node_colors = [CLUSTER_COLORS[cluster_labels[i] - 1] for i in range(N)]
node_labels = {i: str(neuron_ids[i]) for i in range(N)}

all_s_vals = [abs(s) for _, _, s in excit_edges + inhib_edges]
norm = Normalize(vmin=EDGE_THRESHOLD,
                 vmax=np.percentile(all_s_vals, 95) if all_s_vals else EDGE_THRESHOLD + 1)

fig, ax = plt.subplots(figsize=(16, 16))
ax.set_aspect('equal')
ax.axis('off')
ax.set_title(
    f'V2 directed connectivity graph  (edges: S_ij > {EDGE_THRESHOLD})\n'
    f'Coral = excitatory (E_S dominant),  Teal = inhibitory (I_S dominant)\n'
    f'Node color = cluster,  edge width = S_ij strength\n'
    f'Excitatory: {n_excit},  Inhibitory: {n_inhib}',
    fontsize=12, fontweight='bold'
)

# Excitatory edges
if excit_edges:
    excit_colors = [plt.cm.YlOrRd(norm(abs(s))) for _, _, s in excit_edges]
    excit_widths = [0.5 + 2.5 * norm(abs(s)) for _, _, s in excit_edges]
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=[(i, j) for i, j, _ in excit_edges],
        edge_color=excit_colors, width=excit_widths,
        arrows=True, arrowstyle='-|>', arrowsize=15,
        connectionstyle='arc3,rad=0.12',
        min_source_margin=18, min_target_margin=18,
    )

# Inhibitory edges
if inhib_edges:
    inhib_colors = [plt.cm.GnBu(norm(abs(s))) for _, _, s in inhib_edges]
    inhib_widths = [0.5 + 2.5 * norm(abs(s)) for _, _, s in inhib_edges]
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=[(i, j) for i, j, _ in inhib_edges],
        edge_color=inhib_colors, width=inhib_widths,
        arrows=True, arrowstyle='-|>', arrowsize=15,
        connectionstyle='arc3,rad=-0.12',
        min_source_margin=18, min_target_margin=18,
    )

nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                       node_size=600, edgecolors='black', linewidths=1.2)
nx.draw_networkx_labels(G, pos, ax=ax, labels=node_labels,
                        font_size=8, font_color='white', font_weight='bold')

# Legend
legend_handles = [mpatches.Patch(color=CLUSTER_COLORS[c - 1], label=f'Cluster {c}')
                  for c in range(1, N_CLUSTERS + 1)]
legend_handles += [
    mpatches.Patch(color='coral', label='Excitatory edge'),
    mpatches.Patch(color='teal',  label='Inhibitory edge'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=10,
          title='Legend', title_fontsize=10)

plt.tight_layout()
plt.savefig('sij_v2_directed_graph.pdf', bbox_inches='tight')
plt.close()
print("Saved: sij_v2_directed_graph.pdf")
print("Done.")
