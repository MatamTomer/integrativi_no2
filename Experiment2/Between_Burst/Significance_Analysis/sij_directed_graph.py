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
# 1. Load data and recompute S_ij
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

early = z_scores[:, :, 10:15]
late  = z_scores[:, :, 15:20]
S     = np.max(early, axis=2) - np.median(late, axis=2)

for i in range(N):
    for j in range(N):
        if i == j or observed[i][j].sum() < 300:
            S[i][j] = np.nan

# ─────────────────────────────────────────────
# 2. Hierarchical clustering — assigns each neuron to a cluster
# ─────────────────────────────────────────────
N_CLUSTERS     = 4
EDGE_THRESHOLD = 50

features     = np.nan_to_num(S, nan=0.0)
distances    = pdist(features, metric='euclidean')
linkage_tree = linkage(distances, method='ward')
cluster_labels = fcluster(linkage_tree, N_CLUSTERS, criterion='maxclust')

print(f"  Cluster sizes: { {c: int(np.sum(cluster_labels == c)) for c in range(1, N_CLUSTERS+1)} }")

# ─────────────────────────────────────────────
# 3. Build the directed graph
#
#    Each neuron is a node.
#    Each pair (i, j) with S_ij > threshold becomes a directed edge.
#    The edge weight = S_ij value (used for visual thickness/color).
# ─────────────────────────────────────────────
G = nx.DiGraph()

for i in range(N):
    G.add_node(i, label=str(neuron_ids[i]), cluster=int(cluster_labels[i]))

edges      = []
for i in range(N):
    for j in range(N):
        if i == j or np.isnan(S[i][j]):
            continue
        if S[i][j] > EDGE_THRESHOLD:
            G.add_edge(i, j, weight=float(S[i][j]))
            edges.append((i, j, float(S[i][j])))

print(f"  Edges with S_ij > {EDGE_THRESHOLD}: {len(edges)}")

# ─────────────────────────────────────────────
# 4. Compute spring layout
#
#    spring_layout treats edge weights as attraction forces:
#    strongly connected nodes are pulled closer together.
#    k controls the spacing — larger k = more spread out.
#    seed ensures the layout is reproducible.
# ─────────────────────────────────────────────
pos = nx.spring_layout(G, weight='weight', k=2.5, seed=42, iterations=100)

# ─────────────────────────────────────────────
# 5. Draw the graph
# ─────────────────────────────────────────────
CLUSTER_COLORS = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759']

node_colors = [CLUSTER_COLORS[cluster_labels[i] - 1] for i in range(N)]
node_labels = {i: str(neuron_ids[i]) for i in range(N)}

edge_weights = np.array([S[i][j] for i, j, _ in edges])
norm         = Normalize(vmin=EDGE_THRESHOLD, vmax=np.percentile(edge_weights, 95))
cmap         = plt.cm.OrRd
edge_colors  = [cmap(norm(S[i][j])) for i, j, _ in edges]
edge_widths  = [0.5 + 2.0 * norm(S[i][j]) for i, j, _ in edges]

fig, ax = plt.subplots(figsize=(16, 16))
ax.set_aspect('equal')
ax.axis('off')
ax.set_title(
    f'Directed connectivity graph  (edges: S_ij > {EDGE_THRESHOLD})\n'
    f'Node color = hierarchical cluster (k={N_CLUSTERS}), '
    f'edge color/width = S_ij strength',
    fontsize=13, fontweight='bold'
)

# Draw edges
nx.draw_networkx_edges(
    G, pos, ax=ax,
    edgelist=[(i, j) for i, j, _ in edges],
    edge_color=edge_colors,
    width=edge_widths,
    arrows=True,
    arrowstyle='-|>',
    arrowsize=15,
    connectionstyle='arc3,rad=0.12',   # slight curve so i->j and j->i don't overlap
    min_source_margin=18,
    min_target_margin=18,
)

# Draw nodes
nx.draw_networkx_nodes(
    G, pos, ax=ax,
    node_color=node_colors,
    node_size=600,
    edgecolors='black',
    linewidths=1.2,
)

# Draw labels
nx.draw_networkx_labels(
    G, pos, ax=ax,
    labels=node_labels,
    font_size=8,
    font_color='white',
    font_weight='bold',
)

# Colorbar for edge strength
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, shrink=0.4)
cbar.set_label('S_ij value', fontsize=10)

# Cluster legend
legend_handles = [
    mpatches.Patch(color=CLUSTER_COLORS[c - 1], label=f'Cluster {c}')
    for c in range(1, N_CLUSTERS + 1)
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=10,
          title='Clusters', title_fontsize=10)

plt.tight_layout()
plt.savefig('sij_directed_graph.pdf', bbox_inches='tight')
plt.close()
print("Saved: sij_directed_graph.pdf")
print("Done.")
