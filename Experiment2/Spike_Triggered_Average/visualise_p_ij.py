import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

EXPERIMENT2 = Path(__file__).parent.parent

CONFIG_PATHS = {
    'all':          EXPERIMENT2 / 'Spike_Triggered_Average' / 'p_ij.json',
    'within_burst': EXPERIMENT2 / 'Within_Burst' / 'Spike_Triggered_Average' / 'p_ij.json',
    'between_burst':EXPERIMENT2 / 'Between_Burst' / 'Spike_Triggered_Average' / 'p_ij.json',
}

parser = argparse.ArgumentParser()
parser.add_argument('--config', choices=['all', 'within_burst', 'between_burst'], default='all')
args = parser.parse_args()

print(f"Loading p_ij for config: {args.config}...")
with open(CONFIG_PATHS[args.config], 'r') as f:
    data = json.load(f)

p_ij      = np.array(data['p_ij'])       # (N, N, NUM_BINS)
neuron_ids = data['neuron_ids']
N          = data['n_neurons']

# Summary: max P_ij over all tau bins
p_max = p_ij.max(axis=2)                 # (N, N)

# Mask self-pairs
np.fill_diagonal(p_max, np.nan)

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(p_max, aspect='auto', interpolation='none',
               cmap='hot', vmin=0, vmax=np.nanpercentile(p_max, 99))

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('max P_ij(tau)  =  max C_ij(tau) / N_i', fontsize=10)

# Tick labels — show every 10th neuron to avoid crowding
step = 10
ticks = list(range(0, N, step))
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels([neuron_ids[t] for t in ticks], fontsize=7, rotation=90)
ax.set_yticklabels([neuron_ids[t] for t in ticks], fontsize=7)

ax.set_xlabel('Neuron j  (target)', fontsize=11)
ax.set_ylabel('Neuron i  (trigger)', fontsize=11)
ax.set_title(f'Peak conditional firing probability  —  config: {args.config}', fontsize=13)

plt.tight_layout()
out = Path(__file__).parent / f'p_ij_heatmap_{args.config}.pdf'
plt.savefig(out, bbox_inches='tight')
print(f"Saved: {out}")
