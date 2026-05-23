import json
import argparse
import numpy as np
from pathlib import Path

EXPERIMENT2 = Path(__file__).parent.parent

CONFIG_PATHS = {
    'all': {
        'histogram': EXPERIMENT2 / 'Spike_Triggered_Average' / 'histogram_data.json',
        'neuron_data': EXPERIMENT2 / 'neuron_data.json',
        'output': EXPERIMENT2 / 'Spike_Triggered_Average' / 'p_ij.json',
    },
    'within_burst': {
        'histogram': EXPERIMENT2 / 'Within_Burst' / 'Spike_Triggered_Average' / 'histogram_data.json',
        'neuron_data': EXPERIMENT2 / 'Within_Burst' / 'neuron_data.json',
        'output': EXPERIMENT2 / 'Within_Burst' / 'Spike_Triggered_Average' / 'p_ij.json',
    },
    'between_burst': {
        'histogram': EXPERIMENT2 / 'Between_Burst' / 'Spike_Triggered_Average' / 'histogram_data.json',
        'neuron_data': EXPERIMENT2 / 'Between_Burst' / 'neuron_data.json',
        'output': EXPERIMENT2 / 'Between_Burst' / 'Spike_Triggered_Average' / 'p_ij.json',
    },
}

parser = argparse.ArgumentParser()
parser.add_argument('--config', choices=['all', 'within_burst', 'between_burst'], default='all')
args = parser.parse_args()

paths = CONFIG_PATHS[args.config]
print(f"Config: {args.config}")

# ─────────────────────────────────────────────
# Load histogram data
# ─────────────────────────────────────────────
print(f"Loading histograms from {paths['histogram']}...")
with open(paths['histogram'], 'r') as f:
    hist_data = json.load(f)

histograms = np.array(hist_data['histograms'], dtype=np.float64)  # (N, N, NUM_BINS)
neuron_ids = hist_data['neuron_ids']
bins       = hist_data['bins']
N          = hist_data['n_neurons']
print(f"  {N} neurons, {len(bins)} bins")

# ─────────────────────────────────────────────
# Load neuron data to compute N_i (spike counts)
# ─────────────────────────────────────────────
print(f"Loading neuron data from {paths['neuron_data']}...")
with open(paths['neuron_data'], 'r') as f:
    neuron_data = json.load(f)

neuron_ids_from_data = [neuron[0]['unit'] for neuron in neuron_data if neuron]
spike_counts = {
    nid: len(neuron)
    for nid, neuron in zip(neuron_ids_from_data, neuron_data)
}

# N_i vector aligned to histogram neuron order
N_i = np.array([spike_counts[nid] for nid in neuron_ids], dtype=np.float64)
print(f"  Spike counts: min={int(N_i.min())}, max={int(N_i.max())}, mean={N_i.mean():.1f}")

# ─────────────────────────────────────────────
# Compute P_ij(tau) = C_ij(tau) / N_i
# ─────────────────────────────────────────────
# histograms shape: (N, N, NUM_BINS)
# N_i shape: (N,) — divide each row i by N_i[i]
p_ij = histograms / N_i[:, np.newaxis, np.newaxis]   # broadcast over j and bins

# Self-pairs are zero in histograms already, leave them as 0

print(f"  P_ij range: [{p_ij.min():.6f}, {p_ij.max():.6f}]")

# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────
output = {
    'neuron_ids': neuron_ids,
    'bins':       bins,
    'config':     args.config,
    'n_neurons':  N,
    'n_i':        N_i.tolist(),
    'description': (
        'p_ij[i][j][k] = C_ij(bins[k]) / N_i  '
        '= probability of neuron j firing at offset bins[k] ms '
        'given a spike of neuron i'
    ),
    'p_ij': p_ij.tolist(),
}

with open(paths['output'], 'w') as f:
    json.dump(output, f, indent=2)
print(f"Saved: {paths['output']}")
