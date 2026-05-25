import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

EXPERIMENT2 = Path(__file__).parent.parent          # Within_Burst/
CONFIGS     = Path(__file__).parent.parent.parent / 'Configurations'  # Experiment2/Configurations/

# ─── EDIT THIS ───────────────────────────────────────────────────────────────
# Set to an integer (e.g. 5) to build weights from a specific per-config p_ij.
# Per-config files live in: Spike_Triggered_Average/Per_Config/p_ij_config{N}.json
# Set to None to use the --config argument instead (all).
PER_CONFIG_ID = None
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_PATHS = {
    'all': EXPERIMENT2 / 'Spike_Triggered_Average' / 'p_ij.json',
}

parser = argparse.ArgumentParser()
parser.add_argument('--config', choices=['all'], default='all')
parser.add_argument('--per-config-id', type=int, default=None,
                    help='Override PER_CONFIG_ID: load p_ij from Per_Config/p_ij_config{N}.json')
args = parser.parse_args()

# CLI flag overrides the top-of-file constant
if args.per_config_id is not None:
    PER_CONFIG_ID = args.per_config_id

if PER_CONFIG_ID is not None:
    per_config_path = (EXPERIMENT2 / 'Spike_Triggered_Average' / 'Per_Config'
                       / f'p_ij_config{PER_CONFIG_ID}.json')
    if not per_config_path.exists():
        raise FileNotFoundError(f"No per-config p_ij found at: {per_config_path}")
    config_label = f'per_config_{PER_CONFIG_ID}'
    print(f"Mode: per-config  (config id = {PER_CONFIG_ID})")
else:
    config_label = args.config
    print(f"Config: {args.config}")

# ─────────────────────────────────────────────
# Load p_ij
# ─────────────────────────────────────────────
pij_path = per_config_path if PER_CONFIG_ID is not None else CONFIG_PATHS[args.config]
print(f"Loading p_ij from {pij_path}...")
with open(pij_path) as f:
    data = json.load(f)

p_ij       = np.array(data['p_ij'])   # (N, N, NUM_BINS)
neuron_ids = data['neuron_ids']
bins       = data['bins']
N          = data['n_neurons']
print(f"  {N} neurons, {len(bins)} bins, shape: {p_ij.shape}")

# ─────────────────────────────────────────────
# Load B_j — baseline firing probability
#   per-config mode : B_i for that specific config
#   other modes     : mean B_i across all recording configs
# ─────────────────────────────────────────────
print(f"Loading B_j from {CONFIGS / 'config_B_i.csv'}...")
bi_df = pd.read_csv(CONFIGS / 'config_B_i.csv')

if PER_CONFIG_ID is not None:
    cfg_rows = bi_df[bi_df['config'] == PER_CONFIG_ID]
    if cfg_rows.empty:
        raise ValueError(f"No B_i rows found for config {PER_CONFIG_ID} in config_B_i.csv")
    B_source = cfg_rows.set_index('unit')['B_i']
    print(f"  Using config-specific B_i for config {PER_CONFIG_ID} ({len(cfg_rows)} rows)")
else:
    B_source = bi_df.groupby('unit')['B_i'].mean()   # mean across all recording configs
    print(f"  Using mean B_i across all recording configs")

# Align to neuron_ids order
B_j = np.array([B_source.get(nid, 0.0) for nid in neuron_ids], dtype=np.float64)
print(f"  B_j range: [{B_j.min():.6f}, {B_j.max():.6f}]  mean={B_j.mean():.6f}")

# ─────────────────────────────────────────────
# Weights: w_ij(tau) = p_ij(tau) - B_j
# B_j broadcast: shape (N,) -> subtract from axis=1 (j dimension)
# ─────────────────────────────────────────────
# p_ij shape: (N, N, NUM_BINS)
# B_j shape:  (N,)  — one baseline per target neuron j
w_ij = p_ij - B_j[np.newaxis, :, np.newaxis]   # broadcast over i and bins

# Zero out self-pairs
for k in range(w_ij.shape[2]):
    np.fill_diagonal(w_ij[:, :, k], 0.0)

# ─────────────────────────────────────────────
# Summary stats
# ─────────────────────────────────────────────
w_peak   = w_ij.max(axis=2)
w_trough = w_ij.min(axis=2)
np.fill_diagonal(w_peak,   np.nan)
np.fill_diagonal(w_trough, np.nan)

excit_pairs = int((w_peak   > 0).sum())
inhib_pairs = int((w_trough < 0).sum())
total_pairs = N * (N - 1)

print(f"  Pairs with any excitation (w>0): {excit_pairs} / {total_pairs}")
print(f"  Pairs with any inhibition (w<0): {inhib_pairs} / {total_pairs}")
print(f"  Peak excitatory w range : [{np.nanmin(w_peak[w_peak > 0]):.6f}, {np.nanmax(w_peak):.6f}]")
print(f"  Peak inhibitory w range : [{np.nanmin(w_trough):.6f}, {np.nanmax(w_trough[w_trough < 0]):.6f}]")

# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────
b_j_description = (
    f'config-specific B_i for config {PER_CONFIG_ID}' if PER_CONFIG_ID is not None
    else 'mean B_i across all recording configs'
)
output = {
    'neuron_ids':  neuron_ids,
    'bins':        bins,
    'config':      config_label,
    'n_neurons':   N,
    'B_j':         B_j.tolist(),
    'description': (
        'w_ij[i][j][k] = p_ij[i][j][k] - B_j[j]  '
        '= excess conditional firing probability of neuron j at offset bins[k] ms '
        f'given a spike of neuron i, above neuron j baseline. '
        f'B_j = {b_j_description}. Shape: (N, N, NUM_BINS).'
    ),
    'w_ij': w_ij.tolist(),
}

out_path = Path(__file__).parent / f'weights_pij_{config_label}.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"Saved: {out_path}")
