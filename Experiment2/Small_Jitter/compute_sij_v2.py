import json
import argparse
import numpy as np
from pathlib import Path

SMALL_JITTER = Path(__file__).parent

CONFIG_PATHS = {
    'all':          SMALL_JITTER / 'All',
    'within_burst': SMALL_JITTER / 'Within_Burst',
    'between_burst':SMALL_JITTER / 'Between_Burst',
}

parser = argparse.ArgumentParser()
parser.add_argument('--config', choices=['all', 'within_burst', 'between_burst'], default='all')
args = parser.parse_args()

folder = CONFIG_PATHS[args.config]
print(f"Config: {args.config}  ({folder})")

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
with open(folder / 'histogram_data.json', 'r') as f:
    hist_data = json.load(f)
with open(folder / 'null_histograms.json', 'r') as f:
    null_mean_data = json.load(f)
with open(folder / 'null_std.json', 'r') as f:
    null_std_data = json.load(f)

observed_full = np.array(hist_data['histograms'])
null_mean     = np.array(null_mean_data['data'])
null_std      = np.array(null_std_data['data'])
neuron_ids    = hist_data['neuron_ids']
obs_bins      = hist_data['bins']
null_bins     = null_mean_data['bins']
N             = hist_data['n_neurons']

# Align observed to null bins in case of mismatch (e.g. 21-bin obs vs 20-bin null)
if obs_bins != null_bins:
    print(f"  Bin mismatch: obs={len(obs_bins)} bins, null={len(null_bins)} bins — aligning to null bins")
    keep = [obs_bins.index(b) for b in null_bins if b in obs_bins]
    observed = observed_full[:, :, keep]
else:
    observed = observed_full
bins = null_bins
print(f"  {N} neurons, {len(bins)} bins")

# ─────────────────────────────────────────────
# Z-scores
# ─────────────────────────────────────────────
with np.errstate(invalid='ignore', divide='ignore'):
    z_scores = np.where(null_std > 0,
                        (observed - null_mean) / null_std,
                        0.0)

# Bin layout depends on whether zero bin is included
# Find indices for +1ms, +2ms, +3ms, +4ms, +5ms
bins_arr = np.array(bins)
idx = {b: int(np.where(bins_arr == b)[0][0]) for b in [1, 2, 3, 4, 5] if b in bins_arr}

i1  = idx.get(1)
i2  = idx.get(2)
i3  = idx.get(3)
i5  = idx.get(5)

if i1 is None or i2 is None or i3 is None or i5 is None:
    raise ValueError(f"Expected bins +1..+5 not found. bins={bins}")

z_12     = z_scores[:, :, i1:i2+1]     # +1ms, +2ms
z_35     = z_scores[:, :, i3:i5+1]     # +3ms..+5ms
baseline = np.mean(z_35, axis=2)
E_S      = np.max(z_12, axis=2) - baseline    # excitatory score
I_S      = np.min(z_12, axis=2) - baseline    # inhibitory score

# Mask self-pairs
np.fill_diagonal(E_S, np.nan)
np.fill_diagonal(I_S, np.nan)

print(f"  E_S range (non-nan): [{np.nanmin(E_S):.3f}, {np.nanmax(E_S):.3f}]")
print(f"  I_S range (non-nan): [{np.nanmin(I_S):.3f}, {np.nanmax(I_S):.3f}]")

# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────
output = {
    'neuron_ids':  neuron_ids,
    'bins':        bins,
    'config':      args.config,
    'n_neurons':   N,
    'jitter_min_ms': null_mean_data['jitter_min_ms'],
    'jitter_max_ms': null_mean_data['jitter_max_ms'],
    'description': (
        'E_S[i][j] = max z-score over +1..+2ms minus mean z-score over +3..+5ms. '
        'I_S[i][j] = min z-score over +1..+2ms minus mean z-score over +3..+5ms.'
    ),
    'E_S': E_S.tolist(),
    'I_S': I_S.tolist(),
    'z_scores': z_scores.tolist(),
}

out_path = folder / 'sij_v2_scores.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"Saved: {out_path}")
