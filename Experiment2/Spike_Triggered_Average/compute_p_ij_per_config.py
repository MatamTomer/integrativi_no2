import json
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
EXPERIMENT2    = Path(__file__).parent.parent
HERE           = Path(__file__).parent
PER_CONFIG_DIR = HERE / 'Per_Config'

# ─────────────────────────────────────────────
# Load neuron spike times (needed to compute N_i per config)
# ─────────────────────────────────────────────
print("Loading neuron data...")
with open(EXPERIMENT2 / 'neuron_data.json', 'r') as f:
    neuron_data = json.load(f)

neuron_ids  = [neuron[0]['unit'] for neuron in neuron_data if neuron]
spike_times = []
for neuron in neuron_data:
    times = np.array([spike['spikeTime'] for spike in neuron], dtype=float)
    times.sort()
    spike_times.append(times)

N = len(neuron_ids)
print(f"  {N} neurons, {sum(len(t) for t in spike_times):,} total spikes")

# ─────────────────────────────────────────────
# Find all per-config histogram files
# ─────────────────────────────────────────────
hist_files = sorted(
    PER_CONFIG_DIR.glob('histogram_data_config*.json'),
    key=lambda p: int(p.stem.replace('histogram_data_config', ''))
)
print(f"  Found {len(hist_files)} per-config histogram files in {PER_CONFIG_DIR}\n")

# ─────────────────────────────────────────────
# Process each config
# ─────────────────────────────────────────────
for hist_path in hist_files:
    with open(hist_path, 'r') as f:
        hist_data = json.load(f)

    cfg_id  = hist_data['config_id']
    t_start = hist_data['t_start']
    t_end   = hist_data['t_end']
    bins    = hist_data['bins']

    print(f"Config {cfg_id}: {t_start:.3f}s – {t_end:.3f}s")

    histograms = np.array(hist_data['histograms'], dtype=np.float64)  # (N, N, NUM_BINS)

    # N_i = spike count of neuron i within this config's time window
    N_i = np.array([
        np.sum((st >= t_start) & (st <= t_end))
        for st in spike_times
    ], dtype=np.float64)

    print(f"  N_i: min={int(N_i.min())}, max={int(N_i.max())}, mean={N_i.mean():.1f}")

    # P_ij(tau) = C_ij(tau) / N_i
    # Avoid division by zero for neurons with no spikes in this window
    with np.errstate(invalid='ignore', divide='ignore'):
        p_ij = np.where(
            N_i[:, np.newaxis, np.newaxis] > 0,
            histograms / N_i[:, np.newaxis, np.newaxis],
            0.0
        )

    print(f"  P_ij range: [{p_ij.min():.6f}, {p_ij.max():.6f}]")

    output = {
        'neuron_ids': neuron_ids,
        'bins':       bins,
        'config_id':  cfg_id,
        't_start':    t_start,
        't_end':      t_end,
        'n_neurons':  N,
        'n_i':        N_i.tolist(),
        'description': (
            'p_ij[i][j][k] = C_ij(bins[k]) / N_i  '
            '= probability of neuron j firing at offset bins[k] ms '
            'given a spike of neuron i, within config time window'
        ),
        'p_ij': p_ij.tolist(),
    }

    out_path = PER_CONFIG_DIR / f'p_ij_config{cfg_id}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_path}")

print(f"\nAll done. {len(hist_files)} p_ij files written to {PER_CONFIG_DIR}/")
