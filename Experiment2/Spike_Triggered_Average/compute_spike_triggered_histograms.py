import json
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────
WINDOW_MS  = 10
BIN_EDGES  = list(range(-WINDOW_MS, WINDOW_MS + 1))   # [-10..10], 21 bins
NUM_BINS   = len(BIN_EDGES)
WINDOW_SEC = WINDOW_MS / 1000.0

EXPERIMENT2 = Path(__file__).parent.parent
HERE        = Path(__file__).parent

# ─────────────────────────────────────────────
# Load neuron data
# ─────────────────────────────────────────────
print("Loading neuron data...")
with open(EXPERIMENT2 / 'neuron_data.json', 'r') as f:
    neuron_data = json.load(f)

N          = len(neuron_data)
neuron_ids = [neuron[0]['unit'] for neuron in neuron_data if neuron]
spike_times = []
for neuron in neuron_data:
    times = np.array([spike['spikeTime'] for spike in neuron], dtype=float)
    times.sort()
    spike_times.append(times)

print(f"  {N} neurons, {sum(len(t) for t in spike_times):,} total spikes")

# ─────────────────────────────────────────────
# Load config time windows
# ─────────────────────────────────────────────
cfg_times = pd.read_csv(EXPERIMENT2 / 'Configurations' / 'config_times.csv')
print(f"  {len(cfg_times)} configs loaded from config_times.csv")


# ─────────────────────────────────────────────
# Core histogram function
# ─────────────────────────────────────────────
def compute_histograms(spike_times_list, label=''):
    """Compute (N, N, NUM_BINS) histogram array from a list of spike time arrays."""
    histograms = np.zeros((N, N, NUM_BINS), dtype=int)
    for i in range(N):
        trigger_spikes = spike_times_list[i]
        if len(trigger_spikes) == 0:
            continue
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  {label}Neuron {neuron_ids[i]} ({i+1}/{N})")
        for j in range(N):
            if i == j:
                continue
            other_spikes = spike_times_list[j]
            if len(other_spikes) == 0:
                continue
            for t in trigger_spikes:
                lo = np.searchsorted(other_spikes, t - WINDOW_SEC, side='left')
                hi = np.searchsorted(other_spikes, t + WINDOW_SEC, side='right')
                for t_j in other_spikes[lo:hi]:
                    offset_ms = int(round((t_j - t) * 1000))
                    if -WINDOW_MS <= offset_ms <= WINDOW_MS:
                        histograms[i][j][offset_ms + WINDOW_MS] += 1
    return histograms


def save_histogram(histograms, path, extra=None):
    output = {
        'neuron_ids':  neuron_ids,
        'bins':        BIN_EDGES,
        'window_ms':   WINDOW_MS,
        'n_neurons':   N,
        'description': 'histograms[i][j][k] = count of neuron j spikes at offset bins[k] ms relative to neuron i spikes',
        'histograms':  histograms.tolist(),
    }
    if extra:
        output.update(extra)
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# 1. All-spikes histogram (existing)
# ─────────────────────────────────────────────
print(f"\nComputing all-spikes histogram...")
hist_all = compute_histograms(spike_times, label='[all] ')
save_histogram(hist_all, HERE / 'histogram_data.json')

# ─────────────────────────────────────────────
# 2. Per-config histograms → Per_Config/
# ─────────────────────────────────────────────
per_config_dir = HERE / 'Per_Config'
per_config_dir.mkdir(exist_ok=True)

for _, row in cfg_times.iterrows():
    cfg_id    = int(row['config'])
    t_start   = float(row['t_start'])
    t_end     = float(row['t_end'])

    print(f"\nConfig {cfg_id}: {t_start:.3f}s – {t_end:.3f}s")

    # Filter each neuron's spikes to this config's time window
    cfg_spike_times = [
        st[(st >= t_start) & (st <= t_end)]
        for st in spike_times
    ]
    n_spikes = sum(len(s) for s in cfg_spike_times)
    print(f"  {n_spikes:,} spikes in window")

    hist_cfg = compute_histograms(cfg_spike_times, label=f'[cfg{cfg_id}] ')
    save_histogram(
        hist_cfg,
        per_config_dir / f'histogram_data_config{cfg_id}.json',
        extra={'config_id': cfg_id, 't_start': t_start, 't_end': t_end}
    )

print(f"\nAll done.")
print(f"  All-spikes : {HERE / 'histogram_data.json'}")
print(f"  Per-config : {per_config_dir}/  ({len(cfg_times)} files)")
print(f"  Run generate_pdfs.py to visualise.")
