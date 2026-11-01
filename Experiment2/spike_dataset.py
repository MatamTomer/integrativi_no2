"""
spike_dataset.py
Converts spike trains from each recording config into a binned spike-count array.

Output shape: (num_configs, num_units, seq_len)

Adjustable at the top of the file.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Adjustable ────────────────────────────────────────────────────────────────
BIN_MS     = 1       # ms per bin
N_CONFIGS  = 20       # how many configs to include (None = all)
NEURONS    = None    # list of unit IDs to include, or None = all
CONFIG_IDS = None    # list of config IDs to include, or None = all
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT2 = Path(__file__).parent

# ── Load neuron data ──────────────────────────────────────────────────────────
print("Loading neuron data...")
with open(EXPERIMENT2 / 'neuron_data.json') as f:
    neuron_data = json.load(f)

all_units = [n[0]['unit'] for n in neuron_data if n]
spike_trains = {
    n[0]['unit']: np.array([s['spikeTime'] for s in n], dtype=np.float64)
    for n in neuron_data if n
}

unit_list = [u for u in (NEURONS if NEURONS is not None else all_units)
             if u in spike_trains]
N = len(unit_list)
print(f"  {N} neurons")

# ── Load config times ─────────────────────────────────────────────────────────
cfg_df = pd.read_csv(EXPERIMENT2 / 'Configurations' / 'config_times.csv')
if CONFIG_IDS is not None:
    cfg_df = cfg_df[cfg_df['config'].isin(CONFIG_IDS)].reset_index(drop=True)
cfg_df = cfg_df.sort_values('config').reset_index(drop=True)
if N_CONFIGS is not None:
    cfg_df = cfg_df.iloc[:N_CONFIGS].reset_index(drop=True)

# ── Verify all configs have the same duration ─────────────────────────────────
cfg_df['seq_len'] = ((cfg_df['t_end'] - cfg_df['t_start']) * 1000 / BIN_MS).astype(int)
unique_lens = cfg_df['seq_len'].unique()

if len(unique_lens) == 1:
    SEQ_LEN = int(unique_lens[0])
    print(f"  All {len(cfg_df)} configs same length: {SEQ_LEN} bins × {BIN_MS}ms = {SEQ_LEN * BIN_MS / 1000:.1f}s ✓")
else:
    SEQ_LEN = int(cfg_df['seq_len'].min())
    print(f"  WARNING: mixed lengths {unique_lens} — truncating all to {SEQ_LEN} bins")

# ── Build array: (num_configs, num_units, seq_len) ────────────────────────────
num_configs = len(cfg_df)
data = np.zeros((num_configs, N, SEQ_LEN), dtype=np.float32)

print(f"Building array ({num_configs} × {N} × {SEQ_LEN})...")
for c_idx, row in cfg_df.iterrows():
    t0 = float(row['t_start'])
    for n_idx, unit in enumerate(unit_list):
        spikes = spike_trains[unit]
        spikes = spikes[(spikes >= t0) & (spikes < t0 + SEQ_LEN * BIN_MS * 1e-3)]
        bins   = ((spikes - t0) * 1000 / BIN_MS).astype(int)
        bins   = bins[(bins >= 0) & (bins < SEQ_LEN)]
        np.add.at(data[c_idx, n_idx], bins, 1)

print(f"  Done. shape={data.shape}  total spikes={int(data.sum()):,}")

out = EXPERIMENT2 / 'spike_data_1ms_binned.npy'
np.save(out, data)
print(f"Saved: {out}")
