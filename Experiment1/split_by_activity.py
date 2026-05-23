import json
import numpy as np

# ─────────────────────────────────────────────
# 1. Load burst intervals
# ─────────────────────────────────────────────
print("Loading burst data...")

with open('Raster_Plot/bursts.json', 'r') as f:
    burst_data = json.load(f)

bursts       = burst_data['bursts']
burst_starts = np.array([b['t_start_s'] for b in bursts])
burst_ends   = np.array([b['t_end_s']   for b in bursts])

print(f"  {len(bursts)} bursts loaded")
print(f"  Parameters: threshold={burst_data['threshold']:.2f}, "
      f"min_duration={burst_data['min_burst_duration_ms']}ms, "
      f"merge_gap={burst_data['merge_gap_ms']}ms")

# ─────────────────────────────────────────────
# 2. Load original spike data
# ─────────────────────────────────────────────
print("\nLoading neuron data...")

with open('data/neuron_data.json', 'r') as f:
    neuron_data = json.load(f)

N            = len(neuron_data)
total_spikes = sum(len(neuron) for neuron in neuron_data)
print(f"  {N} neurons, {total_spikes:,} total spikes")

# ─────────────────────────────────────────────
# 3. Assign each spike to within-burst or between-burst
#
#    For each spike at time t:
#    - Find the last burst that started at or before t (binary search)
#    - If t also falls before that burst's end → within burst
#    - Otherwise → between burst
# ─────────────────────────────────────────────
print("\nSplitting spikes by burst membership...")

between_burst = []
within_burst  = []

for neuron in neuron_data:
    neuron_between = []
    neuron_within  = []

    for spike in neuron:
        t   = spike['spikeTime']
        idx = np.searchsorted(burst_starts, t, side='right') - 1

        if idx >= 0 and t <= burst_ends[idx]:
            neuron_within.append(spike)
        else:
            neuron_between.append(spike)

    between_burst.append(neuron_between)
    within_burst.append(neuron_within)

# ─────────────────────────────────────────────
# 4. Print summary
# ─────────────────────────────────────────────
n_within  = sum(len(n) for n in within_burst)
n_between = sum(len(n) for n in between_burst)

print(f"\n  Within burst:   {n_within:,} spikes  ({100*n_within/total_spikes:.1f}%)")
print(f"  Between burst:  {n_between:,} spikes  ({100*n_between/total_spikes:.1f}%)")

for neuron, nb, nw in zip(neuron_data, between_burst, within_burst):
    if not neuron:
        continue
    nid = neuron[0]['unit']
    print(f"    Neuron {nid:3d}: {len(nw):6,} within   {len(nb):6,} between")

# ─────────────────────────────────────────────
# 5. Save
# ─────────────────────────────────────────────
print("\nSaving...")

with open('Within_Burst/neuron_data.json', 'w') as f:
    json.dump(within_burst, f)
print("  Saved: Within_Burst/neuron_data.json")

with open('Between_Burst/neuron_data.json', 'w') as f:
    json.dump(between_burst, f)
print("  Saved: Between_Burst/neuron_data.json")

print("\nDone.")
