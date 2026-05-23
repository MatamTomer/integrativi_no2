import json
import numpy as np
import matplotlib.pyplot as plt

print("Loading neuron data...")
with open('../data/neuron_data.json', 'r') as f:
    neuron_data = json.load(f)

N = len(neuron_data)
spike_times = [np.array([s['spikeTime'] for s in n]) for n in neuron_data]

all_spikes = np.concatenate([t for t in spike_times if len(t) > 0])
t_min = all_spikes.min()
t_max = all_spikes.max()

# Build full 1ms grid for the entire recording
bin_width = 0.001  # 1ms in seconds
bins = np.arange(t_min, t_max + bin_width, bin_width)
n_bins = len(bins) - 1
print(f"Total 1ms bins: {n_bins:,}  ({n_bins/1000:.1f} seconds)")

# For each 1ms bin t, count how many neurons had >= 1 spike in [t-100ms, t+100ms]
# Using cumsum trick: rolling sum over window of 201 bins (+-100ms)
half_window = 100  # ms
window = 2 * half_window + 1  # 201 bins

print(f"Computing activity with +-{half_window}ms sliding window...")

# active_in_window[t] = number of neurons active in [t-100ms, t+100ms]
active_in_window = np.zeros(n_bins, dtype=int)
spikes_per_bin = np.zeros(n_bins, dtype=int)

for neuron_id, times in enumerate(spike_times):
    if len(times) == 0:
        continue
    if neuron_id % 5 == 0:
        print(f"  Neuron {neuron_id+1}/{N}...")

    # Bin spikes into the 1ms grid
    counts, _ = np.histogram(times, bins=bins)
    spike_indicator = (counts > 0).astype(np.int32)
    spikes_per_bin += counts

    # Rolling sum using cumsum trick
    cs = np.zeros(n_bins + 1, dtype=np.int32)
    cs[1:] = np.cumsum(spike_indicator)

    lo = np.clip(np.arange(n_bins) - half_window, 0, n_bins)
    hi = np.clip(np.arange(n_bins) + half_window + 1, 0, n_bins)
    rolling = cs[hi] - cs[lo]

    # 1 if this neuron was active in the window around this bin
    active_in_window += (rolling > 0).astype(int)

print(f"Done. Max active neurons in +-{half_window}ms window: {active_in_window.max()}")
print(f"Bins with >= 15 active neurons: {np.sum(active_in_window >= 15):,}")

# Find the 10-second window with the most burst bins
chunk_bins = 10000
best_start = 0
best_count = 0
for i in range(0, n_bins - chunk_bins, 1000):
    c = np.sum(active_in_window[i:i+chunk_bins] >= 15)
    if c > best_count:
        best_count = c
        best_start = i

print(f"Best 10s window: bin {best_start} (offset={bins[best_start]-t_min:.1f}s), {best_count} bins >= 15")

# Extract window
win_activity = active_in_window[best_start:best_start + chunk_bins]
time_ms = np.arange(chunk_bins)  # relative ms

# Plot
threshold = 15
colors = ['red' if v >= threshold else '#4a90d9' for v in win_activity]

fig, ax = plt.subplots(figsize=(18, 4))
ax.bar(time_ms, win_activity, width=1.0, color=colors, linewidth=0)
ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.2, alpha=0.8,
           label=f'Threshold: {threshold} active neurons')

offset_s = bins[best_start] - t_min
ax.set_xlabel('Time (ms, relative to window start)', fontsize=12)
ax.set_ylabel(f'Active neurons in +-{half_window}ms window', fontsize=12)
ax.set_title(
    f'Network Activity -- 10-second window (recording offset ~{offset_s:.0f}s)\n'
    f'Each bar = # neurons with >=1 spike in +-{half_window}ms   |   '
    f'Red = Network Spike (>={threshold} active neurons)',
    fontsize=12
)
ax.set_xlim([0, chunk_bins])
ax.set_ylim([0, N + 1])
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('network_activity_window.png', dpi=150, bbox_inches='tight')
print("Saved: network_activity_window.png")
