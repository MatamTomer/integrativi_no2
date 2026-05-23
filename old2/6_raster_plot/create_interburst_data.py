import json
import numpy as np

print("Loading neuron data...")
with open('../data/neuron_data.json', 'r') as f:
    neuron_data = json.load(f)

N = len(neuron_data)
spike_times = [np.array([s['spikeTime'] for s in n]) for n in neuron_data]

all_spikes = np.concatenate([t for t in spike_times if len(t) > 0])
t_min = all_spikes.min()
t_max = all_spikes.max()

# Build full 1ms grid
bin_width = 0.001  # 1ms in seconds
bins = np.arange(t_min, t_max + bin_width, bin_width)
n_bins = len(bins) - 1
print(f"Total 1ms bins: {n_bins:,}")

# Compute active_in_window[t] = neurons with >=1 spike in [t-100ms, t+100ms]
half_window = 100  # ms
threshold = 10

print(f"Computing activity with +-{half_window}ms sliding window...")
active_in_window = np.zeros(n_bins, dtype=int)

for neuron_id, times in enumerate(spike_times):
    if len(times) == 0:
        continue
    if neuron_id % 5 == 0:
        print(f"  Neuron {neuron_id+1}/{N}...")
    counts, _ = np.histogram(times, bins=bins)
    spike_indicator = (counts > 0).astype(np.int32)
    cs = np.zeros(n_bins + 1, dtype=np.int32)
    cs[1:] = np.cumsum(spike_indicator)
    lo = np.clip(np.arange(n_bins) - half_window, 0, n_bins)
    hi = np.clip(np.arange(n_bins) + half_window + 1, 0, n_bins)
    rolling = cs[hi] - cs[lo]
    active_in_window += (rolling > 0).astype(int)

print(f"Done.")

# Filter: keep spikes in bins where active_in_window <= threshold
print(f"\nFiltering spikes (keeping activity <= {threshold})...")

filtered_neuron_data = []
total_kept = 0
total_removed = 0

for neuron_id in range(N):
    original_spikes = neuron_data[neuron_id]
    filtered_spikes = []

    for spike in original_spikes:
        spike_time = spike['spikeTime']
        bin_idx = int((spike_time - t_min) / bin_width)
        bin_idx = min(bin_idx, n_bins - 1)

        if active_in_window[bin_idx] <= threshold:
            filtered_spikes.append(spike)
            total_kept += 1
        else:
            total_removed += 1

    filtered_neuron_data.append(filtered_spikes)

total_spikes = total_kept + total_removed
print(f"\nResults:")
print(f"  Total spikes:   {total_spikes:,}")
print(f"  Kept (<=10):    {total_kept:,} ({total_kept/total_spikes*100:.2f}%)")
print(f"  Removed (>10):  {total_removed:,} ({total_removed/total_spikes*100:.2f}%)")

output_file = '../data/neuron_data_interburst_lte10.json'
print(f"\nSaving to {output_file}...")
with open(output_file, 'w') as f:
    json.dump(filtered_neuron_data, f)
print("Done.")
