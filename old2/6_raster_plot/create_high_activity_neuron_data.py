import json
import numpy as np

print("Loading neuron data...")
with open('../data/neuron_data.json', 'r') as f:
    neuron_data = json.load(f)

N = len(neuron_data)

# Extract spike times for each neuron
spike_times = []
for neuron_id in range(N):
    times = [spike['spikeTime'] for spike in neuron_data[neuron_id]]
    spike_times.append(np.array(times))

# Get time range for entire recording
all_spikes = np.concatenate([times for times in spike_times if len(times) > 0])
t_min = np.min(all_spikes)
t_max = np.max(all_spikes)
duration = t_max - t_min

print(f"Original data:")
print(f"  Total neurons: {N}")
print(f"  Total spikes: {len(all_spikes)}")
print(f"  Duration: {duration:.2f} ms ({duration/1000:.2f} seconds)")

# Calculate number of active neurons per 1ms bin for entire recording
bin_width = 1  # ms
bins = np.arange(t_min, t_max + bin_width, bin_width)
n_bins = len(bins) - 1

print(f"\nCalculating active neurons per bin for {n_bins} bins...")

active_neurons_per_bin = []

for i in range(n_bins):
    if i % 1000 == 0:
        print(f"  Processing bin {i}/{n_bins}...")

    bin_start = bins[i]
    bin_end = bins[i + 1]

    # Count how many neurons have at least one spike in this bin
    active_count = 0

    for neuron_id in range(N):
        times = spike_times[neuron_id]
        if len(times) > 0:
            spikes_in_bin = np.sum((times >= bin_start) & (times < bin_end))
            if spikes_in_bin > 0:
                active_count += 1

    active_neurons_per_bin.append(active_count)

active_neurons_per_bin = np.array(active_neurons_per_bin)

print("\nFiltering spikes where activity >= 15 neurons...")

# Create filtered neuron data (KEEP high-activity spikes)
filtered_neuron_data = []
total_spikes_kept = 0
total_spikes_removed = 0

for neuron_id in range(N):
    print(f"  Filtering neuron {neuron_id + 1}/{N}...")

    original_spikes = neuron_data[neuron_id]
    filtered_spikes = []

    for spike in original_spikes:
        spike_time = spike['spikeTime']

        # Find which bin this spike belongs to
        bin_idx = int((spike_time - t_min) / bin_width)

        # Handle edge case
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1

        # Keep spike only if activity in that bin was >= 15
        if active_neurons_per_bin[bin_idx] >= 15:
            filtered_spikes.append(spike)
            total_spikes_kept += 1
        else:
            total_spikes_removed += 1

    filtered_neuron_data.append(filtered_spikes)
    print(f"    Neuron {neuron_id + 1}: {len(original_spikes)} -> {len(filtered_spikes)} spikes "
          f"({len(filtered_spikes)/len(original_spikes)*100:.1f}% kept)" if len(original_spikes) > 0 else f"    Neuron {neuron_id + 1}: 0 spikes")

# Save filtered data
output_file = '../data/neuron_data_high_activity_gte15.json'
print(f"\nSaving high-activity filtered data to {output_file}...")
with open(output_file, 'w') as f:
    json.dump(filtered_neuron_data, f, indent=2)

print("\n" + "="*60)
print("HIGH-ACTIVITY FILTERING SUMMARY")
print("="*60)
print(f"Original total spikes: {len(all_spikes)}")
print(f"Spikes kept (activity >= 15): {total_spikes_kept} ({total_spikes_kept/len(all_spikes)*100:.2f}%)")
print(f"Spikes removed (activity < 15): {total_spikes_removed} ({total_spikes_removed/len(all_spikes)*100:.2f}%)")
print(f"\nFiltered data saved to: {output_file}")

# Create summary file
summary = {
    'original_total_spikes': int(len(all_spikes)),
    'filtered_total_spikes': int(total_spikes_kept),
    'removed_spikes': int(total_spikes_removed),
    'percentage_kept': float(total_spikes_kept / len(all_spikes) * 100),
    'percentage_removed': float(total_spikes_removed / len(all_spikes) * 100),
    'filter_criterion': 'activity >= 15 active neurons per 1ms bin',
    'n_neurons': N,
    'spikes_per_neuron': [len(neuron_spikes) for neuron_spikes in filtered_neuron_data]
}

with open('high_activity_data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved to: filtered_data_summary.json")
print("\nDone!")
