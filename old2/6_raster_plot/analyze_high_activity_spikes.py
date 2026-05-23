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

print(f"Total neurons: {N}")
print(f"Total spikes: {len(all_spikes)}")
print(f"Time range: {t_min:.2f} to {t_max:.2f} seconds")
print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")

# Calculate number of active neurons per 1ms bin for entire recording
# Convert duration to milliseconds to get number of bins
duration_ms = int(duration * 1000)
n_bins = duration_ms
bin_width_sec = 0.001  # 1ms in seconds

# Create bins
bins = np.linspace(t_min, t_max, n_bins + 1)

print(f"\nCalculating active neurons per bin for {n_bins} bins...")

active_neurons_per_bin = []
spikes_per_bin = []

for i in range(n_bins):
    if i % 1000 == 0:
        print(f"  Processing bin {i}/{n_bins}...")

    bin_start = bins[i]
    bin_end = bins[i + 1]

    # Count how many neurons have at least one spike in this bin
    active_count = 0
    spike_count = 0

    for neuron_id in range(N):
        times = spike_times[neuron_id]
        if len(times) > 0:
            spikes_in_bin = np.sum((times >= bin_start) & (times < bin_end))
            spike_count += spikes_in_bin
            if spikes_in_bin > 0:
                active_count += 1

    active_neurons_per_bin.append(active_count)
    spikes_per_bin.append(spike_count)

active_neurons_per_bin = np.array(active_neurons_per_bin)
spikes_per_bin = np.array(spikes_per_bin)

print("\nAnalyzing high-activity spikes (activity >= 15 neurons)...")

# Calculate spikes in bins with >= 15 active neurons
bins_gte_15 = active_neurons_per_bin >= 15
spikes_in_high_activity = np.sum(spikes_per_bin[bins_gte_15])
spikes_in_low_activity = np.sum(spikes_per_bin[~bins_gte_15])

n_bins_gte_15 = np.sum(bins_gte_15)
n_bins_lt_15 = np.sum(~bins_gte_15)

print("\n" + "="*60)
print("HIGH-ACTIVITY SPIKE ANALYSIS (>= 15 ACTIVE NEURONS)")
print("="*60)

print(f"\nBins with >= 15 active neurons (HIGH ACTIVITY - KEPT):")
print(f"  Number of bins: {n_bins_gte_15} ({n_bins_gte_15/n_bins*100:.2f}%)")
print(f"  Total spikes: {spikes_in_high_activity} ({spikes_in_high_activity/len(all_spikes)*100:.2f}%)")

print(f"\nBins with < 15 active neurons (LOW ACTIVITY - EXCLUDED):")
print(f"  Number of bins: {n_bins_lt_15} ({n_bins_lt_15/n_bins*100:.2f}%)")
print(f"  Total spikes: {spikes_in_low_activity} ({spikes_in_low_activity/len(all_spikes)*100:.2f}%)")

print(f"\nVerification:")
print(f"  Total spikes (sum): {spikes_in_low_activity + spikes_in_high_activity}")
print(f"  Total spikes (expected): {len(all_spikes)}")

# Additional breakdown for high activity bins
print("\n" + "="*60)
print("DETAILED HIGH-ACTIVITY BREAKDOWN")
print("="*60)

for threshold in [15, 18, 20, 23, 25, 28]:
    bins_gte_thresh = active_neurons_per_bin >= threshold
    spikes_gte_thresh = np.sum(spikes_per_bin[bins_gte_thresh])
    print(f"Activity >= {threshold:2d} neurons: {spikes_gte_thresh:6d} spikes ({spikes_gte_thresh/len(all_spikes)*100:5.2f}%)")

print("\n" + "="*60)

# Save results
results = {
    'total_spikes': int(len(all_spikes)),
    'total_bins': int(n_bins),
    'duration_seconds': float(duration),
    'spikes_with_activity_lt_15': int(spikes_in_low_activity),
    'spikes_with_activity_gte_15': int(spikes_in_high_activity),
    'percentage_spikes_lt_15': float(spikes_in_low_activity / len(all_spikes) * 100),
    'percentage_spikes_gte_15': float(spikes_in_high_activity / len(all_spikes) * 100),
    'bins_with_activity_lt_15': int(n_bins_lt_15),
    'bins_with_activity_gte_15': int(n_bins_gte_15),
    'percentage_bins_lt_15': float(n_bins_lt_15 / n_bins * 100),
    'percentage_bins_gte_15': float(n_bins_gte_15 / n_bins * 100)
}

with open('high_activity_spikes_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved to: high_activity_spikes_summary.json")
print("\nDone!")
