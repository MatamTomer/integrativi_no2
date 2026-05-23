import json
import numpy as np
import matplotlib.pyplot as plt

print("Loading neuron data...")
with open('../data/neuron_data.json', 'r') as f:
    neuron_data = json.load(f)

N = len(neuron_data)

# Extract spike times for each neuron
spike_times = []
for neuron_id in range(N):
    times = [spike['spikeTime'] for spike in neuron_data[neuron_id]]
    spike_times.append(np.array(times))

# Get time range
all_spikes = np.concatenate([times for times in spike_times if len(times) > 0])
t_min = np.min(all_spikes)
t_max = np.max(all_spikes)

# Define 1 second window
zoom_duration = 1000  # 1 second in ms
t_zoom_end = t_min + zoom_duration

print(f"Creating 1-second raster plot with number of active neurons...")
print(f"Time window: {t_min:.2f} to {t_zoom_end:.2f} ms")

# Filter spikes within 1 second window
spikes_in_window = []
for neuron_id in range(N):
    times = spike_times[neuron_id]
    if len(times) > 0:
        times_zoom = times[(times >= t_min) & (times < t_zoom_end)]
        spikes_in_window.append(times_zoom)

# Calculate number of active neurons per 1ms bin
bin_width = 1  # ms
bins = np.arange(t_min, t_zoom_end + bin_width, bin_width)
n_bins = len(bins) - 1

# For each bin, count how many neurons fired at least once
active_neurons_per_bin = []
for i in range(n_bins):
    bin_start = bins[i]
    bin_end = bins[i + 1]

    # Count how many neurons have at least one spike in this bin
    active_count = 0
    for neuron_id in range(N):
        times_zoom = spikes_in_window[neuron_id]
        if len(times_zoom) > 0:
            spikes_in_bin = np.sum((times_zoom >= bin_start) & (times_zoom < bin_end))
            if spikes_in_bin > 0:
                active_count += 1

    active_neurons_per_bin.append(active_count)

active_neurons_per_bin = np.array(active_neurons_per_bin)

print(f"Number of active neurons per 1ms bin calculated")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                gridspec_kw={'height_ratios': [3, 1]},
                                sharex=True)

# Top: Raster plot
for neuron_id in range(N):
    times_zoom = spikes_in_window[neuron_id]
    if len(times_zoom) > 0:
        ax1.scatter(times_zoom, np.ones(len(times_zoom)) * (neuron_id + 1),
                   marker='|', s=50, linewidths=1.5, color='black', alpha=0.9)

ax1.set_ylabel('Neuron ID', fontsize=14)
ax1.set_yticks(range(1, N + 1))
ax1.set_ylim(0.5, N + 0.5)
ax1.set_title(f'Raster Plot: First 1 Second (High Resolution)\n'
              f'All 29 Neurons with Number of Active Neurons',
              fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Bottom: Number of active neurons
bin_centers = bins[:-1] + bin_width / 2
ax2.fill_between(bin_centers, active_neurons_per_bin,
                 color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Time (ms)', fontsize=14)
ax2.set_ylabel('Number of\nActive Neurons\n(per 1ms bin)', fontsize=12)
ax2.set_xlim(t_min, t_zoom_end)
ax2.set_ylim(0, N)
ax2.grid(True, alpha=0.3)

# Add statistics
mean_active = np.mean(active_neurons_per_bin)
max_active = np.max(active_neurons_per_bin)
ax2.axhline(mean_active, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'Mean: {mean_active:.1f}')
ax2.legend(loc='upper right')

# Add reference line for all neurons
ax2.axhline(N, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax2.text(t_min + 10, N - 0.5, f'All {N} neurons', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig('raster_1sec_active_neurons.png', dpi=150)
print("Saved: raster_1sec_active_neurons.png")
plt.close()

print(f"\nActive neurons statistics (1ms bins):")
print(f"  Mean: {mean_active:.2f} neurons active per ms")
print(f"  Max: {max_active} neurons (out of {N})")
print(f"  Min: {np.min(active_neurons_per_bin)} neurons")
print(f"  Std: {np.std(active_neurons_per_bin):.2f}")

# Calculate percentage of time with different activity levels
bins_all_active = np.sum(active_neurons_per_bin == N)
bins_high_active = np.sum(active_neurons_per_bin >= 20)
bins_medium_active = np.sum((active_neurons_per_bin >= 10) & (active_neurons_per_bin < 20))
bins_low_active = np.sum(active_neurons_per_bin < 10)

print(f"\nActivity distribution:")
print(f"  All {N} neurons active: {bins_all_active} bins ({bins_all_active/n_bins*100:.1f}%)")
print(f"  High activity (>=20 neurons): {bins_high_active} bins ({bins_high_active/n_bins*100:.1f}%)")
print(f"  Medium activity (10-19 neurons): {bins_medium_active} bins ({bins_medium_active/n_bins*100:.1f}%)")
print(f"  Low activity (<10 neurons): {bins_low_active} bins ({bins_low_active/n_bins*100:.1f}%)")

print("\nDone!")
