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

print(f"Creating 1-second raster plot with population firing rate...")
print(f"Time window: {t_min:.2f} to {t_zoom_end:.2f} ms")

# Filter spikes within 1 second window
spikes_in_window = []
for neuron_id in range(N):
    times = spike_times[neuron_id]
    if len(times) > 0:
        times_zoom = times[(times >= t_min) & (times < t_zoom_end)]
        spikes_in_window.append(times_zoom)

# Get all spikes in window for population rate
all_spikes_zoom = np.concatenate([times for times in spikes_in_window if len(times) > 0])

# Calculate population firing rate in 1ms bins
bin_width = 1  # ms
bins = np.arange(t_min, t_zoom_end + bin_width, bin_width)
pop_rate, _ = np.histogram(all_spikes_zoom, bins=bins)

print(f"Total spikes in 1-second window: {len(all_spikes_zoom)}")

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
              f'All 29 Neurons with Population Firing Rate',
              fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Bottom: Population firing rate
bin_centers = bins[:-1] + bin_width / 2
ax2.fill_between(bin_centers, pop_rate, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Time (ms)', fontsize=14)
ax2.set_ylabel('Population\nFiring Rate\n(spikes/ms)', fontsize=12)
ax2.set_xlim(t_min, t_zoom_end)
ax2.grid(True, alpha=0.3)

# Add statistics
mean_rate = np.mean(pop_rate)
max_rate = np.max(pop_rate)
ax2.axhline(mean_rate, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_rate:.1f}')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('raster_1sec_with_population_rate.png', dpi=150)
print("Saved: raster_1sec_with_population_rate.png")
plt.close()

print(f"\nPopulation firing rate statistics (1ms bins):")
print(f"  Mean: {mean_rate:.2f} spikes/ms")
print(f"  Max: {max_rate} spikes/ms")
print(f"  Min: {np.min(pop_rate)} spikes/ms")
print(f"  Std: {np.std(pop_rate):.2f} spikes/ms")
print("\nDone!")
