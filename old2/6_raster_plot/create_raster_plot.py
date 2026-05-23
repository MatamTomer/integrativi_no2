import json
import numpy as np
import matplotlib.pyplot as plt

print("Loading neuron data...")
with open('../data/neuron_data.json', 'r') as f:
    neuron_data = json.load(f)

N = len(neuron_data)
print(f"Number of neurons: {N}")

# Extract spike times for each neuron
spike_times = []
for neuron_id in range(N):
    times = [spike['spikeTime'] for spike in neuron_data[neuron_id]]
    spike_times.append(np.array(times))
    print(f"Neuron {neuron_id + 1}: {len(times)} spikes")

# Get time range
all_spikes = np.concatenate([times for times in spike_times if len(times) > 0])
t_min = np.min(all_spikes)
t_max = np.max(all_spikes)
duration = t_max - t_min

print(f"\nTime range: {t_min:.2f} to {t_max:.2f} seconds")
print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")

# Create full raster plot
print("\nCreating full raster plot...")
fig, ax = plt.subplots(figsize=(16, 10))

for neuron_id in range(N):
    times = spike_times[neuron_id]
    if len(times) > 0:
        # Plot spikes as vertical lines
        ax.scatter(times, np.ones(len(times)) * (neuron_id + 1),
                  marker='|', s=10, linewidths=0.5, color='black', alpha=0.6)

ax.set_xlabel('Time (seconds)', fontsize=14)
ax.set_ylabel('Neuron ID', fontsize=14)
ax.set_yticks(range(1, N + 1))
ax.set_ylim(0.5, N + 0.5)
ax.set_xlim(t_min, t_max)
ax.set_title(f'Raster Plot: All 29 Neurons\n'
             f'Duration: {duration:.2f} seconds ({duration/60:.1f} min), Total spikes: {len(all_spikes)}',
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('raster_plot_full.png', dpi=150)
print("Saved: raster_plot_full.png")
plt.close()

# Create zoomed raster plot (first 10 seconds)
print("\nCreating zoomed raster plot (first 10 seconds)...")
zoom_duration = 10.0  # 10 seconds
t_zoom_end = t_min + zoom_duration

fig, ax = plt.subplots(figsize=(16, 10))

for neuron_id in range(N):
    times = spike_times[neuron_id]
    if len(times) > 0:
        # Filter spikes within zoom window
        times_zoom = times[(times >= t_min) & (times < t_zoom_end)]
        if len(times_zoom) > 0:
            ax.scatter(times_zoom, np.ones(len(times_zoom)) * (neuron_id + 1),
                      marker='|', s=20, linewidths=0.8, color='black', alpha=0.8)

ax.set_xlabel('Time (seconds)', fontsize=14)
ax.set_ylabel('Neuron ID', fontsize=14)
ax.set_yticks(range(1, N + 1))
ax.set_ylim(0.5, N + 0.5)
ax.set_xlim(t_min, t_zoom_end)
ax.set_title(f'Raster Plot: First 10 Seconds\n'
             f'All 29 Neurons',
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('raster_plot_10sec.png', dpi=150)
print("Saved: raster_plot_10sec.png")
plt.close()

# Create very zoomed raster plot (first 1 second)
print("\nCreating very zoomed raster plot (first 1 second)...")
zoom_duration = 1.0  # 1 second
t_zoom_end = t_min + zoom_duration

fig, ax = plt.subplots(figsize=(16, 10))

for neuron_id in range(N):
    times = spike_times[neuron_id]
    if len(times) > 0:
        # Filter spikes within zoom window
        times_zoom = times[(times >= t_min) & (times < t_zoom_end)]
        if len(times_zoom) > 0:
            ax.scatter(times_zoom, np.ones(len(times_zoom)) * (neuron_id + 1),
                      marker='|', s=50, linewidths=1.5, color='black', alpha=0.9)

ax.set_xlabel('Time (seconds)', fontsize=14)
ax.set_ylabel('Neuron ID', fontsize=14)
ax.set_yticks(range(1, N + 1))
ax.set_ylim(0.5, N + 0.5)
ax.set_xlim(t_min, t_zoom_end)
ax.set_title(f'Raster Plot: First 1 Second\n'
             f'All 29 Neurons - High temporal resolution',
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('raster_plot_1sec.png', dpi=150)
print("Saved: raster_plot_1sec.png")
plt.close()

# Create raster plot with population firing rate
print("\nCreating raster plot with population firing rate histogram...")

# Calculate population firing rate in 10ms bins
bin_width = 0.01  # 10ms in seconds
bins = np.arange(t_min, t_max + bin_width, bin_width)
pop_rate, _ = np.histogram(all_spikes, bins=bins)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12),
                                gridspec_kw={'height_ratios': [3, 1]},
                                sharex=True)

# Top: Raster plot
for neuron_id in range(N):
    times = spike_times[neuron_id]
    if len(times) > 0:
        ax1.scatter(times, np.ones(len(times)) * (neuron_id + 1),
                   marker='|', s=10, linewidths=0.5, color='black', alpha=0.6)

ax1.set_ylabel('Neuron ID', fontsize=14)
ax1.set_yticks(range(1, N + 1))
ax1.set_ylim(0.5, N + 0.5)
ax1.set_title(f'Raster Plot with Population Firing Rate\n'
              f'Duration: {duration:.2f} seconds ({duration/60:.1f} min), Total spikes: {len(all_spikes)}',
              fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Bottom: Population firing rate
bin_centers = bins[:-1] + bin_width / 2
ax2.fill_between(bin_centers, pop_rate, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Time (seconds)', fontsize=14)
ax2.set_ylabel('Population\nFiring Rate\n(spikes/10ms)', fontsize=12)
ax2.set_xlim(t_min, t_max)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('raster_plot_with_population_rate.png', dpi=150)
print("Saved: raster_plot_with_population_rate.png")
plt.close()

# Create raster plot sorted by average firing rate
print("\nCreating raster plot sorted by firing rate...")

# Calculate firing rates
firing_rates = []
for neuron_id in range(N):
    rate = len(spike_times[neuron_id]) / duration  # spikes per second
    firing_rates.append((neuron_id + 1, rate))

# Sort by firing rate (descending)
firing_rates_sorted = sorted(firing_rates, key=lambda x: x[1], reverse=True)
neuron_order = [x[0] for x in firing_rates_sorted]

fig, ax = plt.subplots(figsize=(16, 10))

for plot_idx, neuron_id in enumerate(neuron_order):
    times = spike_times[neuron_id - 1]  # Convert to 0-indexed
    if len(times) > 0:
        ax.scatter(times, np.ones(len(times)) * (plot_idx + 1),
                  marker='|', s=10, linewidths=0.5, color='black', alpha=0.6)

ax.set_xlabel('Time (seconds)', fontsize=14)
ax.set_ylabel('Neuron ID (sorted by firing rate)', fontsize=14)
ax.set_yticks(range(1, N + 1))
ax.set_yticklabels([str(n) for n in neuron_order])
ax.set_ylim(0.5, N + 0.5)
ax.set_xlim(t_min, t_max)
ax.set_title(f'Raster Plot: Sorted by Firing Rate (High to Low)\n'
             f'Duration: {duration:.2f} seconds ({duration/60:.1f} min)',
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add firing rate labels on right side
ax2 = ax.twinx()
ax2.set_ylim(0.5, N + 0.5)
ax2.set_yticks(range(1, N + 1))
ax2.set_yticklabels([f'{firing_rates_sorted[i][1]:.1f} Hz' for i in range(N)], fontsize=8)
ax2.set_ylabel('Firing Rate', fontsize=12)

plt.tight_layout()
plt.savefig('raster_plot_sorted.png', dpi=150)
print("Saved: raster_plot_sorted.png")
plt.close()

print("\nDone! Created 5 raster plots:")
print("  1. raster_plot_full.png - Full recording")
print("  2. raster_plot_10sec.png - First 10 seconds")
print("  3. raster_plot_1sec.png - First 1 second (high detail)")
print("  4. raster_plot_with_population_rate.png - With population firing rate")
print("  5. raster_plot_sorted.png - Sorted by firing rate")
