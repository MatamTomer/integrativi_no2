import json
import numpy as np
import matplotlib.pyplot as plt
import sys

# Get duration from command line argument
duration_sec = int(sys.argv[1]) if len(sys.argv) > 1 else None

print("Loading high-activity neuron data...")
with open('../data/neuron_data_high_activity_gte15.json', 'r') as f:
    neuron_data = json.load(f)

N = len(neuron_data)

# Extract all spike times
all_spikes = []
for neuron_id in range(N):
    for spike in neuron_data[neuron_id]:
        all_spikes.append(spike['spikeTime'])

all_spikes = np.array(all_spikes)
all_spikes.sort()

# Get time range
t_min = np.min(all_spikes)
t_max = np.max(all_spikes)
total_duration = t_max - t_min

print(f"Time range: {t_min:.3f} to {t_max:.3f} seconds")
print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
print(f"Total spikes in high-activity data: {len(all_spikes)}")

# Set duration
if duration_sec is None:
    t_start = t_min
    t_end = t_max
    label = f"Full Recording ({total_duration/60:.1f} min)"
    filename = "network_burst_timing_full_recording.png"
    figsize = (30, 4)
else:
    t_start = t_min
    t_end = min(t_min + duration_sec, t_max)
    label = f"First {duration_sec} Seconds"
    filename = f"network_burst_timing_first_{duration_sec}sec.png"
    if duration_sec <= 100:
        figsize = (20, 4)
    else:
        figsize = (30, 4)

actual_duration = t_end - t_start

print(f"\nAnalyzing {label}")
print(f"Duration: {actual_duration:.2f} seconds")

# Filter spikes to time range
spikes_in_range = all_spikes[(all_spikes >= t_start) & (all_spikes < t_end)]
print(f"Spikes in range: {len(spikes_in_range)}")

# Create visualization using vertical lines (much faster than bars)
print("\nCreating visualization...")
fig, ax = plt.subplots(figsize=figsize)

# Convert spike times to relative seconds
spike_times_rel = spikes_in_range - t_start

# Plot as vertical lines
ax.vlines(spike_times_rel, 0, 1, colors='red', alpha=0.5, linewidths=0.5)

ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Spike Activity', fontsize=12)
ax.set_title(f'Spike Timing During High-Activity Periods - {label}\n(Spikes from 1-second bins with ≥15 active neurons)', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.2])
ax.set_xlim([0, actual_duration])
ax.set_yticks([0, 1])
ax.set_yticklabels(['', 'Spikes'])
ax.grid(axis='x', alpha=0.3)

# Add text with statistics
n_spikes = len(spikes_in_range)
avg_rate = n_spikes / actual_duration
ax.text(0.98, 0.95, f'{n_spikes} spikes\n{avg_rate:.2f} spikes/sec avg',
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
dpi = 150 if duration_sec is None or duration_sec > 100 else 300
plt.savefig(filename, dpi=dpi, bbox_inches='tight')
print(f"\nVisualization saved to: {filename}")

print("\nDone!")
