import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

print("Loading neuron data...")
with open('neuron_data.json', 'r') as f:
    neuron_data = json.load(f)

# Extract spike times for each neuron
N = len(neuron_data)
spike_times = []
for neuron_id in range(N):
    times = [spike['spikeTime'] for spike in neuron_data[neuron_id]]
    spike_times.append(np.array(times))
    print(f"Neuron {neuron_id + 1}: {len(times)} spikes")

print(f"\nTotal neurons: {N}")
print(f"Total spikes: {sum(len(times) for times in spike_times)}")

# Define the window for network activity calculation
window_half_width = 10.0  # +/- 10 ms around each spike

print("\nCalculating network activity for each spike...")

# For each neuron, store the network activity level at each of its spikes
neuron_activity_histograms = []

for neuron_id in range(N):
    print(f"\nProcessing neuron {neuron_id + 1}/{N}...")

    # Get spikes for this neuron
    neuron_spikes = spike_times[neuron_id]

    # For each spike of this neuron, calculate network activity
    network_activities = []

    for spike_idx, spike_time in enumerate(neuron_spikes):
        if spike_idx % 1000 == 0:
            print(f"  Spike {spike_idx}/{len(neuron_spikes)}")

        # Count how many OTHER neurons are active within +/- 10ms
        active_neurons = 0

        for other_neuron_id in range(N):
            if other_neuron_id == neuron_id:
                continue  # Skip self

            other_spikes = spike_times[other_neuron_id]

            # Check if any spike from other neuron is within the window
            time_diffs = np.abs(other_spikes - spike_time)
            if np.any(time_diffs <= window_half_width):
                active_neurons += 1

        network_activities.append(active_neurons)

    # Create histogram of network activity levels for this neuron
    neuron_activity_histograms.append(network_activities)

    print(f"  Neuron {neuron_id + 1}: Network activity range {min(network_activities)} to {max(network_activities)}")

# Save results to JSON
print("\nSaving results to JSON...")
results = {
    'neuron_activity_histograms': [list(map(int, hist)) for hist in neuron_activity_histograms],
    'description': 'For each neuron, list of network activity levels (number of other active neurons within +/- 10ms) at each of its spikes',
    'window_ms': 20.0,
    'n_neurons': N
}

with open('network_activity_by_spike.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Saved to network_activity_by_spike.json")

# Create output directory for histograms
output_dir = 'network_activity_histograms'
os.makedirs(output_dir, exist_ok=True)

# Create histograms for each neuron
print("\nCreating histogram visualizations...")

for neuron_id in range(N):
    network_activities = neuron_activity_histograms[neuron_id]

    if len(network_activities) == 0:
        continue

    plt.figure(figsize=(10, 6))

    # Create histogram
    max_activity = max(network_activities)
    bins = range(0, max_activity + 2)
    counts, _, _ = plt.hist(network_activities, bins=bins, edgecolor='black', alpha=0.7)

    plt.xlabel('Network Activity Level (# of other active neurons)', fontsize=12)
    plt.ylabel('Number of Spikes', fontsize=12)
    plt.title(f'Neuron {neuron_id + 1}: Spike Distribution by Network Activity\n'
              f'Total spikes: {len(network_activities)}, Mean activity: {np.mean(network_activities):.2f}',
              fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add text with statistics
    stats_text = f'Min: {min(network_activities)}\n'
    stats_text += f'Max: {max(network_activities)}\n'
    stats_text += f'Median: {np.median(network_activities):.1f}\n'
    stats_text += f'Std: {np.std(network_activities):.2f}'

    plt.text(0.98, 0.97, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/neuron_{neuron_id + 1}_network_activity.png', dpi=150)
    plt.close()

    print(f"Saved histogram for neuron {neuron_id + 1}")

print("\nDone! Histograms saved to", output_dir)

# Create summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

all_activities = []
for neuron_id in range(N):
    activities = neuron_activity_histograms[neuron_id]
    if len(activities) > 0:
        all_activities.extend(activities)
        mean_act = np.mean(activities)
        print(f"Neuron {neuron_id + 1:2d}: Mean network activity = {mean_act:6.2f}, "
              f"Range = [{min(activities):2d}, {max(activities):2d}]")

print("\nOverall statistics across all spikes:")
print(f"  Mean network activity: {np.mean(all_activities):.2f}")
print(f"  Median network activity: {np.median(all_activities):.1f}")
print(f"  Std network activity: {np.std(all_activities):.2f}")
print(f"  Min network activity: {min(all_activities)}")
print(f"  Max network activity: {max(all_activities)}")
