import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_neuron_data(json_path: Path):
    """Load neuron spike data from JSON file."""
    with json_path.open('r') as f:
        data = json.load(f)
    return data


def extract_spike_times(neuron_spikes):
    """Extract spike times from a neuron's spike list."""
    if not neuron_spikes:
        return np.array([], dtype=float)
    spike_times = np.array([float(spike['spikeTime']) for spike in neuron_spikes], dtype=float)
    spike_times.sort()
    return spike_times


def compute_active_neurons_per_second(all_spike_times):
    """
    Compute how many neurons were active in each second.

    Returns:
        seconds: Array of second timestamps
        active_counts: Number of active neurons in each second
    """
    # Find the time range
    min_time = float('inf')
    max_time = float('-inf')

    for spike_times in all_spike_times:
        if len(spike_times) > 0:
            min_time = min(min_time, spike_times[0])
            max_time = max(max_time, spike_times[-1])

    print(f"Time range: {min_time:.3f}s to {max_time:.3f}s")
    print(f"Duration: {(max_time - min_time) / 3600:.2f} hours")

    # Create bins for each second
    start_second = int(np.floor(min_time))
    end_second = int(np.ceil(max_time))
    num_seconds = end_second - start_second + 1

    print(f"Total seconds: {num_seconds:,}")

    # Count active neurons in each second
    active_counts = np.zeros(num_seconds, dtype=int)

    for neuron_idx, spike_times in enumerate(all_spike_times):
        if len(spike_times) == 0:
            continue

        # Convert spike times to second bins
        spike_seconds = np.floor(spike_times).astype(int) - start_second

        # Mark which seconds this neuron was active in
        unique_seconds = np.unique(spike_seconds)

        # Ensure indices are within bounds
        valid_seconds = unique_seconds[(unique_seconds >= 0) & (unique_seconds < num_seconds)]

        # Increment count for each second this neuron was active
        active_counts[valid_seconds] += 1

        if (neuron_idx + 1) % 5 == 0 or neuron_idx + 1 == len(all_spike_times):
            print(f"  Processed neuron {neuron_idx + 1}/{len(all_spike_times)}", end='\r')

    print()

    # Create array of second timestamps
    seconds = np.arange(start_second, start_second + num_seconds)

    return seconds, active_counts


def create_histogram(active_counts, output_path):
    """Create histogram of active neuron counts."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create histogram
    bins = np.arange(0, np.max(active_counts) + 2) - 0.5
    counts, edges, patches = ax.hist(active_counts, bins=bins, color='steelblue',
                                      alpha=0.8, edgecolor='black', linewidth=0.5)

    # Labels and title
    ax.set_xlabel('Number of Active Neurons', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Seconds', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Network Activity per Second', fontsize=14, fontweight='bold')

    # Set x-ticks to integers
    ax.set_xticks(np.arange(0, np.max(active_counts) + 1))

    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Add statistics text
    mean_active = np.mean(active_counts)
    median_active = np.median(active_counts)
    std_active = np.std(active_counts)
    max_active = np.max(active_counts)
    min_active = np.min(active_counts)

    stats_text = (
        f'Mean: {mean_active:.2f} neurons\n'
        f'Median: {median_active:.0f} neurons\n'
        f'Std: {std_active:.2f}\n'
        f'Min: {min_active} neurons\n'
        f'Max: {max_active} neurons'
    )

    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, family='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Histogram saved to: {output_path}")


def create_timeline_plot(seconds, active_counts, output_path):
    """Create timeline plot showing activity over time."""
    fig, ax = plt.subplots(figsize=(16, 6))

    # Convert seconds to hours for better readability
    time_hours = (seconds - seconds[0]) / 3600

    # Plot as line
    ax.plot(time_hours, active_counts, linewidth=0.5, color='steelblue', alpha=0.7)

    # Fill under the curve
    ax.fill_between(time_hours, active_counts, alpha=0.3, color='steelblue')

    # Labels and title
    ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Active Neurons per Second', fontsize=12, fontweight='bold')
    ax.set_title('Network Activity Timeline', fontsize=14, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set y-axis limits
    ax.set_ylim(0, np.max(active_counts) + 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Timeline plot saved to: {output_path}")


def main():
    neuron_data_path = Path('neuron_data.json')
    output_dir = Path('neuron_histograms')
    histogram_path = output_dir / 'network_activity_histogram.png'
    timeline_path = output_dir / 'network_activity_timeline.png'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading neuron data...")
    neuron_data = load_neuron_data(neuron_data_path)

    # Extract spike times
    all_spike_times = []
    for neuron_idx, neuron_spikes in enumerate(neuron_data):
        spike_times = extract_spike_times(neuron_spikes)
        all_spike_times.append(spike_times)

    num_neurons = len(all_spike_times)
    print(f"Found {num_neurons} neurons")

    # Compute active neurons per second
    print("\nComputing network activity per second...")
    seconds, active_counts = compute_active_neurons_per_second(all_spike_times)

    # Print statistics
    print("\n" + "="*60)
    print("STATISTICS:")
    print("="*60)
    print(f"Total seconds analyzed: {len(seconds):,}")
    print(f"Mean active neurons per second: {np.mean(active_counts):.2f}")
    print(f"Median active neurons per second: {np.median(active_counts):.0f}")
    print(f"Std dev: {np.std(active_counts):.2f}")
    print(f"Min active neurons: {np.min(active_counts)}")
    print(f"Max active neurons: {np.max(active_counts)}")

    # Count distribution
    print("\nDistribution:")
    for num_active in range(np.max(active_counts) + 1):
        count = np.sum(active_counts == num_active)
        percentage = (count / len(seconds)) * 100
        print(f"  {num_active:2d} neurons active: {count:6,} seconds ({percentage:5.2f}%)")

    # Create histogram
    print("\n" + "="*60)
    print("Creating histogram...")
    create_histogram(active_counts, histogram_path)

    # Create timeline plot
    print("\nCreating timeline plot...")
    create_timeline_plot(seconds, active_counts, timeline_path)

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
