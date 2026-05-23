import json
from pathlib import Path

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


def compute_active_neurons_per_second(all_spike_times, start_time=None, end_time=None):
    """
    Compute how many neurons were active in each second.

    Args:
        all_spike_times: List of spike time arrays for all neurons
        start_time: Start of time range (if None, use min spike time)
        end_time: End of time range (if None, use max spike time)

    Returns:
        seconds: Array of second timestamps
        active_counts: Number of active neurons in each second
    """
    # Find the time range
    if start_time is None or end_time is None:
        min_time = float('inf')
        max_time = float('-inf')
        for spike_times in all_spike_times:
            if len(spike_times) > 0:
                min_time = min(min_time, spike_times[0])
                max_time = max(max_time, spike_times[-1])

        if start_time is None:
            start_time = min_time
        if end_time is None:
            end_time = max_time

    # Create bins for each second
    start_second = int(np.floor(start_time))
    end_second = int(np.ceil(end_time))
    num_seconds = end_second - start_second + 1

    # Count active neurons in each second
    active_counts = np.zeros(num_seconds, dtype=int)

    for spike_times in all_spike_times:
        if len(spike_times) == 0:
            continue

        # Filter spikes to the time range
        valid_spikes = spike_times[(spike_times >= start_time) & (spike_times <= end_time)]

        if len(valid_spikes) == 0:
            continue

        # Convert spike times to second bins
        spike_seconds = np.floor(valid_spikes).astype(int) - start_second

        # Mark which seconds this neuron was active in
        unique_seconds = np.unique(spike_seconds)

        # Ensure indices are within bounds
        valid_seconds = unique_seconds[(unique_seconds >= 0) & (unique_seconds < num_seconds)]

        # Increment count for each second this neuron was active
        active_counts[valid_seconds] += 1

    # Create array of second timestamps
    seconds = np.arange(start_second, start_second + num_seconds)

    return seconds, active_counts


def main():
    neuron_data_path = Path('neuron_data.json')

    print("Loading neuron data...")
    neuron_data = load_neuron_data(neuron_data_path)

    # Extract spike times
    all_spike_times = []
    for neuron_spikes in neuron_data:
        spike_times = extract_spike_times(neuron_spikes)
        all_spike_times.append(spike_times)

    num_neurons = len(all_spike_times)
    print(f"Found {num_neurons} neurons")

    # Find overall time range
    min_time = float('inf')
    max_time = float('-inf')
    for spike_times in all_spike_times:
        if len(spike_times) > 0:
            min_time = min(min_time, spike_times[0])
            max_time = max(max_time, spike_times[-1])

    mid_time = (min_time + max_time) / 2

    print(f"\nTime range: {min_time:.3f}s to {max_time:.3f}s")
    print(f"Duration: {(max_time - min_time) / 3600:.2f} hours")
    print(f"Midpoint: {mid_time:.3f}s")

    # Compute activity for first half
    print("\n" + "="*60)
    print("FIRST HALF ANALYSIS")
    print("="*60)
    print("Computing network activity for first half...")
    first_seconds, first_active = compute_active_neurons_per_second(
        all_spike_times, start_time=min_time, end_time=mid_time
    )

    first_mean = np.mean(first_active)
    first_median = np.median(first_active)
    first_std = np.std(first_active)
    first_min = np.min(first_active)
    first_max = np.max(first_active)

    print(f"Duration: {len(first_seconds):,} seconds ({len(first_seconds)/3600:.2f} hours)")
    print(f"Mean active neurons: {first_mean:.2f}")
    print(f"Median active neurons: {first_median:.0f}")
    print(f"Std dev: {first_std:.2f}")
    print(f"Min: {first_min}")
    print(f"Max: {first_max}")

    # Compute activity for second half
    print("\n" + "="*60)
    print("SECOND HALF ANALYSIS")
    print("="*60)
    print("Computing network activity for second half...")
    second_seconds, second_active = compute_active_neurons_per_second(
        all_spike_times, start_time=mid_time, end_time=max_time
    )

    second_mean = np.mean(second_active)
    second_median = np.median(second_active)
    second_std = np.std(second_active)
    second_min = np.min(second_active)
    second_max = np.max(second_active)

    print(f"Duration: {len(second_seconds):,} seconds ({len(second_seconds)/3600:.2f} hours)")
    print(f"Mean active neurons: {second_mean:.2f}")
    print(f"Median active neurons: {second_median:.0f}")
    print(f"Std dev: {second_std:.2f}")
    print(f"Min: {second_min}")
    print(f"Max: {second_max}")

    # Comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    mean_diff = second_mean - first_mean
    mean_pct_change = (mean_diff / first_mean) * 100

    median_diff = second_median - first_median

    print(f"Mean active neurons:")
    print(f"  First half:  {first_mean:.2f}")
    print(f"  Second half: {second_mean:.2f}")
    print(f"  Difference:  {mean_diff:+.2f} ({mean_pct_change:+.1f}%)")

    print(f"\nMedian active neurons:")
    print(f"  First half:  {first_median:.0f}")
    print(f"  Second half: {second_median:.0f}")
    print(f"  Difference:  {median_diff:+.0f}")

    if mean_diff > 0:
        print(f"\nThe network was MORE active in the second half.")
    elif mean_diff < 0:
        print(f"\nThe network was LESS active in the second half.")
    else:
        print(f"\nThe network activity was approximately EQUAL in both halves.")


if __name__ == "__main__":
    main()
