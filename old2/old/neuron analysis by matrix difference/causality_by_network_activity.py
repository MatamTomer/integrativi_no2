import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np


# Causality window in seconds (20ms)
CAUSALITY_WINDOW = 20e-3


def load_neuron_data(json_path: Path) -> List[List[dict]]:
    """Load neuron spike data from JSON file."""
    with json_path.open('r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected the neuron JSON to contain a list at the top level.")
    return data


def extract_spike_times(neuron_spikes: List[dict]) -> np.ndarray:
    """Extract spike times from a neuron's spike list."""
    if not neuron_spikes:
        return np.array([], dtype=float)
    spike_times = np.array([float(spike['spikeTime']) for spike in neuron_spikes], dtype=float)
    spike_times.sort()
    return spike_times


def compute_network_activity_for_all_spikes(
    all_spike_times: List[np.ndarray],
    causality_window: float,
) -> List[List[Tuple[float, int]]]:
    """
    For each neuron and each of its spikes, compute network activity in the causality window.

    Args:
        all_spike_times: List of spike time arrays for all neurons
        causality_window: Time window for counting active neurons

    Returns:
        List of lists of tuples: neuron_spike_with_activity[neuron_idx] = [(spike_time, num_active_neurons), ...]
    """
    num_neurons = len(all_spike_times)
    neuron_spikes_with_activity = []

    for neuron_idx in range(num_neurons):
        spikes = all_spike_times[neuron_idx]
        spike_activity_list = []

        for spike_time in spikes:
            # Count how many neurons were active in [spike_time - window, spike_time + window]
            lower_bound = spike_time - causality_window
            upper_bound = spike_time + causality_window

            active_count = 0
            for other_neuron_spikes in all_spike_times:
                # Check if this neuron has any spikes in the window
                in_window = (other_neuron_spikes >= lower_bound) & (other_neuron_spikes <= upper_bound)
                if np.any(in_window):
                    active_count += 1

            spike_activity_list.append((float(spike_time), int(active_count)))

        neuron_spikes_with_activity.append(spike_activity_list)

        if (neuron_idx + 1) % 5 == 0 or neuron_idx + 1 == num_neurons:
            print(f"  Computed network activity for neuron {neuron_idx + 1}/{num_neurons}", end='\r')

    print()
    return neuron_spikes_with_activity


def compute_causality_probability_by_activity(
    spikes_i_with_activity: List[Tuple[float, int]],
    spikes_j_with_activity: List[Tuple[float, int]],
    causality_window: float,
    max_activity: int = 29,
) -> np.ndarray:
    """
    Compute P(i fires | j fires) for each network activity level.

    Args:
        spikes_i_with_activity: List of (spike_time, activity) for neuron i
        spikes_j_with_activity: List of (spike_time, activity) for neuron j
        causality_window: Time window for causality (20ms)
        max_activity: Maximum network activity level

    Returns:
        Vector of length (max_activity + 1) with probabilities for each activity level
    """
    # Initialize counters for each activity level
    j_spikes_at_activity = {a: 0 for a in range(max_activity + 1)}
    j_spikes_with_i_following = {a: 0 for a in range(max_activity + 1)}

    # Convert i spikes to array for faster searching
    i_spike_times = np.array([st for st, _ in spikes_i_with_activity])

    # For each j spike
    for j_spike_time, j_activity in spikes_j_with_activity:
        if 0 <= j_activity <= max_activity:
            # Count this j spike at this activity level
            j_spikes_at_activity[j_activity] += 1

            # Check if neuron i spiked within causality_window AFTER this j spike
            # Looking for i spikes in (j_spike_time, j_spike_time + causality_window]
            i_following = (i_spike_times > j_spike_time) & (i_spike_times <= j_spike_time + causality_window)

            if np.any(i_following):
                # Neuron i fired after j within the causality window
                j_spikes_with_i_following[j_activity] += 1

    # Compute probabilities
    probabilities = np.full(max_activity + 1, np.nan, dtype=float)

    for activity in range(max_activity + 1):
        if j_spikes_at_activity[activity] > 0:
            probabilities[activity] = j_spikes_with_i_following[activity] / j_spikes_at_activity[activity]

    return probabilities


def save_causality_vector(
    vector: np.ndarray,
    output_path: Path,
    neuron_i_id: int,
    neuron_j_id: int,
    metadata: Dict,
) -> None:
    """Save causality probability vector to JSON."""
    # Convert NaN to None for JSON
    vector_list = [None if np.isnan(v) else float(v) for v in vector]

    output = {
        "neuron_i_id": neuron_i_id,
        "neuron_j_id": neuron_j_id,
        "causality_window_seconds": metadata["causality_window"],
        "max_network_activity": len(vector) - 1,
        "description": (
            f"Causality probability vector for pair ({neuron_i_id}, {neuron_j_id}). "
            f"causality_vector[A] = P(neuron {neuron_i_id} fires | neuron {neuron_j_id} fires) "
            f"when network activity is A neurons. "
            f"Computed as: (# of j spikes at activity A followed by i spike within {metadata['causality_window']*1000}ms) / "
            f"(# of j spikes at activity A). "
            "null values indicate no j spikes at that activity level."
        ),
        "causality_vector": vector_list
    }

    with output_path.open('w') as f:
        json.dump(output, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute causality probabilities conditioned on network activity. "
            "For each pair (i,j), computes P(i fires | j fires) for each activity level."
        )
    )
    parser.add_argument(
        "--input",
        default="../neuron_data.json",
        help="Path to neuron data JSON file (default: ../neuron_data.json)"
    )
    parser.add_argument(
        "--output-dir",
        default="causality_vectors",
        help="Directory to save causality vectors (default: causality_vectors)"
    )
    parser.add_argument(
        "--causality-window",
        type=float,
        default=20e-3,
        help="Causality window in seconds (default: 0.02 = 20ms)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    causality_window = args.causality_window

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading neuron data from: {input_path}")
    neuron_data = load_neuron_data(input_path)
    num_neurons = len(neuron_data)

    print(f"Found {num_neurons} neurons")
    print(f"Causality window: {causality_window*1000} ms")

    # Extract spike times for all neurons
    print("\nExtracting spike times...")
    all_spike_times = []
    neuron_ids = []
    for neuron_idx, neuron_spikes in enumerate(neuron_data):
        spike_times = extract_spike_times(neuron_spikes)
        all_spike_times.append(spike_times)
        neuron_id = neuron_spikes[0].get('unit') if neuron_spikes else neuron_idx + 1
        neuron_ids.append(neuron_id)
        print(f"  Neuron {neuron_id}: {len(spike_times)} spikes")

    # Compute network activity for all spikes
    print("\nComputing network activity for all spikes...")
    neuron_spikes_with_activity = compute_network_activity_for_all_spikes(
        all_spike_times,
        causality_window
    )

    metadata = {
        "causality_window": causality_window,
    }

    # Compute causality vectors for all pairs
    print(f"\nComputing {num_neurons}x{num_neurons} = {num_neurons**2} causality vectors...")

    total_pairs = num_neurons * num_neurons
    pair_count = 0

    for i in range(num_neurons):
        for j in range(num_neurons):
            # Get spikes with activity for both neurons
            spikes_i_with_activity = neuron_spikes_with_activity[i]
            spikes_j_with_activity = neuron_spikes_with_activity[j]

            # Compute causality probability by activity
            causality_vector = compute_causality_probability_by_activity(
                spikes_i_with_activity,
                spikes_j_with_activity,
                causality_window
            )

            # Save to file
            output_filename = f"causality_{neuron_ids[i]}_{neuron_ids[j]}.json"
            output_path = output_dir / output_filename
            save_causality_vector(
                causality_vector,
                output_path,
                neuron_ids[i],
                neuron_ids[j],
                metadata
            )

            pair_count += 1
            if pair_count % 29 == 0 or pair_count == total_pairs:
                print(f"  Processed pair {pair_count}/{total_pairs}", end='\r')

    print(f"\n\nDone! All {total_pairs} causality vectors saved to: {output_dir}/")

    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Total neuron pairs: {total_pairs}")
    print(f"Causality window: {causality_window*1000} ms")
    print(f"Vector length: {num_neurons + 1} (activity levels 0 to {num_neurons})")
    print(f"Vectors saved in: {output_dir}/")


if __name__ == "__main__":
    main()
