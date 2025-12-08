import argparse
import json
from pathlib import Path
from typing import List, Dict

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


def compute_causality_vector(
    spikes_i: np.ndarray,
    spikes_j: np.ndarray,
    causality_window: float,
) -> np.ndarray:
    """
    Find all spikes of neuron i that occurred within causality_window after a spike of neuron j.

    Args:
        spikes_i: Spike times for neuron i (n spikes)
        spikes_j: Spike times for neuron j (m spikes)
        causality_window: Time window in seconds

    Returns:
        causality_spikes: Array of neuron i spike times that follow neuron j spikes
    """
    if len(spikes_i) == 0 or len(spikes_j) == 0:
        return np.array([], dtype=float)

    causality_spikes = []

    # For each spike in neuron i
    for spike_i_time in spikes_i:
        # Check if any spike in neuron j occurred within [spike_i_time - causality_window, spike_i_time)
        # This means: neuron j fired, then neuron i fired within the window

        # Find j spikes that are before spike_i_time and within the window
        time_differences = spike_i_time - spikes_j

        # Check if difference is in (0, causality_window]
        # Positive difference means j fired before i
        is_causal = (time_differences > 0) & (time_differences <= causality_window)

        # If at least one j spike caused this i spike, include it
        if np.any(is_causal):
            causality_spikes.append(spike_i_time)

    return np.array(causality_spikes, dtype=float)


def save_causality_vector(
    vector: np.ndarray,
    output_path: Path,
    neuron_i_id: int,
    neuron_j_id: int,
    spikes_i: np.ndarray,
    spikes_j: np.ndarray,
    metadata: Dict,
) -> None:
    """Save the causality vector to a JSON file."""
    output = {
        "neuron_i_id": neuron_i_id,
        "neuron_j_id": neuron_j_id,
        "total_spikes_i": len(spikes_i),
        "total_spikes_j": len(spikes_j),
        "causality_window_seconds": metadata["causality_window"],
        "causality_spike_count": len(vector),
        "causality_ratio": len(vector) / len(spikes_i) if len(spikes_i) > 0 else 0.0,
        "description": (
            f"Causality vector for neuron pair ({neuron_i_id}, {neuron_j_id}). "
            f"Contains spike times of neuron {neuron_i_id} that occurred within "
            f"{metadata['causality_window']*1000}ms after a spike from neuron {neuron_j_id}. "
            "This represents potential causal influence of j on i."
        ),
        "spikeTimes_i_j": vector.tolist()
    }

    with output_path.open('w') as f:
        json.dump(output, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute causality vectors for all neuron pairs. "
            "For each pair (i, j), creates a vector of neuron i spike times "
            "that occurred within the causality window after a neuron j spike."
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
    print(f"Output directory: {output_dir}")

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

    metadata = {
        "causality_window": causality_window,
    }

    # Compute causality vectors for all pairs
    print(f"\nComputing {num_neurons}x{num_neurons} = {num_neurons**2} causality vectors...")

    total_pairs = num_neurons * num_neurons
    pair_count = 0

    for i in range(num_neurons):
        for j in range(num_neurons):
            # Get spike times
            spikes_i = all_spike_times[i]
            spikes_j = all_spike_times[j]

            # Compute causality vector
            causality_vector = compute_causality_vector(spikes_i, spikes_j, causality_window)

            # Save to file
            output_filename = f"spikeTimes_{neuron_ids[i]}_{neuron_ids[j]}.json"
            output_path = output_dir / output_filename
            save_causality_vector(
                causality_vector,
                output_path,
                neuron_ids[i],
                neuron_ids[j],
                spikes_i,
                spikes_j,
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
    print(f"Vectors saved in: {output_dir}/")


if __name__ == "__main__":
    main()
