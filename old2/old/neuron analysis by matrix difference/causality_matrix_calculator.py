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


def compute_difference_matrix(
    spikes_i: np.ndarray,
    spikes_j: np.ndarray,
) -> np.ndarray:
    """
    Compute difference matrix: matrix[k, l] = spikes_i[k] - spikes_j[l]

    Args:
        spikes_i: Spike times for neuron i (n spikes)
        spikes_j: Spike times for neuron j (m spikes)

    Returns:
        difference_matrix: n x m matrix of time differences
    """
    n = len(spikes_i)
    m = len(spikes_j)

    if n == 0 or m == 0:
        return np.array([]).reshape(n, m)

    # Create meshgrid for vectorized computation
    # spikes_i[:, np.newaxis] creates column vector (n x 1)
    # spikes_j creates row vector (1 x m)
    # Broadcasting creates (n x m) matrix
    difference_matrix = spikes_i[:, np.newaxis] - spikes_j

    return difference_matrix


def apply_causality_threshold(
    difference_matrix: np.ndarray,
    causality_window: float,
) -> np.ndarray:
    """
    Convert difference matrix to binary causality matrix.

    Entry is 1 if difference is in [0, causality_window], else 0.
    This captures cases where neuron j fired, then neuron i fired within the window.

    Args:
        difference_matrix: n x m matrix of time differences
        causality_window: Time window in seconds

    Returns:
        causality_matrix: n x m binary matrix
    """
    causality_matrix = np.zeros_like(difference_matrix, dtype=np.uint8)

    # Check if difference is in [0, causality_window]
    # Positive difference means i fired after j (potential causality)
    mask = (difference_matrix >= 0) & (difference_matrix <= causality_window)
    causality_matrix[mask] = 1

    return causality_matrix


def save_matrix(
    matrix: np.ndarray,
    output_path: Path,
    neuron_i_id: int,
    neuron_j_id: int,
    spikes_i: np.ndarray,
    spikes_j: np.ndarray,
    metadata: Dict,
) -> None:
    """Save the causality matrix to a JSON file."""
    output = {
        "neuron_i_id": neuron_i_id,
        "neuron_j_id": neuron_j_id,
        "num_spikes_i": len(spikes_i),
        "num_spikes_j": len(spikes_j),
        "causality_window_seconds": metadata["causality_window"],
        "description": (
            f"Binary causality matrix for neuron pair ({neuron_i_id}, {neuron_j_id}). "
            f"Entry [k, l] is 1 if neuron i spike k occurred within "
            f"{metadata['causality_window']*1000}ms after neuron j spike l, else 0. "
            "This captures potential causal influence of j on i."
        ),
        "matrix_shape": list(matrix.shape),
        "causality_count": int(np.sum(matrix)),
        "causality_matrix": matrix.tolist()
    }

    with output_path.open('w') as f:
        json.dump(output, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute causality difference matrices for all neuron pairs. "
            "For each pair (i, j), creates a matrix where entry [k, l] indicates "
            "whether neuron i's k-th spike occurred within the causality window "
            "after neuron j's l-th spike."
        )
    )
    parser.add_argument(
        "--input",
        default="../neuron_data.json",
        help="Path to neuron data JSON file (default: ../neuron_data.json)"
    )
    parser.add_argument(
        "--output-dir",
        default="causality_matrices",
        help="Directory to save causality matrices (default: causality_matrices)"
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

    # Compute causality matrices for all pairs
    print(f"\nComputing {num_neurons}x{num_neurons} = {num_neurons**2} causality matrices...")

    total_pairs = num_neurons * num_neurons
    pair_count = 0

    for i in range(num_neurons):
        for j in range(num_neurons):
            # Get spike times
            spikes_i = all_spike_times[i]
            spikes_j = all_spike_times[j]

            # Compute difference matrix
            diff_matrix = compute_difference_matrix(spikes_i, spikes_j)

            # Apply causality threshold
            causality_matrix = apply_causality_threshold(diff_matrix, causality_window)

            # Save to file
            output_filename = f"neurons_{neuron_ids[i]}_{neuron_ids[j]}.json"
            output_path = output_dir / output_filename
            save_matrix(
                causality_matrix,
                output_path,
                neuron_ids[i],
                neuron_ids[j],
                spikes_i,
                spikes_j,
                metadata
            )

            pair_count += 1
            print(f"  Processed pair {pair_count}/{total_pairs} ({neuron_ids[i]}, {neuron_ids[j]})", end='\r')

    print(f"\n\nDone! All {total_pairs} causality matrices saved to: {output_dir}/")

    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Total neuron pairs: {total_pairs}")
    print(f"Causality window: {causality_window*1000} ms")
    print(f"Matrices saved in: {output_dir}/")


if __name__ == "__main__":
    main()
