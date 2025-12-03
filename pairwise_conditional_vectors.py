import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_binned_activity(path: Path) -> Dict:
    """Load the binned activity JSON file."""
    with path.open("r") as f:
        data = json.load(f)
    required_keys = {"bin_size_seconds", "start_time", "end_time", "num_bins", "neurons"}
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"Missing keys in binned activity file: {missing}")
    return data


def load_activity_matrix(data: Dict) -> tuple[np.ndarray, List[int], int, float]:
    """
    Convert the binned activity data into a matrix.

    Returns:
        activity_matrix: 2D array (num_neurons x num_bins) of binary activity
        neuron_ids: List of neuron IDs
        num_bins: Total number of 1ms bins
        base_bin_size: Size of each bin in seconds
    """
    num_bins = int(data["num_bins"])
    num_neurons = len(data["neurons"])
    base_bin_size = float(data["bin_size_seconds"])

    activity_matrix = np.zeros((num_neurons, num_bins), dtype=np.uint8)
    neuron_ids = []

    for neuron_idx, neuron_entry in enumerate(data["neurons"]):
        activity = np.array(neuron_entry["binary_activity"], dtype=np.uint8)
        if activity.size != num_bins:
            raise ValueError(
                f"Neuron {neuron_idx + 1} activity length {activity.size} "
                f"does not match num_bins={num_bins}."
            )
        activity_matrix[neuron_idx, :] = activity
        neuron_ids.append(neuron_entry.get("neuron_id", neuron_idx + 1))

    return activity_matrix, neuron_ids, num_bins, base_bin_size


def compute_pairwise_conditional_vectors(
    activity_matrix: np.ndarray,
    time_bin_size_s: float,
    base_bin_size: float,
) -> np.ndarray:
    """
    Compute conditional probability vectors for all neuron pairs.

    For each pair (i, j) and each time bin:
        P(i fires | j fires) = (# of 1ms bins where both fire) / (# of 1ms bins where j fires)

    Args:
        activity_matrix: 2D array (num_neurons x num_bins) of binary activity
        time_bin_size_s: Size of time bins in seconds (e.g., 1.0)
        base_bin_size: Size of base bins in seconds (e.g., 0.001)

    Returns:
        conditional_vectors: 3D array (num_neurons x num_neurons x num_time_bins)
    """
    num_neurons, num_base_bins = activity_matrix.shape

    # Check that time_bin_size_s is a multiple of base_bin_size
    ratio = time_bin_size_s / base_bin_size
    bins_per_time_bin = int(round(ratio))
    if not math.isclose(ratio, bins_per_time_bin, rel_tol=1e-9):
        raise ValueError(
            f"Time bin size must be an integer multiple of the source bin size. "
            f"Got {time_bin_size_s} s vs {base_bin_size} s."
        )

    num_time_bins = (num_base_bins + bins_per_time_bin - 1) // bins_per_time_bin

    # Pad activity matrix if necessary
    remainder = num_base_bins % bins_per_time_bin
    if remainder:
        pad_width = bins_per_time_bin - remainder
        activity_matrix = np.pad(activity_matrix, ((0, 0), (0, pad_width)), constant_values=0)

    # Reshape to (num_neurons, num_time_bins, bins_per_time_bin)
    reshaped = activity_matrix.reshape(num_neurons, num_time_bins, bins_per_time_bin)

    # Initialize output: (num_neurons x num_neurons x num_time_bins)
    conditional_vectors = np.full((num_neurons, num_neurons, num_time_bins), np.nan, dtype=np.float32)

    print(f"Computing {num_neurons}x{num_neurons} conditional vectors...")
    print(f"Each vector has {num_time_bins} time bins of {time_bin_size_s}s")

    # For each pair (i, j)
    for i in range(num_neurons):
        for j in range(num_neurons):
            # For each time bin, count coincident spikes and j's spikes
            for t in range(num_time_bins):
                neuron_i_activity = reshaped[i, t, :]
                neuron_j_activity = reshaped[j, t, :]

                # Count 1ms bins where both fired
                coincident_count = np.sum(neuron_i_activity & neuron_j_activity)

                # Count 1ms bins where j fired
                j_count = np.sum(neuron_j_activity)

                # Compute conditional probability
                if j_count > 0:
                    conditional_vectors[i, j, t] = coincident_count / j_count
                # else: remains NaN (neuron j didn't fire)

        print(f"  Processed neuron {i + 1}/{num_neurons}", end='\r')

    print()
    return conditional_vectors


def write_output_json(
    path: Path,
    conditional_vectors: np.ndarray,
    neuron_ids: List[int],
    metadata: Dict,
) -> None:
    """Write the pairwise conditional vectors to JSON."""
    num_neurons = len(neuron_ids)

    # Convert to nested list structure, handling NaN values
    vectors_list = []
    for i in range(num_neurons):
        row = []
        for j in range(num_neurons):
            vector = conditional_vectors[i, j, :].tolist()
            # Convert NaN to None for JSON compatibility
            vector_cleaned = [None if (isinstance(v, float) and math.isnan(v)) else float(v) for v in vector]
            row.append(vector_cleaned)
        vectors_list.append(row)

    output = {
        "source_bin_size_seconds": metadata["base_bin_size"],
        "time_bin_size_seconds": metadata["time_bin_size"],
        "start_time": metadata["start_time"],
        "end_time": metadata["end_time"],
        "num_time_bins": metadata["num_time_bins"],
        "num_neurons": num_neurons,
        "neuron_ids": neuron_ids,
        "description": (
            "Pairwise conditional probability vectors. "
            "conditional_vectors[i][j][t] = P(neuron i fires | neuron j fires) in time bin t. "
            "null values indicate neuron j did not fire in that time bin."
        ),
        "conditional_vectors": vectors_list
    }

    print(f"Writing output to: {path}")
    with path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"Output written successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute pairwise conditional probability vectors for all neuron pairs. "
            "For each pair (i,j) and each time bin: P(i fires | j fires)."
        )
    )
    parser.add_argument(
        "--input",
        default="binned_activity_1ms.json",
        help="Path to the binned activity JSON file (default: binned_activity_1ms.json)"
    )
    parser.add_argument(
        "--output",
        default="pairwise_conditional_vectors.json",
        help="Output JSON file path (default: pairwise_conditional_vectors.json)"
    )
    parser.add_argument(
        "--time-bin-size",
        type=float,
        default=1.0,
        help="Time bin size in seconds (default: 1.0). Must be a multiple of source bin size."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading binned activity from: {input_path}")
    data = load_binned_activity(input_path)

    print("Converting to activity matrix...")
    activity_matrix, neuron_ids, num_bins, base_bin_size = load_activity_matrix(data)

    print(f"Activity matrix shape: {activity_matrix.shape}")
    print(f"Computing conditional vectors with {args.time_bin_size}s time bins...")

    conditional_vectors = compute_pairwise_conditional_vectors(
        activity_matrix,
        args.time_bin_size,
        base_bin_size
    )

    metadata = {
        "base_bin_size": base_bin_size,
        "time_bin_size": args.time_bin_size,
        "start_time": data["start_time"],
        "end_time": data["end_time"],
        "num_time_bins": conditional_vectors.shape[2],
    }

    write_output_json(output_path, conditional_vectors, neuron_ids, metadata)

    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Generated {len(neuron_ids)}x{len(neuron_ids)} = {len(neuron_ids)**2} conditional vectors")
    print(f"Each vector length: {conditional_vectors.shape[2]} time bins")
    print(f"Time bin size: {args.time_bin_size} seconds")

    # Count non-NaN values
    non_nan_count = np.sum(~np.isnan(conditional_vectors))
    total_values = conditional_vectors.size
    print(f"Non-NaN values: {non_nan_count}/{total_values} ({100*non_nan_count/total_values:.1f}%)")


if __name__ == "__main__":
    main()
