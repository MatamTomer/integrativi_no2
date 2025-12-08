import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_network_activity(path: Path) -> Dict:
    """Load the network activity JSON file."""
    with path.open("r") as f:
        data = json.load(f)
    required_keys = {"active_neuron_counts", "num_time_bins"}
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"Missing keys in network activity file: {missing}")
    return data


def load_pairwise_vectors(path: Path) -> Dict:
    """Load the pairwise conditional vectors JSON file."""
    with path.open("r") as f:
        data = json.load(f)
    required_keys = {"conditional_vectors", "num_neurons", "neuron_ids"}
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"Missing keys in pairwise vectors file: {missing}")
    return data


def compute_activity_conditioned_vectors(
    network_activity: List[int],
    conditional_vectors: List[List[List]],
    num_neurons: int,
) -> np.ndarray:
    """
    Compute average pairwise conditional probabilities grouped by network activity level.

    For each pair (i, j), creates a vector of length (num_neurons + 1) where entry k is:
    - The average of P(i fires | j fires) across all seconds where k neurons were active
    - NaN if no such seconds exist or all values were NaN

    Args:
        network_activity: List of active neuron counts per time bin
        conditional_vectors: 3D structure [i][j][t] of conditional probabilities
        num_neurons: Total number of neurons (29)

    Returns:
        activity_conditioned: 3D array (num_neurons x num_neurons x (num_neurons+1))
    """
    num_time_bins = len(network_activity)
    max_activity = num_neurons  # Activity levels: 0 to num_neurons

    # Initialize output: (num_neurons x num_neurons x (num_neurons + 1))
    activity_conditioned = np.full(
        (num_neurons, num_neurons, max_activity + 1),
        np.nan,
        dtype=np.float64
    )

    print(f"Computing activity-conditioned vectors for {num_neurons}x{num_neurons} pairs...")
    print(f"Network activity levels: 0 to {max_activity}")
    print(f"Time bins: {num_time_bins}")

    # For each pair (i, j)
    for i in range(num_neurons):
        for j in range(num_neurons):
            # Get the conditional vector for this pair
            pair_vector = conditional_vectors[i][j]

            if len(pair_vector) != num_time_bins:
                raise ValueError(
                    f"Pair ({i},{j}) vector length {len(pair_vector)} "
                    f"does not match num_time_bins={num_time_bins}"
                )

            # Group values by network activity level
            # activity_groups[k] will contain all non-NaN values for activity level k
            activity_groups = [[] for _ in range(max_activity + 1)]

            for t in range(num_time_bins):
                activity_level = network_activity[t]
                value = pair_vector[t]

                # Skip NaN values
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    if 0 <= activity_level <= max_activity:
                        activity_groups[activity_level].append(value)

            # Compute average for each activity level
            for k in range(max_activity + 1):
                if len(activity_groups[k]) > 0:
                    activity_conditioned[i, j, k] = np.mean(activity_groups[k])
                # else: remains NaN

        print(f"  Processed neuron {i + 1}/{num_neurons}", end='\r')

    print()
    return activity_conditioned


def write_output_json(
    path: Path,
    activity_conditioned: np.ndarray,
    neuron_ids: List[int],
    metadata: Dict,
) -> None:
    """Write the activity-conditioned vectors to JSON."""
    num_neurons = len(neuron_ids)
    max_activity = num_neurons

    # Convert to nested list structure, handling NaN values
    vectors_list = []
    for i in range(num_neurons):
        row = []
        for j in range(num_neurons):
            vector = activity_conditioned[i, j, :].tolist()
            # Convert NaN to None for JSON compatibility
            vector_cleaned = [
                None if (isinstance(v, float) and math.isnan(v)) else float(v)
                for v in vector
            ]
            row.append(vector_cleaned)
        vectors_list.append(row)

    output = {
        "time_bin_size_seconds": metadata["time_bin_size"],
        "num_neurons": num_neurons,
        "neuron_ids": neuron_ids,
        "max_network_activity": max_activity,
        "description": (
            "Activity-conditioned pairwise conditional probabilities. "
            "activity_vectors[i][j][k] = average P(neuron i fires | neuron j fires) "
            "across all time bins where k neurons were active in the network. "
            "null values indicate either no time bins with that activity level "
            "or all values were undefined (neuron j never fired when activity was k)."
        ),
        "activity_conditioned_vectors": vectors_list
    }

    print(f"Writing output to: {path}")
    with path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"Output written successfully!")


def compute_statistics(
    activity_conditioned: np.ndarray,
    network_activity: List[int],
) -> None:
    """Print summary statistics about the results."""
    num_neurons = activity_conditioned.shape[0]
    max_activity = activity_conditioned.shape[2] - 1

    print("\n=== Summary Statistics ===")
    print(f"Generated {num_neurons}x{num_neurons} activity-conditioned vectors")
    print(f"Each vector length: {max_activity + 1} (activity levels 0 to {max_activity})")

    # Count non-NaN values
    non_nan_count = np.sum(~np.isnan(activity_conditioned))
    total_values = activity_conditioned.size
    print(f"Non-NaN values: {non_nan_count}/{total_values} ({100*non_nan_count/total_values:.1f}%)")

    # Distribution of network activity
    activity_counts = np.bincount(network_activity, minlength=max_activity + 1)
    print("\n=== Network Activity Distribution ===")
    print("Activity Level | Count | % of Time")
    print("-" * 40)
    for k in range(max_activity + 1):
        count = activity_counts[k]
        percentage = 100 * count / len(network_activity)
        if count > 0:
            print(f"{k:14d} | {count:5d} | {percentage:6.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute activity-conditioned pairwise conditional probabilities. "
            "For each pair (i,j), creates a vector where entry k is the average "
            "P(i fires | j fires) when network activity was k neurons."
        )
    )
    parser.add_argument(
        "--network-activity",
        default="network_activity.json",
        help="Path to network activity JSON file (default: network_activity.json)"
    )
    parser.add_argument(
        "--pairwise-vectors",
        default="pairwise_conditional_vectors.json",
        help="Path to pairwise conditional vectors JSON file (default: pairwise_conditional_vectors.json)"
    )
    parser.add_argument(
        "--output",
        default="pairwise_activity_by_network_state.json",
        help="Output JSON file path (default: pairwise_activity_by_network_state.json)"
    )
    args = parser.parse_args()

    network_path = Path(args.network_activity)
    pairwise_path = Path(args.pairwise_vectors)
    output_path = Path(args.output)

    print(f"Loading network activity from: {network_path}")
    network_data = load_network_activity(network_path)
    network_activity = network_data["active_neuron_counts"]

    print(f"Loading pairwise conditional vectors from: {pairwise_path}")
    pairwise_data = load_pairwise_vectors(pairwise_path)
    conditional_vectors = pairwise_data["conditional_vectors"]
    neuron_ids = pairwise_data["neuron_ids"]
    num_neurons = pairwise_data["num_neurons"]

    # Verify dimensions match
    if len(network_activity) != pairwise_data.get("num_time_bins"):
        raise ValueError(
            f"Mismatch in time bins: network_activity has {len(network_activity)}, "
            f"pairwise_vectors has {pairwise_data.get('num_time_bins')}"
        )

    print(f"Computing activity-conditioned vectors...")
    activity_conditioned = compute_activity_conditioned_vectors(
        network_activity,
        conditional_vectors,
        num_neurons,
    )

    metadata = {
        "time_bin_size": pairwise_data.get("time_bin_size_seconds", 1.0),
    }

    write_output_json(output_path, activity_conditioned, neuron_ids, metadata)
    compute_statistics(activity_conditioned, network_activity)


if __name__ == "__main__":
    main()
