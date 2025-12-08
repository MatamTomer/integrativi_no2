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


def calculate_network_activity(
    data: Dict,
    time_bin_size_s: float,
) -> tuple[List[int], Dict]:
    """
    Calculate how many neurons were active in each time bin.

    Args:
        data: The loaded binned activity data
        time_bin_size_s: Size of time bins in seconds (must be multiple of base bin size)

    Returns:
        Tuple of (activity_counts, metadata)
    """
    base_bin_size = float(data["bin_size_seconds"])
    if base_bin_size <= 0:
        raise ValueError("Base bin size from input JSON must be positive.")
    if time_bin_size_s <= 0:
        raise ValueError("Time bin size must be positive.")

    # Check that time_bin_size_s is a multiple of base_bin_size
    ratio = time_bin_size_s / base_bin_size
    bins_per_group = int(round(ratio))
    if not math.isclose(ratio, bins_per_group, rel_tol=1e-9):
        raise ValueError(
            f"Time bin size must be an integer multiple of the source bin size. "
            f"Got {time_bin_size_s} s vs {base_bin_size} s."
        )

    num_base_bins = int(data["num_bins"])
    num_time_bins = (num_base_bins + bins_per_group - 1) // bins_per_group

    # Initialize activity count for each time bin
    activity_counts = np.zeros(num_time_bins, dtype=np.uint32)

    # Process each neuron
    for neuron_idx, neuron_entry in enumerate(data["neurons"]):
        activity = np.array(neuron_entry["binary_activity"], dtype=np.uint8)

        if activity.size != num_base_bins:
            raise ValueError(
                f"Neuron {neuron_idx + 1} activity length {activity.size} "
                f"does not match num_bins={num_base_bins}."
            )

        # Pad if necessary
        remainder = activity.size % bins_per_group
        if remainder:
            pad_width = bins_per_group - remainder
            activity = np.pad(activity, (0, pad_width), constant_values=0)

        # Reshape into time bins and check if neuron was active (any spike) in each bin
        reshaped = activity.reshape(num_time_bins, bins_per_group)
        neuron_active_per_bin = reshaped.any(axis=1).astype(np.uint8)

        # Add to activity count
        activity_counts += neuron_active_per_bin

        print(f"  Processed neuron {neuron_idx + 1}/{len(data['neurons'])}", end='\r')

    print()  # New line after progress

    metadata = {
        "source_bin_size_seconds": base_bin_size,
        "time_bin_size_seconds": time_bin_size_s,
        "start_time": data["start_time"],
        "end_time": data["end_time"],
        "num_time_bins": num_time_bins,
        "total_neurons": len(data["neurons"])
    }

    return activity_counts.tolist(), metadata


def write_output_json(
    path: Path,
    activity_counts: List[int],
    metadata: Dict
) -> None:
    """Write the network activity results to JSON."""
    output = {
        "source_bin_size_seconds": metadata["source_bin_size_seconds"],
        "time_bin_size_seconds": metadata["time_bin_size_seconds"],
        "start_time": metadata["start_time"],
        "end_time": metadata["end_time"],
        "num_time_bins": metadata["num_time_bins"],
        "total_neurons": metadata["total_neurons"],
        "description": (
            f"Count of neurons that fired at least once in each "
            f"{metadata['time_bin_size_seconds']} second time bin."
        ),
        "active_neuron_counts": activity_counts
    }

    with path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"\nOutput written to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate network activity: count of active neurons per time bin."
    )
    parser.add_argument(
        "--input",
        default="binned_activity_1ms.json",
        help="Path to the binned activity JSON file (default: binned_activity_1ms.json)"
    )
    parser.add_argument(
        "--output",
        default="network_activity.json",
        help="Output JSON file path (default: network_activity.json)"
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

    print(f"Calculating network activity with {args.time_bin_size}s time bins...")
    activity_counts, metadata = calculate_network_activity(data, args.time_bin_size)

    write_output_json(output_path, activity_counts, metadata)

    # Print summary statistics
    total_bins = metadata["num_time_bins"]
    max_active = max(activity_counts)
    min_active = min(activity_counts)
    avg_active = sum(activity_counts) / len(activity_counts)

    print("\n=== Summary ===")
    print(f"Time bin size: {metadata['time_bin_size_seconds']} seconds")
    print(f"Total time bins: {total_bins}")
    print(f"Total neurons: {metadata['total_neurons']}")
    print(f"Active neurons per bin - Min: {min_active}, Max: {max_active}, Avg: {avg_active:.2f}")


if __name__ == "__main__":
    main()
