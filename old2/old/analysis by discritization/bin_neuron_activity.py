import argparse
import json
import math
from pathlib import Path
from typing import List

import numpy as np


def load_neuron_data(json_path: Path) -> List[List[dict]]:
    with json_path.open('r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected the neuron JSON to contain a list at the top level.")
    return data


def compute_time_bounds(neuron_data: List[List[dict]]) -> tuple[float, float]:
    try:
        min_time = min(spike['spikeTime'] for neuron in neuron_data for spike in neuron)
        max_time = max(spike['spikeTime'] for neuron in neuron_data for spike in neuron)
    except ValueError as exc:
        raise ValueError("No spike times found in the provided data.") from exc
    return float(min_time), float(max_time)


def spikes_to_numpy(neuron_spikes: List[dict]) -> np.ndarray:
    if not neuron_spikes:
        return np.array([], dtype=float)
    spike_times = np.array([float(spike['spikeTime']) for spike in neuron_spikes], dtype=float)
    spike_times.sort()
    return spike_times


def build_binary_vector(spike_times, start_time, num_bins, bin_size_s):
    vector = np.zeros(num_bins, dtype=np.uint8)
    if spike_times.size == 0:
        return vector

    relative_times = spike_times - start_time
    bin_indices = np.floor(relative_times / bin_size_s).astype(np.int64)
    valid = (bin_indices >= 0) & (bin_indices < num_bins)
    if not np.any(valid):
        return vector

    unique_bins = np.unique(bin_indices[valid])
    vector[unique_bins] = 1
    return vector


def stream_binary_array(vector: np.ndarray, destination) -> None:
    first = True
    for value in vector:
        if first:
            first = False
        else:
            destination.write(',')
        destination.write('1' if int(value) else '0')


def dump_binned_activity(neuron_data, output_path: Path, bin_size_ms: float) -> None:
    bin_size_s = bin_size_ms / 1000.0
    if bin_size_s <= 0:
        raise ValueError("Bin size must be positive.")

    start_time, end_time = compute_time_bounds(neuron_data)
    if end_time < start_time:
        raise ValueError("Invalid spike times encountered (end < start).")

    duration = end_time - start_time
    num_bins = int(math.floor(duration / bin_size_s)) + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing binary activity to '{output_path}'...")
    print(f"Bin size: {bin_size_ms} ms ({num_bins} bins spanning {duration:.3f} s).")

    with output_path.open('w') as out_file:
        out_file.write('{\n')
        out_file.write(f'  "bin_size_seconds": {json.dumps(bin_size_s)},\n')
        out_file.write(f'  "start_time": {json.dumps(start_time)},\n')
        out_file.write(f'  "end_time": {json.dumps(end_time)},\n')
        out_file.write(f'  "num_bins": {json.dumps(num_bins)},\n')
        out_file.write('  "neurons": [\n')

        for neuron_idx, neuron_spikes in enumerate(neuron_data):
            if neuron_idx > 0:
                out_file.write(',\n')

            spike_times = spikes_to_numpy(neuron_spikes)
            binary_vector = build_binary_vector(spike_times, start_time, num_bins, bin_size_s)
            neuron_id = neuron_spikes[0].get('unit') if neuron_spikes else neuron_idx + 1

            out_file.write('    {\n')
            out_file.write(f'      "neuron_id": {json.dumps(neuron_id)},\n')
            out_file.write('      "binary_activity": [')
            stream_binary_array(binary_vector, out_file)
            out_file.write(']\n')
            out_file.write('    }')

            print(f"  Processed neuron {neuron_idx + 1}/{len(neuron_data)}", end='\r')

        out_file.write('\n  ]\n}\n')

    print("\nDone. Remember that the resulting JSON can be extremely large at 1 ms resolution.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create time-binned binary activity vectors for each neuron."
    )
    parser.add_argument("--input", default="neuron_data.json")
    parser.add_argument("--output", default="binned_activity_1ms.json")
    parser.add_argument("--bin-size-ms", type=float, default=1.0)
    args = parser.parse_args()

    neuron_data = load_neuron_data(Path(args.input))
    dump_binned_activity(neuron_data, Path(args.output), args.bin_size_ms)


if __name__ == "__main__":
    main()
