import json
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_neuron_spikes(json_file_path):
    """
    Analyzes neuron spike data from a JSON file to find correlations and plots histograms.

    For each neuron, it calculates how many spikes from other neurons occurred within a
    small time window (0.001s) of its own spikes. It then generates a histogram for each
    neuron showing this spike correlation data.
    """
    try:
        with open(json_file_path, 'r') as f:
            all_neurons_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at '{json_file_path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file_path}'")
        return

    num_neurons = len(all_neurons_data)
    if num_neurons == 0:
        print("No neuron data found in the JSON file.")
        return

    # Extract spike times for each neuron, convert to sorted numpy arrays for fast search
    neuron_spike_times = [
        np.sort(np.array([spike['spikeTime'] for spike in neuron], dtype=float))
        for neuron in all_neurons_data
    ]

    time_window = 0.001

    def count_correlated_spikes(source_times, target_times):
        """Return number of target spikes within +/- time_window of each source spike."""
        if source_times.size == 0 or target_times.size == 0:
            return 0
        lower_bounds = np.searchsorted(target_times, source_times - time_window, side='left')
        upper_bounds = np.searchsorted(target_times, source_times + time_window, side='right')
        return int(np.sum(upper_bounds - lower_bounds))

    # --- Analysis ---
    # For each neuron i, create a vector of correlations with all other neurons j
    all_neurons_correlation_vectors = []

    for i in range(num_neurons):
        correlation_vector = np.zeros(num_neurons, dtype=int)
        source_times = neuron_spike_times[i]

        for j in range(num_neurons):
            correlation_vector[j] = count_correlated_spikes(source_times, neuron_spike_times[j])

        all_neurons_correlation_vectors.append(correlation_vector)
        print(f"Neuron {i+1} correlation vector: {correlation_vector}")

    # Ensure destination folder exists before saving any outputs
    histogram_dir = "neuron_histograms"
    os.makedirs(histogram_dir, exist_ok=True)

    # Persist correlation vectors for later inspection
    correlation_vectors_path = os.path.join(histogram_dir, "correlation_vectors.json")
    with open(correlation_vectors_path, 'w') as corr_file:
        json.dump([vec.tolist() for vec in all_neurons_correlation_vectors], corr_file, indent=2)
    print(f"Saved correlation vectors to '{correlation_vectors_path}'")

    # --- Plotting Histograms ---
    for i, correlation_vector in enumerate(all_neurons_correlation_vectors):
        plt.figure(figsize=(10, 6))
        
        # Neuron indices for the x-axis (e.g., 1 to 29)
        neuron_indices = np.arange(1, num_neurons + 1)
        
        plt.bar(neuron_indices, correlation_vector, color='skyblue')
        
        plt.xlabel("Neuron Index")
        plt.ylabel("Number of Correlated Spikes (within 0.001s window)")
        plt.title(f"Spike Correlation Histogram for Neuron {i+1}")
        plt.xticks(neuron_indices)
        plt.grid(axis='y', linestyle='--')
        
        # Save the plot to a file inside the histogram directory
        plot_filename = os.path.join(histogram_dir, f"neuron_{i+1}_histogram.png")
        plt.savefig(plot_filename)
        print(f"Saved histogram for neuron {i+1} to '{plot_filename}'")
        
        plt.close() # Close the figure to free up memory

if __name__ == '__main__':
    # Assuming 'neuron_data.json' is in the same directory
    analyze_neuron_spikes('neuron_data.json')
