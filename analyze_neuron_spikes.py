
import json
import numpy as np
import matplotlib.pyplot as plt

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

    # Extract spike times for each neuron for faster access
    neuron_spike_times = [[spike['spikeTime'] for spike in neuron] for neuron in all_neurons_data]

    # --- Analysis ---
    # For each neuron i, create a vector of correlations with all other neurons j
    all_neurons_correlation_vectors = []

    for i in range(num_neurons):
        correlation_vector = np.zeros(num_neurons, dtype=int)
        
        # Iterate through each spike of the 'source' neuron i
        for spike_time_i in neuron_spike_times[i]:
            
            # Compare with all 'target' neurons j
            for j in range(num_neurons):
                
                # Count spikes in neuron j within the time window of spike_time_i
                for spike_time_j in neuron_spike_times[j]:
                    if abs(spike_time_i - spike_time_j) <= 0.001:
                        correlation_vector[j] += 1
        
        all_neurons_correlation_vectors.append(correlation_vector)
        print(f"Neuron {i+1} correlation vector: {correlation_vector}")

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
        
        # Save the plot to a file
        plot_filename = f"neuron_{i+1}_histogram.png"
        plt.savefig(plot_filename)
        print(f"Saved histogram for neuron {i+1} to '{plot_filename}'")
        
        plt.close() # Close the figure to free up memory

if __name__ == '__main__':
    # Assuming 'neuron_data.json' is in the same directory
    analyze_neuron_spikes('neuron_data.json')
