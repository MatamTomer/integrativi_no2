import json
import re

def parse_neuron_data(input_file_path, output_file_path):
    """
    Parses a text file with neuron spike data and converts it to a JSON file.

    The input file is expected to have one neuron per line, with spike data enclosed
    in curly braces {}. Each spike's data is a comma-separated list of values.
    """
    all_neurons_data = []
    keys = ["index", "instance", "spikeFrameNo", "spikeTime", "unit", "Config"]

    try:
        with open(input_file_path, 'r') as f:
            # Each line represents a neuron with its list of spikes
            for neuron_line in f:
                # Clean up the line by removing outer container braces/brackets if they exist
                # and stripping whitespace
                cleaned_line = neuron_line.strip()
                if cleaned_line.startswith('{{') and cleaned_line.endswith('}}'):
                     cleaned_line = cleaned_line[2:-2]
                elif cleaned_line.startswith('{') and cleaned_line.endswith('}'):
                    cleaned_line = cleaned_line[1:-1]


                # Find all individual spike data points, which are enclosed in braces
                spike_matches = re.findall(r'\{([^}]+)\}', cleaned_line)
                
                neuron_spikes = []
                for spike_str in spike_matches:
                    # Split the comma-separated values
                    values = [v.strip() for v in spike_str.split(',')]
                    
                    # Ensure we have the correct number of values
                    if len(values) != len(keys):
                        print(f"Skipping malformed spike data (wrong number of values): '{spike_str}'")
                        continue

                    # Convert values to appropriate types
                    try:
                        typed_values = [
                            int(values[0]),
                            int(values[1]),
                            int(values[2]),
                            float(values[3]),
                            int(values[4]),
                            int(values[5])
                        ]
                        
                        # Create a dictionary for the spike
                        spike_dict = dict(zip(keys, typed_values))
                        neuron_spikes.append(spike_dict)
                    except ValueError as e:
                        print(f"Skipping malformed spike data (type conversion error): '{spike_str}'. Error: {e}")
                        continue
                
                if neuron_spikes:
                    all_neurons_data.append(neuron_spikes)

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file_path}'")
        return

    # Write the data to a JSON file
    try:
        with open(output_file_path, 'w') as json_file:
            json.dump(all_neurons_data, json_file, indent=4)
        print(f"Successfully converted data to '{output_file_path}'")
    except IOError as e:
        print(f"Error writing to JSON file: {e}")


if __name__ == '__main__':
    # The script expects 'tomer.txt' to be in the same directory.
    # The output will be 'neuron_data.json' in the same directory.
    parse_neuron_data('tomer.txt', 'neuron_data.json')
