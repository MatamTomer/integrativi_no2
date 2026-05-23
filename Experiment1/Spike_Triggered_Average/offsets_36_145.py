import json
import numpy as np
from pathlib import Path

WINDOW_MS  = 10
WINDOW_SEC = WINDOW_MS / 1000.0

with open(Path(__file__).parent.parent / 'Experiment2' / 'neuron_data.json', 'r') as f:
    neuron_data = json.load(f)

neuron_ids = [neuron[0]['unit'] for neuron in neuron_data if neuron]
spike_times = {
    nid: np.array([spike['spikeTime'] for spike in neuron], dtype=np.float64)
    for nid, neuron in zip(neuron_ids, neuron_data)
}

spikes_93  = np.sort(spike_times[93])
spikes_145 = np.sort(spike_times[145])

offsets_ms = []
for t in spikes_93:
    lo = np.searchsorted(spikes_145, t - WINDOW_SEC, side='left')
    hi = np.searchsorted(spikes_145, t + WINDOW_SEC, side='right')
    for t_j in spikes_145[lo:hi]:
        offsets_ms.append((t_j - t) * 1000.0)

offsets_ms = np.array(offsets_ms)

with open(Path(__file__).parent / 'offsets_93_145.json', 'w') as f:
    json.dump(offsets_ms.tolist(), f)
print("Saved to offsets_93_145.json")
