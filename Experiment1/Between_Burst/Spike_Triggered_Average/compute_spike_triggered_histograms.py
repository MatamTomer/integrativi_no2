import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────
WINDOW_MS  = 10          # look ±10ms around each trigger spike
BIN_EDGES  = list(range(-WINDOW_MS, 0)) + list(range(1, WINDOW_MS + 1))
# = [-10, -9, ..., -1, 1, 2, ..., 10]  → 20 bins, zero excluded
NUM_BINS   = len(BIN_EDGES)   # 20

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
print("Loading neuron data...")
with open('../neuron_data.json', 'r') as f:
    neuron_data = json.load(f)

# Filter out neurons with no spikes (can happen after burst splitting)
neuron_data = [neuron for neuron in neuron_data if neuron]

N = len(neuron_data)
print(f"  {N} neurons found")

# Extract spike times (seconds) for each neuron, sorted ascending
spike_times = []
for neuron in neuron_data:
    times = np.array([spike['spikeTime'] for spike in neuron], dtype=float)
    times.sort()
    spike_times.append(times)

neuron_ids = [neuron[0]['unit'] for neuron in neuron_data]

total_spikes = sum(len(t) for t in spike_times)
print(f"  {total_spikes:,} total spikes")
print(f"  Neuron IDs: {neuron_ids}")

# ─────────────────────────────────────────────
# Compute histograms
# ─────────────────────────────────────────────
# histograms[i][j][k] = number of times neuron j fired at offset BIN_EDGES[k]
#                        relative to a spike of neuron i
histograms = np.zeros((N, N, NUM_BINS), dtype=int)

WINDOW_SEC = WINDOW_MS / 1000.0   # convert window to seconds for comparisons

print(f"\nComputing spike-triggered histograms (window = ±{WINDOW_MS}ms)...")

for i in range(N):
    trigger_spikes = spike_times[i]   # every spike of neuron i is a trigger

    if len(trigger_spikes) == 0:
        continue

    print(f"  Neuron {neuron_ids[i]} ({i+1}/{N}): {len(trigger_spikes)} trigger spikes")

    for j in range(N):
        if i == j:
            continue   # skip self

        other_spikes = spike_times[j]

        if len(other_spikes) == 0:
            continue

        # For each trigger spike of neuron i, find nearby spikes of neuron j
        for t in trigger_spikes:

            # Binary search: find the slice of other_spikes within [t-10ms, t+10ms]
            lo = np.searchsorted(other_spikes, t - WINDOW_SEC, side='left')
            hi = np.searchsorted(other_spikes, t + WINDOW_SEC, side='right')

            nearby = other_spikes[lo:hi]   # spikes of j within the window

            for t_j in nearby:
                # Compute offset in ms, rounded to nearest integer
                offset_ms = int(round((t_j - t) * 1000))

                # Skip the zero bin (simultaneous spikes)
                if offset_ms == 0:
                    continue

                # Find which bin this offset belongs to
                # BIN_EDGES = [-10,...,-1, 1,...,10]
                if offset_ms in range(-WINDOW_MS, 0):
                    bin_idx = offset_ms + WINDOW_MS          # -10 → 0, -1 → 9
                elif offset_ms in range(1, WINDOW_MS + 1):
                    bin_idx = offset_ms + WINDOW_MS - 1      # 1 → 10, 10 → 19
                else:
                    continue   # outside ±10ms (edge case from rounding)

                histograms[i][j][bin_idx] += 1

print("\nDone computing histograms.")

# ─────────────────────────────────────────────
# Save histogram data to JSON
# ─────────────────────────────────────────────
output = {
    'neuron_ids':  neuron_ids,
    'bins':        BIN_EDGES,
    'window_ms':   WINDOW_MS,
    'n_neurons':   N,
    'description': 'histograms[i][j][k] = count of neuron j spikes at offset bins[k] ms relative to neuron i spikes',
    'histograms':  histograms.tolist()
}

json_path = Path('histogram_data.json')
with open(json_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"Saved histogram data to: {json_path}")

# ─────────────────────────────────────────────
# Visualize: one PDF page per trigger neuron
# ─────────────────────────────────────────────
pdf_path = Path('spike_triggered_histograms.pdf')
print(f"Creating PDF: {pdf_path}")

cols = 7
rows = (N - 1 + cols - 1) // cols   # enough rows to fit N-1 subplots

with PdfPages(pdf_path) as pdf:
    for i in range(N):
        fig = plt.figure(figsize=(17, 2.5 * rows))
        fig.suptitle(f'Spike-Triggered Histograms — Trigger: Neuron {neuron_ids[i]}',
                     fontsize=13, fontweight='bold')

        plot_idx = 0
        for j in range(N):
            if i == j:
                continue

            ax = fig.add_subplot(rows, cols, plot_idx + 1)
            ax.bar(BIN_EDGES, histograms[i][j], width=0.8,
                   color='steelblue', edgecolor='black', linewidth=0.3, alpha=0.8)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.set_title(f'Neuron {neuron_ids[j]}', fontsize=7, fontweight='bold')
            ax.set_xlabel('Offset (ms)', fontsize=5)
            ax.set_ylabel('Count', fontsize=5)
            ax.tick_params(labelsize=5)
            ax.set_xticks(BIN_EDGES[::2])   # every other tick to avoid crowding

            plot_idx += 1

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"  Page {i+1}/{N} done (neuron {neuron_ids[i]})")

print(f"\nAll done.")
print(f"  Histogram data : {json_path}")
print(f"  PDF            : {pdf_path}")
