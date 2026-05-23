import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

print("Loading histogram data...")
with open(Path(__file__).parent / 'histogram_data.json', 'r') as f:
    hist_data = json.load(f)

import numpy as np
neuron_ids  = hist_data['neuron_ids']
BIN_EDGES   = hist_data['bins']
histograms  = np.array(hist_data['histograms'])
N           = hist_data['n_neurons']
print(f"  {N} neurons, {len(BIN_EDGES)} bins")

cols      = 7
rows      = (N - 1 + cols - 1) // cols
N_PARTS   = 4
part_size = (N + N_PARTS - 1) // N_PARTS

for part in range(N_PARTS):
    i_start  = part * part_size
    i_end    = min(i_start + part_size, N)
    pdf_path = Path(__file__).parent / f'spike_triggered_histograms_part{part + 1}.pdf'
    print(f"Creating PDF {part + 1}/4: neurons {neuron_ids[i_start]}-{neuron_ids[i_end - 1]} -> {pdf_path.name}")

    with PdfPages(pdf_path) as pdf:
        for i in range(i_start, i_end):
            fig = plt.figure(figsize=(17, 2.5 * rows))
            fig.suptitle(f'Spike-Triggered Histograms - Trigger: Neuron {neuron_ids[i]}',
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
                ax.set_xticks(BIN_EDGES[::2])
                plot_idx += 1

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print(f"  Page {i - i_start + 1}/{i_end - i_start} (neuron {neuron_ids[i]})")

    print(f"  Saved: {pdf_path.name}")

print("\nAll done.")
