import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

EXPERIMENT2 = Path(__file__).parent.parent

CONFIG_PATHS = {
    'all':          EXPERIMENT2 / 'Spike_Triggered_Average' / 'p_ij.json',
    'within_burst': EXPERIMENT2 / 'Within_Burst' / 'Spike_Triggered_Average' / 'p_ij.json',
    'between_burst':EXPERIMENT2 / 'Between_Burst' / 'Spike_Triggered_Average' / 'p_ij.json',
}

parser = argparse.ArgumentParser()
parser.add_argument('--config', choices=['all', 'within_burst', 'between_burst'], default='all')
parser.add_argument('--pairs_per_page', type=int, default=20,
                    help='Number of neuron pairs per PDF page')
args = parser.parse_args()

print(f"Loading p_ij for config: {args.config}...")
with open(CONFIG_PATHS[args.config]) as f:
    data = json.load(f)

p_ij      = np.array(data['p_ij'])    # (N, N, NUM_BINS)
neuron_ids = data['neuron_ids']
bins       = data['bins']
N          = data['n_neurons']
bins_arr   = np.array(bins)

print(f"  {N} neurons, {len(bins)} bins")

# ─────────────────────────────────────────────
# Collect all non-self pairs sorted by peak P_ij descending
# ─────────────────────────────────────────────
pairs = []
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        peak = p_ij[i, j].max()
        if peak > 0:
            pairs.append((i, j, peak))

pairs.sort(key=lambda x: -x[2])
print(f"  {len(pairs)} non-zero pairs")

# ─────────────────────────────────────────────
# Plot: one bar chart per pair, grouped into pages
# ─────────────────────────────────────────────
PER_PAGE = args.pairs_per_page
COLS     = 4
ROWS     = (PER_PAGE + COLS - 1) // COLS

n_pages    = (len(pairs) + PER_PAGE - 1) // PER_PAGE
PAGES_PER_PDF = 40   # ~800 pairs per file — keeps each PDF small and loadable
n_parts    = (n_pages + PAGES_PER_PDF - 1) // PAGES_PER_PDF
out_dir    = Path(__file__).parent

from matplotlib.backends.backend_pdf import PdfPages

print(f"Generating {n_pages} pages across {n_parts} PDFs...")

for part in range(n_parts):
    page_start = part * PAGES_PER_PDF
    page_end   = min(page_start + PAGES_PER_PDF, n_pages)
    out_path   = out_dir / f'pij_histograms_{args.config}_part{part+1}.pdf'

    with PdfPages(out_path) as pdf:
        for page in range(page_start, page_end):
            page_pairs = pairs[page * PER_PAGE : (page + 1) * PER_PAGE]
            n_plots = len(page_pairs)
            actual_rows = (n_plots + COLS - 1) // COLS

            fig, axes = plt.subplots(actual_rows, COLS,
                                     figsize=(COLS * 3.5, actual_rows * 2.8),
                                     squeeze=False)
            fig.suptitle(
                f'P_ij(tau) histograms — config: {args.config}  '
                f'(page {page+1}/{n_pages}, part {part+1}/{n_parts}, '
                f'pairs ranked by peak P_ij)',
                fontsize=10
            )

            for k, (i, j, peak) in enumerate(page_pairs):
                ax = axes[k // COLS][k % COLS]
                vals = p_ij[i, j]
                colors = ['steelblue' if b < 0 else ('tomato' if b > 0 else 'gray')
                          for b in bins]
                ax.bar(bins_arr, vals, width=0.8, color=colors)
                ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
                ax.set_title(f'N{neuron_ids[i]} -> N{neuron_ids[j]}\npeak={peak:.4f}',
                             fontsize=7)
                ax.set_xlabel('tau (ms)', fontsize=6)
                ax.set_ylabel('P_ij(tau)', fontsize=6)
                ax.tick_params(labelsize=5)

            # Hide unused axes
            for k in range(n_plots, actual_rows * COLS):
                axes[k // COLS][k % COLS].set_visible(False)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"  Saved: {out_path}  (pages {page_start+1}-{page_end})")

print("Done.")
