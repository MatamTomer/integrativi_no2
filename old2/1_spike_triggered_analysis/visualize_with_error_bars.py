import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def load_data():
    """Load all required data files."""
    # Load original histograms
    with open('spike_triggered_histograms/histogram_data.json', 'r') as f:
        hist_data = json.load(f)
    histograms = np.array(hist_data['histograms'])
    bins = np.array(hist_data['bins'])
    neuron_ids = hist_data['neuron_ids']

    # Load null hypothesis data (mu and sigma)
    with open('significance_analysis/null_hypothesis_data.json', 'r') as f:
        null_data = json.load(f)
    mu = np.array(null_data['mu'])
    sigma = np.array(null_data['sigma'])

    # Load Z-scores for color coding
    with open('significance_analysis/z_scores.json', 'r') as f:
        z_data = json.load(f)
    z_scores = np.array(z_data['z_scores'])

    return histograms, bins, neuron_ids, mu, sigma, z_scores


def create_histogram_with_envelope_page(neuron_i, neuron_id_i, histograms, bins, neuron_ids, mu, sigma, z_scores):
    """
    Create a PDF page showing spike-triggered histograms with error envelopes.

    Shows:
    - Shaded region: mu +- sigma (null distribution envelope)
    - Bars: Observed histogram (color-coded by Z-score significance)
    """
    N = len(neuron_ids)
    num_others = N - 1

    cols = 7
    rows = (num_others + cols - 1) // cols

    fig = plt.figure(figsize=(17, 2.5 * rows))
    fig.suptitle(f'Spike-Triggered Histogram with Error Envelope: Neuron {neuron_id_i} (trigger)',
                 fontsize=14, fontweight='bold')

    plot_idx = 0
    for j in range(N):
        if j == neuron_i:
            continue

        neuron_id_j = neuron_ids[j]
        observed = histograms[neuron_i][j]
        mu_ij = mu[neuron_i][j]
        sigma_ij = sigma[neuron_i][j]
        z_ij = z_scores[neuron_i][j]

        ax = fig.add_subplot(rows, cols, plot_idx + 1)

        # Check if this pair was skipped (all NaN)
        if np.all(np.isnan(mu_ij)):
            ax.text(0.5, 0.5, 'Skipped\n(<300 spikes)', ha='center', va='center',
                   fontsize=6, color='gray', transform=ax.transAxes)
            ax.set_title(f'Neuron {neuron_id_j}', fontsize=7, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            plot_idx += 1
            continue

        # Plot mu (null mean) as background bars
        ax.bar(bins, mu_ij, width=0.8, color='lightgray', alpha=0.5,
              edgecolor='none', label='mu (null mean)')

        # Add error bars showing sigma range
        ax.errorbar(bins, mu_ij, yerr=sigma_ij, fmt='none',
                   ecolor='gray', elinewidth=1, capsize=2, alpha=0.6,
                   label='mu +- sigma')

        # Color bars by Z-score significance
        colors = []
        for z_val in z_ij:
            abs_z = abs(z_val)
            if abs_z >= 3:
                colors.append('red')
            elif abs_z >= 2:
                colors.append('orange')
            else:
                colors.append('steelblue')

        # Plot observed histogram bars (prominent, on top)
        ax.bar(bins, observed, width=0.6, color=colors, alpha=0.8,
              edgecolor='black', linewidth=0.4, label='Observed')

        # Add vertical line at zero
        ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8, alpha=0.6)

        # Labels and formatting
        ax.set_title(f'Neuron {neuron_id_j}', fontsize=7, fontweight='bold')
        if plot_idx >= (rows - 1) * cols:  # Bottom row
            ax.set_xlabel('Time offset (ms)', fontsize=6)
        if plot_idx % cols == 0:  # Left column
            ax.set_ylabel('Spike count', fontsize=6)
        ax.tick_params(axis='both', labelsize=5)
        ax.grid(True, alpha=0.2, axis='y')
        ax.set_xlim(bins[0] - 1, bins[-1] + 1)
        ax.set_ylim(bottom=0)

        # Add legend only to first subplot
        if plot_idx == 0:
            ax.legend(fontsize=4, loc='upper right')

        plot_idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main():
    output_dir = Path('spike_triggered_histograms')
    pdf_path = output_dir / 'histograms_with_error_bars.pdf'

    print("="*70)
    print("SPIKE-TRIGGERED HISTOGRAMS WITH ERROR ENVELOPES")
    print("="*70)

    # Load data
    print("\nLoading data...")
    histograms, bins, neuron_ids, mu, sigma, z_scores = load_data()
    N = len(neuron_ids)
    C = len(bins) // 2

    print(f"  Loaded {N} neurons")
    print(f"  Causality window: ±{C}ms")
    print(f"  Bins: {len(bins)} bins")

    # Create PDF
    print(f"\nCreating PDF: {pdf_path}")
    with PdfPages(pdf_path) as pdf:
        for i in range(N):
            neuron_id_i = neuron_ids[i]

            fig = create_histogram_with_envelope_page(
                i, neuron_id_i, histograms, bins, neuron_ids, mu, sigma, z_scores
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            print(f"  Created page {i + 1}/{N} (neuron {neuron_id_i})", end='\r')

    print(f"\n  Done!")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Output PDF: {pdf_path}")
    print()
    print("Visualization:")
    print("  - Light gray bars (background): mu (expected count from null)")
    print("  - Gray error bars: +- sigma (null distribution variability)")
    print("  - Colored bars (foreground): Observed spike counts")
    print("    - Blue: Not significant (|Z| < 2)")
    print("    - Orange: Significant (2 <= |Z| < 3)")
    print("    - Red: Highly significant (|Z| >= 3)")
    print("  - Red vertical line: Time of trigger spike (t=0)")
    print()
    print("Interpretation:")
    print("  - Bars above envelope: More spikes than expected (positive correlation)")
    print("  - Bars below envelope: Fewer spikes than expected (negative correlation)")
    print("  - Bars within envelope: Consistent with random chance")
    print("="*70)


if __name__ == "__main__":
    main()
