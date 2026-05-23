import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

print("Loading network activity data...")
with open('network_activity_by_spike.json', 'r') as f:
    data = json.load(f)

neuron_activity_histograms = data['neuron_activity_histograms']
N = data['n_neurons']

print(f"Creating combined PDF with {N} neurons...")
print("Each neuron will have 2 pages: counts and percentages")

# Create PDF
pdf_path = 'network_activity_combined.pdf'
with PdfPages(pdf_path) as pdf:

    for neuron_id in range(N):
        network_activities = neuron_activity_histograms[neuron_id]

        if len(network_activities) == 0:
            print(f"Skipping neuron {neuron_id + 1} (no spikes)")
            continue

        print(f"Adding neuron {neuron_id + 1}/{N}...")

        max_activity = max(network_activities)
        min_activity = min(network_activities)
        bins = range(min_activity, max_activity + 2)

        # PAGE 1: Counts
        fig, ax = plt.subplots(figsize=(10, 6))

        counts, bin_edges, _ = ax.hist(network_activities, bins=bins,
                                        edgecolor='black', alpha=0.7, color='steelblue')

        ax.set_xlabel('Network Activity Level (# of other active neurons)', fontsize=12)
        ax.set_ylabel('Number of Spikes', fontsize=12)
        ax.set_title(f'Neuron {neuron_id + 1}: Spike Distribution by Network Activity (Counts)\n'
                    f'Total spikes: {len(network_activities)}, Mean activity: {np.mean(network_activities):.2f}',
                    fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add statistics box
        stats_text = f'Min: {min_activity}\n'
        stats_text += f'Max: {max_activity}\n'
        stats_text += f'Median: {np.median(network_activities):.1f}\n'
        stats_text += f'Std: {np.std(network_activities):.2f}'

        ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close()

        # PAGE 2: Percentages
        fig, ax = plt.subplots(figsize=(10, 6))

        total_spikes = len(network_activities)
        percentages = (counts / total_spikes) * 100

        # Create bar plot with percentages
        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
        ax.bar(bin_centers, percentages, width=0.8,
               edgecolor='black', alpha=0.7, color='coral')

        ax.set_xlabel('Network Activity Level (# of other active neurons)', fontsize=12)
        ax.set_ylabel('Percentage of Spikes (%)', fontsize=12)
        ax.set_title(f'Neuron {neuron_id + 1}: Spike Distribution by Network Activity (Percentages)\n'
                    f'Total spikes: {len(network_activities)}, Mean activity: {np.mean(network_activities):.2f}',
                    fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add statistics box with percentages
        stats_text = f'Min: {min_activity}\n'
        stats_text += f'Max: {max_activity}\n'
        stats_text += f'Median: {np.median(network_activities):.1f}\n'
        stats_text += f'Std: {np.std(network_activities):.2f}\n\n'
        stats_text += f'Peak %: {np.max(percentages):.1f}%'

        ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close()

print(f"\nDone! Combined PDF saved to: {pdf_path}")
print(f"Total pages: {N * 2} ({N} neurons × 2 pages each)")
