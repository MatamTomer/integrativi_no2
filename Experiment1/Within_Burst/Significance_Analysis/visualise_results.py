import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
print("Loading data...")

with open('../Spike_Triggered_Average/histogram_data.json', 'r') as f:
    hist_data = json.load(f)

with open('../Null_Hypothesis/null_histograms.json', 'r') as f:
    null_mean_data = json.load(f)

with open('../Null_Hypothesis/null_std.json', 'r') as f:
    null_std_data = json.load(f)

with open('../Null_Hypothesis/p_high.json', 'r') as f:
    p_high_data = json.load(f)

with open('../Null_Hypothesis/p_low.json', 'r') as f:
    p_low_data = json.load(f)

observed    = np.array(hist_data['histograms'])       # (N, N, 20)
null_mean   = np.array(null_mean_data['data'])        # (N, N, 20)
null_std    = np.array(null_std_data['data'])         # (N, N, 20)
p_high      = np.array(p_high_data['data'])           # (N, N, 20)
p_low       = np.array(p_low_data['data'])            # (N, N, 20)

neuron_ids  = hist_data['neuron_ids']
bins        = np.array(hist_data['bins'])             # [-10...-1, 1...10]
N           = len(neuron_ids)

print(f"  {N} neurons, {N*(N-1)} pairs")

# Z-score: (observed - null_mean) / null_std, with divide-by-zero protection
with np.errstate(invalid='ignore', divide='ignore'):
    z_scores = np.where(null_std > 0,
                        (observed - null_mean) / null_std,
                        0.0)

# ─────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────
COLS = 7
ROWS = (N - 1 + COLS - 1) // COLS   # rows needed for N-1 subplots


def make_figure(trigger_id):
    fig = plt.figure(figsize=(17, 2.5 * ROWS))
    fig.suptitle(f'Trigger: Neuron {trigger_id}', fontsize=13, fontweight='bold')
    return fig


def style_ax(ax, xlabel, ylabel, title):
    ax.set_title(title, fontsize=7, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=5)
    ax.set_ylabel(ylabel, fontsize=5)
    ax.tick_params(labelsize=5)
    ax.set_xticks(bins[::2])


# ─────────────────────────────────────────────
# PDF 1 — Observed histogram + null envelope
# ─────────────────────────────────────────────
print("Creating PDF 1: observed histograms with null envelope...")

with PdfPages('histograms_with_null_envelope.pdf') as pdf:
    for i in range(N):
        fig = make_figure(neuron_ids[i])
        plot_idx = 0

        for j in range(N):
            if i == j:
                continue

            ax = fig.add_subplot(ROWS, COLS, plot_idx + 1)

            obs  = observed[i][j]

            # Skip pairs with fewer than 300 correlated spikes
            if obs.sum() < 300:
                ax.text(0.5, 0.5, f'Neuron {neuron_ids[j]}\n< 300 spikes',
                        ha='center', va='center', fontsize=6, color='gray',
                        transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                plot_idx += 1
                continue

            nmean = null_mean[i][j]
            nstd  = null_std[i][j]

            # Observed bars
            ax.bar(bins, obs, width=0.8, color='steelblue',
                   edgecolor='black', linewidth=0.2, alpha=0.85, label='Observed')

            # Null envelope: one shaded rectangle per bin showing null_mean ± null_std
            for k, b in enumerate(bins):
                ax.bar(b, 2 * nstd[k],
                       bottom=nmean[k] - nstd[k],
                       width=0.8, color='orange', alpha=0.4,
                       edgecolor='none',
                       label='Null ±1 SD' if k == 0 else None)
                # Null mean tick
                ax.plot([b - 0.4, b + 0.4], [nmean[k], nmean[k]],
                        color='orange', linewidth=0.8, alpha=0.9)

            ax.axvline(x=0, color='red', linestyle='--', linewidth=0.6, alpha=0.5)

            style_ax(ax,
                     xlabel='Time offset (ms)',
                     ylabel='Spike count',
                     title=f'Neuron {neuron_ids[j]}')

            # Legend only on first subplot of each page
            if plot_idx == 0:
                ax.legend(fontsize=4, loc='upper right')

            plot_idx += 1

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"  Page {i+1}/{N} (neuron {neuron_ids[i]})")

print("  Saved: histograms_with_null_envelope.pdf")

# ─────────────────────────────────────────────
# PDF 2 — Z-score bars (color coded)
# ─────────────────────────────────────────────
print("Creating PDF 2: Z-score histograms...")

def z_color(z_val):
    az = abs(z_val)
    if az >= 3:
        return 'red'
    elif az >= 2:
        return 'orange'
    else:
        return 'steelblue'

with PdfPages('z_score_histograms.pdf') as pdf:
    for i in range(N):
        fig = make_figure(neuron_ids[i])
        plot_idx = 0

        for j in range(N):
            if i == j:
                continue

            ax = fig.add_subplot(ROWS, COLS, plot_idx + 1)

            if observed[i][j].sum() < 300:
                ax.text(0.5, 0.5, f'Neuron {neuron_ids[j]}\n< 300 spikes',
                        ha='center', va='center', fontsize=6, color='gray',
                        transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                plot_idx += 1
                continue

            z = z_scores[i][j]
            colors = [z_color(z_val) for z_val in z]

            ax.bar(bins, z, width=0.8, color=colors,
                   edgecolor='black', linewidth=0.2, alpha=0.85)

            # Reference lines
            ax.axhline(y=0,  color='black',  linewidth=0.5, alpha=0.5)
            ax.axhline(y=2,  color='orange', linewidth=0.6, linestyle='--', alpha=0.6)
            ax.axhline(y=-2, color='orange', linewidth=0.6, linestyle='--', alpha=0.6)
            ax.axhline(y=3,  color='red',    linewidth=0.6, linestyle='--', alpha=0.6)
            ax.axhline(y=-3, color='red',    linewidth=0.6, linestyle='--', alpha=0.6)
            ax.axvline(x=0,  color='red',    linewidth=0.6, linestyle=':',  alpha=0.5)

            style_ax(ax,
                     xlabel='Time offset (ms)',
                     ylabel='Z-score (SD units)',
                     title=f'Neuron {neuron_ids[j]}')

            plot_idx += 1

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"  Page {i+1}/{N} (neuron {neuron_ids[i]})")

print("  Saved: z_score_histograms.pdf")

# ─────────────────────────────────────────────
# PDF 3 — p-values (p_high and p_low side by side)
# ─────────────────────────────────────────────
print("Creating PDF 3: p-value histograms...")

BAR_WIDTH = 0.38
P_THRESHOLD = 0.05

with PdfPages('p_value_histograms.pdf') as pdf:
    for i in range(N):
        fig = make_figure(neuron_ids[i])
        plot_idx = 0

        for j in range(N):
            if i == j:
                continue

            ax = fig.add_subplot(ROWS, COLS, plot_idx + 1)

            if observed[i][j].sum() < 300:
                ax.text(0.5, 0.5, f'Neuron {neuron_ids[j]}\n< 300 spikes',
                        ha='center', va='center', fontsize=6, color='gray',
                        transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                plot_idx += 1
                continue

            ph = p_high[i][j]
            pl = p_low[i][j]

            # Side-by-side bars: p_high left of bin center, p_low right
            ax.bar(bins - BAR_WIDTH / 2, ph, width=BAR_WIDTH,
                   color='coral',       edgecolor='black', linewidth=0.2,
                   alpha=0.85, label='p_high (excitation)')
            ax.bar(bins + BAR_WIDTH / 2, pl, width=BAR_WIDTH,
                   color='teal',        edgecolor='black', linewidth=0.2,
                   alpha=0.85, label='p_low (inhibition)')

            # Significance threshold
            ax.axhline(y=P_THRESHOLD, color='black', linewidth=0.7,
                       linestyle='--', alpha=0.7, label='p = 0.05')
            ax.axvline(x=0, color='red', linewidth=0.6, linestyle=':', alpha=0.5)

            ax.set_ylim([0, 1.05])

            style_ax(ax,
                     xlabel='Time offset (ms)',
                     ylabel='p-value (proportion)',
                     title=f'Neuron {neuron_ids[j]}')

            # Show minimum p-values as text in the subplot
            min_ph = np.min(ph)
            min_pl = np.min(pl)
            ax.text(0.02, 0.97,
                    f'min p_high={min_ph:.4f}\nmin p_low={min_pl:.4f}',
                    transform=ax.transAxes, fontsize=4,
                    verticalalignment='top',
                    color='black',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.7, linewidth=0.3))

            if plot_idx == 0:
                ax.legend(fontsize=4, loc='upper right')

            plot_idx += 1

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"  Page {i+1}/{N} (neuron {neuron_ids[i]})")

print("  Saved: p_value_histograms.pdf")
print("\nAll done.")
