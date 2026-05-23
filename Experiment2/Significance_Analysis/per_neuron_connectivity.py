import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THRESHOLD = 10

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
print("Loading...")
df = pd.read_csv('sij_v2_scores.csv')

all_units = sorted(set(df['trigger_unit']) | set(df['response_unit']))
N = len(all_units)
print(f"  {N} units, {len(df)} pairs")

sig = df[df['S_ij'].abs() > THRESHOLD].copy()
excit = sig[sig['S_ij'] >  THRESHOLD]
inhib = sig[sig['S_ij'] < -THRESHOLD]

# ─────────────────────────────────────────────
# Per-neuron counts
# ─────────────────────────────────────────────
out_excit = excit.groupby('trigger_unit').size().reindex(all_units, fill_value=0)
out_inhib = inhib.groupby('trigger_unit').size().reindex(all_units, fill_value=0)
in_excit  = excit.groupby('response_unit').size().reindex(all_units, fill_value=0)
in_inhib  = inhib.groupby('response_unit').size().reindex(all_units, fill_value=0)

total_out = out_excit + out_inhib
total_in  = in_excit  + in_inhib
total     = total_out + total_in

summary = pd.DataFrame({
    'unit':       all_units,
    'out_excit':  out_excit.values,
    'out_inhib':  out_inhib.values,
    'in_excit':   in_excit.values,
    'in_inhib':   in_inhib.values,
    'total_out':  total_out.values,
    'total_in':   total_in.values,
    'total':      total.values,
})
summary.to_csv('per_neuron_connectivity.csv', index=False)
print("Saved: per_neuron_connectivity.csv")

# ─────────────────────────────────────────────
# Plot 1 — Stacked bar charts (outgoing + incoming)
# ─────────────────────────────────────────────
sort_order = np.argsort(-total.values)
units_sorted = [all_units[i] for i in sort_order]

oe = out_excit.values[sort_order]
oi = out_inhib.values[sort_order]
ie = in_excit.values[sort_order]
ii = in_inhib.values[sort_order]

y = np.arange(N)

fig, axes = plt.subplots(1, 2, figsize=(16, max(8, N * 0.18)),
                         sharey=True)
fig.suptitle(
    f'Per-neuron significant connections  (|S_ij| > {THRESHOLD})\n'
    f'Full dataset — {N} neurons',
    fontsize=13, fontweight='bold'
)

# Left: outgoing
ax = axes[0]
ax.barh(y, oe, color='coral',     alpha=0.85, label='Excitatory outgoing')
ax.barh(y, oi, left=oe, color='steelblue', alpha=0.85, label='Inhibitory outgoing')
ax.set_xlabel('Number of connections', fontsize=11)
ax.set_title('Outgoing connections\n(neuron as trigger)', fontsize=11, fontweight='bold')
ax.set_yticks(y)
ax.set_yticklabels(units_sorted, fontsize=6)
ax.legend(fontsize=9, loc='lower right')
ax.axvline(x=0, color='black', linewidth=0.5)

# Right: incoming
ax = axes[1]
ax.barh(y, ie, color='coral',     alpha=0.85, label='Excitatory incoming')
ax.barh(y, ii, left=ie, color='steelblue', alpha=0.85, label='Inhibitory incoming')
ax.set_xlabel('Number of connections', fontsize=11)
ax.set_title('Incoming connections\n(neuron as target)', fontsize=11, fontweight='bold')
ax.set_yticks(y)
ax.set_yticklabels(units_sorted, fontsize=6)
ax.legend(fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('per_neuron_bars.pdf', bbox_inches='tight')
plt.close()
print("Saved: per_neuron_bars.pdf")

# ─────────────────────────────────────────────
# Plot 2 — Scatter: excitatory vs inhibitory
# (outgoing + incoming combined, per neuron)
# ─────────────────────────────────────────────
tot_excit = (out_excit + in_excit).values
tot_inhib = (out_inhib + in_inhib).values

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f'Excitatory vs inhibitory connections per neuron  (|S_ij| > {THRESHOLD})',
    fontsize=13, fontweight='bold'
)

for ax, xe, yi, title in [
    (axes[0], tot_excit,          tot_inhib,          'All connections (outgoing + incoming)'),
    (axes[1], out_excit.values,   out_inhib.values,   'Outgoing only'),
]:
    ax.scatter(xe, yi, s=40, color='mediumpurple', alpha=0.7, edgecolors='black',
               linewidths=0.4)
    # Diagonal: equal excit / inhib
    max_val = max(xe.max(), yi.max(), 1)
    ax.plot([0, max_val], [0, max_val], color='gray', linestyle='--',
            linewidth=0.8, alpha=0.6, label='Equal excit/inhib')
    # Label outlier neurons (top 10 by total)
    top_idx = np.argsort(-(xe + yi))[:10]
    for i in top_idx:
        ax.annotate(str(all_units[i]), (xe[i], yi[i]),
                    fontsize=6, ha='left', va='bottom',
                    xytext=(2, 2), textcoords='offset points')
    ax.set_xlabel('Excitatory connections', fontsize=11)
    ax.set_ylabel('Inhibitory connections', fontsize=11)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('per_neuron_scatter.pdf', bbox_inches='tight')
plt.close()
print("Saved: per_neuron_scatter.pdf")
print("Done.")
