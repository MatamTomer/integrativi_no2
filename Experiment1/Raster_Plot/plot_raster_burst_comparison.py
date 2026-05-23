import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ─────────────────────────────────────────────
# 1. Load spike data
# ─────────────────────────────────────────────
print("Loading spike data...")

with open('../data/neuron_data.json', 'r') as f:
    neuron_data = json.load(f)

neuron_ids  = [neuron[0]['unit'] for neuron in neuron_data if neuron]
spike_times = [
    np.array([spike['spikeTime'] for spike in neuron], dtype=np.float64)
    for neuron in neuron_data
]
N = len(neuron_ids)
print(f"  {N} neurons loaded")

# ─────────────────────────────────────────────
# 2. Load activity
# ─────────────────────────────────────────────
print("Loading activity data...")

data     = np.load('activity.npz')
time_s   = data['time_s']
activity = data['activity']
print(f"  {len(time_s):,} time points loaded")

# ─────────────────────────────────────────────
# 3. Build burst masks for both methods
# ─────────────────────────────────────────────
def build_burst_mask(burst_file, time_s):
    with open(burst_file) as f:
        burst_data = json.load(f)
    mask = np.zeros(len(time_s), dtype=np.float32)
    for burst in burst_data['bursts']:
        lo = np.searchsorted(time_s, burst['t_start_s'], side='left')
        hi = np.searchsorted(time_s, burst['t_end_s'],   side='right')
        mask[lo:hi] = 1.0
    return mask, burst_data['n_bursts']

print("Loading burst files...")
mask_orig, n_orig = build_burst_mask('bursts.json',          time_s)
mask_gaus, n_gaus = build_burst_mask('bursts_gaussian.json', time_s)
print(f"  Original method:  {n_orig} bursts")
print(f"  Gaussian method:  {n_gaus} bursts")

# ─────────────────────────────────────────────
# 4. Figure layout
# ─────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.75, 0.25],
    vertical_spacing=0.04,
)

# ─────────────────────────────────────────────
# 5. Burst shading — original (red) and Gaussian (blue)
#    Add behind spikes so spikes render on top
# ─────────────────────────────────────────────
for row, scale in [(1, N - 1), (2, float(activity.max()))]:
    # Original bursts — red
    fig.add_trace(
        go.Scattergl(
            x=time_s, y=mask_orig * scale,
            mode='lines', line=dict(width=0),
            fill='tozeroy', fillcolor='rgba(220, 50, 50, 0.18)',
            name=f'Original bursts ({n_orig})',
            legendgroup='orig', showlegend=(row == 1),
            hoverinfo='skip',
        ),
        row=row, col=1
    )
    # Gaussian bursts — blue
    fig.add_trace(
        go.Scattergl(
            x=time_s, y=mask_gaus * scale,
            mode='lines', line=dict(width=0),
            fill='tozeroy', fillcolor='rgba(50, 100, 220, 0.18)',
            name=f'Gaussian bursts ({n_gaus})',
            legendgroup='gaus', showlegend=(row == 1),
            hoverinfo='skip',
        ),
        row=row, col=1
    )

# ─────────────────────────────────────────────
# 6. Raster traces
# ─────────────────────────────────────────────
print("Building raster traces...")

colors = [f'hsl({int(360 * i / N)}, 70%, 50%)' for i in range(N)]

for idx, (nid, times) in enumerate(zip(neuron_ids, spike_times)):
    fig.add_trace(
        go.Scattergl(
            x=times,
            y=np.full(len(times), idx),
            mode='markers',
            marker=dict(symbol='line-ns', size=4, color=colors[idx],
                        line=dict(width=1, color=colors[idx])),
            name=f'Neuron {nid}',
            legendgroup=f'neuron_{nid}',
            hovertemplate=f'<b>Neuron {nid}</b><br>Time: %{{x:.3f}} s<br><extra></extra>',
        ),
        row=1, col=1
    )

# ─────────────────────────────────────────────
# 7. Activity trace
# ─────────────────────────────────────────────
print("Building activity trace...")

fig.add_trace(
    go.Scattergl(
        x=time_s, y=activity,
        mode='lines',
        line=dict(color='rgba(50, 50, 200, 0.8)', width=1),
        fill='tozeroy', fillcolor='rgba(50, 50, 200, 0.10)',
        name='Population activity',
        legendgroup='activity',
        hovertemplate='Time: %{x:.3f} s<br>Activity: %{y:.0f} spikes/50ms<br><extra></extra>',
    ),
    row=2, col=1
)

# ─────────────────────────────────────────────
# 8. Layout
# ─────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text=f'Burst detection comparison — Original ({n_orig} bursts, red) vs Gaussian ({n_gaus} bursts, blue)',
        font=dict(size=15)
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=800,
    width=1400,
    legend=dict(font=dict(size=9), itemclick='toggle', itemdoubleclick='toggleothers'),
    hovermode='x unified',
)

fig.update_yaxes(
    tickvals=list(range(N)), ticktext=[f'N{nid}' for nid in neuron_ids],
    showgrid=False, range=[-0.5, N - 0.5], fixedrange=True,
    title_text='Neuron', row=1, col=1
)
fig.update_yaxes(
    showgrid=True, gridcolor='rgba(200,200,200,0.4)',
    fixedrange=True, title_text='Spikes / 50ms', row=2, col=1
)
fig.update_xaxes(
    title_text='Time (s)', showgrid=True,
    gridcolor='rgba(200,200,200,0.3)', row=2, col=1
)

# ─────────────────────────────────────────────
# 9. Save
# ─────────────────────────────────────────────
output_path = 'raster_burst_comparison.html'
fig.write_html(output_path, include_plotlyjs='cdn')
print(f"Saved: {output_path}")
print("Open in browser — red = original method, blue = Gaussian method.")
