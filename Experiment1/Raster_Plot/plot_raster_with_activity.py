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
# 2. Load precomputed population activity
#    (output of compute_activity.py)
# ─────────────────────────────────────────────
print("Loading activity data...")

data        = np.load('activity.npz')
time_s      = data['time_s']       # time axis in seconds, 10ms resolution
activity    = data['activity']     # spike count per 50ms window
print(f"  {len(time_s):,} time points loaded")

# ─────────────────────────────────────────────
# 3. Create figure with two linked subplots
#
#    shared_xaxes=True means both panels zoom/pan together
#    on the time axis — dragging one panel moves the other.
#
#    Row 1 (tall): raster plot — one row per neuron
#    Row 2 (short): population activity trace
# ─────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,          # link the time axes
    row_heights=[0.75, 0.25],   # raster gets 75% of the height
    vertical_spacing=0.04,
)

# ─────────────────────────────────────────────
# 4. Raster plot — one Scattergl trace per neuron
# ─────────────────────────────────────────────
print("Building raster traces...")

colors = [f'hsl({int(360 * i / N)}, 70%, 50%)' for i in range(N)]

for idx, (nid, times) in enumerate(zip(neuron_ids, spike_times)):
    fig.add_trace(
        go.Scattergl(
            x=times,
            y=np.full(len(times), idx),
            mode='markers',
            marker=dict(
                symbol='line-ns',
                size=4,
                color=colors[idx],
                line=dict(width=1, color=colors[idx])
            ),
            name=f'Neuron {nid}',
            legendgroup=f'neuron_{nid}',
            hovertemplate=(
                f'<b>Neuron {nid}</b><br>'
                'Time: %{x:.3f} s<br>'
                '<extra></extra>'
            ),
        ),
        row=1, col=1
    )

# ─────────────────────────────────────────────
# 5. Population activity trace
#
#    Filled area chart showing how many spikes occurred
#    in a 50ms window centered on each time point.
# ─────────────────────────────────────────────
print("Building activity trace...")

fig.add_trace(
    go.Scattergl(
        x=time_s,
        y=activity,
        mode='lines',
        line=dict(color='rgba(50, 50, 200, 0.8)', width=1),
        fill='tozeroy',
        fillcolor='rgba(50, 50, 200, 0.15)',
        name='Population activity',
        legendgroup='activity',
        hovertemplate=(
            'Time: %{x:.3f} s<br>'
            'Activity: %{y:.0f} spikes / 50ms<br>'
            '<extra></extra>'
        ),
    ),
    row=2, col=1
)

# ─────────────────────────────────────────────
# 6. Burst regions — single binary trace per panel
#
#    Instead of one rectangle per burst (slow), we build a
#    single trace that is 1 inside bursts and 0 outside.
#    This renders instantly regardless of burst count.
# ─────────────────────────────────────────────
if Path('bursts_gaussian.json').exists():
    print("Loading burst regions...")
    with open('bursts_gaussian.json', 'r') as f:
        burst_data = json.load(f)

    bursts = burst_data['bursts']
    print(f"  {len(bursts)} bursts — building binary trace...")

    # Build a burst mask on the activity time axis
    burst_mask = np.zeros(len(time_s), dtype=np.float32)
    for burst in bursts:
        lo = np.searchsorted(time_s, burst['t_start_s'], side='left')
        hi = np.searchsorted(time_s, burst['t_end_s'],   side='right')
        burst_mask[lo:hi] = 1.0

    # Scale to plot height for each panel and add as filled traces
    for row, scale in [(1, N - 1), (2, float(activity.max()))]:
        fig.add_trace(
            go.Scattergl(
                x=time_s,
                y=burst_mask * scale,
                mode='lines',
                line=dict(width=0),
                fill='tozeroy',
                fillcolor='rgba(255, 50, 50, 0.15)',
                name='Burst region',
                legendgroup='bursts',
                showlegend=(row == 1),
                hoverinfo='skip',
            ),
            row=row, col=1
        )
else:
    print("No bursts.json found — skipping burst overlay")

# ─────────────────────────────────────────────
# 7. Layout
# ─────────────────────────────────────────────
fig.update_layout(
    title=dict(text='Raster plot + population activity (50ms window)', font=dict(size=16)),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=800,
    width=1400,
    legend=dict(
        title='Neurons (click to toggle)',
        font=dict(size=9),
        itemclick='toggle',
        itemdoubleclick='toggleothers',
    ),
    hovermode='x unified',   # show all neuron values at the same time point on hover
)

# Raster y-axis: fixed range, neuron labels
fig.update_yaxes(
    tickvals=list(range(N)),
    ticktext=[f'N{nid}' for nid in neuron_ids],
    showgrid=False,
    range=[-0.5, N - 0.5],
    fixedrange=True,
    title_text='Neuron',
    row=1, col=1
)

# Activity y-axis
fig.update_yaxes(
    showgrid=True,
    gridcolor='rgba(200,200,200,0.4)',
    fixedrange=True,
    title_text='Spikes / 50ms',
    row=2, col=1
)

# Shared x-axis label on the bottom panel only
fig.update_xaxes(
    title_text='Time (s)',
    showgrid=True,
    gridcolor='rgba(200,200,200,0.3)',
    row=2, col=1
)

# ─────────────────────────────────────────────
# 8. Save
# ─────────────────────────────────────────────
output_path = 'raster_gaussian.html'
fig.write_html(output_path, include_plotlyjs='cdn')
print(f"Saved: {output_path}")
print("Open in any browser — zoom/pan on either panel moves both.")
