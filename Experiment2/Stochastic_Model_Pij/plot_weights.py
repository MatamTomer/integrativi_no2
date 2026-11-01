"""
plot_weights.py
Visualise the weight distributions from weights_pij_all.json files.

Panels:
  Row 1 — Histogram: all off-diagonal weights, Exp2 vs Between_Burst (overlaid)
  Row 2 — Histogram per lag (tau = 1..5 ms) for each weight set
  Row 3 — Heatmap: peak weight matrix (max over lags) for Exp2
  Row 4 — Heatmap: peak weight matrix (max over lags) for Between_Burst
  Row 5 — Scatter: Exp2 peak weight vs Between_Burst peak weight (one dot per pair)

Output: weight_distributions.html
"""

import json
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE        = Path(__file__).parent
EXPERIMENT2 = HERE.parent

WEIGHT_FILES = {
    'Exp2':          HERE / 'weights_pij_all.json',
    'Between_Burst': EXPERIMENT2 / 'Between_Burst' / 'Stochastic_Model_Pij' / 'weights_pij_all.json',
}

COLORS = {
    'Exp2':          '#e6851e',   # orange
    'Between_Burst': '#3d9db3',   # teal
}

TAU_LABELS = [f'τ={t}ms' for t in [1, 2, 3, 4, 5]]
LAGS       = [1, 2, 3, 4, 5]

# ─────────────────────────────────────────────
# Load weights
# ─────────────────────────────────────────────
datasets = {}
for label, path in WEIGHT_FILES.items():
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping {label}")
        continue
    print(f"Loading {label} from {path}...")
    with open(path) as f:
        d = json.load(f)

    w = np.array(d['w_ij'])          # (N, N, num_bins)
    bins = d['bins']
    tau_idx = [bins.index(t) for t in LAGS]
    w = w[:, :, tau_idx]             # (N, N, 5)

    N = w.shape[0]
    # mask diagonal (self-pairs)
    mask = ~np.eye(N, dtype=bool)

    datasets[label] = {
        'w':     w,           # (N, N, 5)
        'mask':  mask,
        'N':     N,
        'ids':   d['neuron_ids'],
    }
    print(f"  shape: {w.shape}  off-diag pairs: {mask.sum()}")

labels = list(datasets.keys())

# ─────────────────────────────────────────────
# Build figure
# ─────────────────────────────────────────────
# Layout:
#   row 1, col 1      : overall histogram (both datasets overlaid)
#   row 2, cols 1-5   : per-lag histograms (one col per tau, both datasets)
#   row 3, cols 1-2   : peak-weight heatmaps
#   row 4, col 1      : scatter Exp2 vs Between_Burst peak weights

n_heatmap_cols = len(labels)
specs = (
    [[{'colspan': 5}, None, None, None, None]] +   # row 1: overall hist
    [[{}] * 5] +                                    # row 2: per-lag hists
    [[{}] * n_heatmap_cols + [None] * (5 - n_heatmap_cols)] +   # row 3: heatmaps
    [[{'colspan': 3}, None, None, None, None]]      # row 4: scatter
)

fig = make_subplots(
    rows=4, cols=5,
    specs=specs,
    subplot_titles=(
        ['All weights (off-diagonal)', None, None, None, None] +
        [f'τ = {t} ms' for t in LAGS] +
        [f'Peak-weight heatmap — {lb}' for lb in labels] + [None] * (5 - n_heatmap_cols) +
        ['Peak weight: Exp2 vs Between_Burst', None, None, None, None]
    ),
    vertical_spacing=0.09,
    horizontal_spacing=0.04,
)

# ── Row 1: overall histogram ──────────────────
for label in labels:
    d = datasets[label]
    w_flat = d['w'][d['mask']].flatten()   # all off-diag × all lags
    fig.add_trace(go.Histogram(
        x=w_flat,
        nbinsx=120,
        name=label,
        legendgroup=label,
        marker_color=COLORS[label],
        opacity=0.65,
        hovertemplate=f'<b>{label}</b><br>w = %{{x:.3f}}<br>count = %{{y}}<extra></extra>',
    ), row=1, col=1)

fig.update_xaxes(title_text='weight w_ij', row=1, col=1)
fig.update_yaxes(title_text='count', row=1, col=1)

# ── Row 2: per-lag histograms ─────────────────
for tau_i, tau in enumerate(LAGS):
    for label in labels:
        d = datasets[label]
        w_lag = d['w'][:, :, tau_i][d['mask']]
        fig.add_trace(go.Histogram(
            x=w_lag,
            nbinsx=80,
            name=label,
            legendgroup=label,
            showlegend=False,
            marker_color=COLORS[label],
            opacity=0.65,
            hovertemplate=f'<b>{label} τ={tau}ms</b><br>w = %{{x:.3f}}<br>count = %{{y}}<extra></extra>',
        ), row=2, col=tau_i + 1)
    fig.update_xaxes(title_text='w_ij', row=2, col=tau_i + 1)

# ── Row 3: peak-weight heatmaps ───────────────
for col_i, label in enumerate(labels):
    d = datasets[label]
    w_peak = d['w'].max(axis=2)           # (N, N) — peak excitatory weight per pair
    np.fill_diagonal(w_peak, np.nan)

    fig.add_trace(go.Heatmap(
        z=w_peak,
        colorscale='RdBu_r',
        zmid=0,
        text=[[f'N{d["ids"][i]}→N{d["ids"][j]}<br>w={w_peak[i,j]:.3f}'
               for j in range(d['N'])] for i in range(d['N'])],
        hovertemplate='%{text}<extra></extra>',
        colorbar=dict(title='peak w', len=0.22, y=0.27,
                      yanchor='middle', x=0.42 + col_i * 0.55),
        showscale=True,
        name=label,
        legendgroup=label,
        showlegend=False,
    ), row=3, col=col_i + 1)
    fig.update_xaxes(title_text='target neuron j', row=3, col=col_i + 1)
    fig.update_yaxes(title_text='source neuron i', row=3, col=col_i + 1)

# ── Row 4: scatter Exp2 vs Between_Burst ─────
if 'Exp2' in datasets and 'Between_Burst' in datasets:
    dA = datasets['Exp2']
    dB = datasets['Between_Burst']
    mask = dA['mask']

    # Peak weight per pair (max over lags)
    wA_peak = dA['w'].max(axis=2)[mask]
    wB_peak = dB['w'].max(axis=2)[mask]

    # Colour by difference
    diff = wA_peak - wB_peak

    fig.add_trace(go.Scattergl(
        x=wA_peak,
        y=wB_peak,
        mode='markers',
        marker=dict(
            size=3,
            color=diff,
            colorscale='RdBu_r',
            cmid=0,
            opacity=0.5,
            colorbar=dict(title='Exp2 − Between', len=0.22, y=0.03,
                          yanchor='bottom', x=0.58),
            showscale=True,
        ),
        hovertemplate='Exp2: %{x:.3f}<br>Between: %{y:.3f}<extra></extra>',
        name='pairs',
        showlegend=False,
    ), row=4, col=1)

    # Diagonal reference
    w_lim = max(abs(wA_peak).max(), abs(wB_peak).max()) * 1.05
    fig.add_trace(go.Scatter(
        x=[-w_lim, w_lim], y=[-w_lim, w_lim],
        mode='lines', line=dict(color='gray', dash='dash', width=1),
        showlegend=False,
    ), row=4, col=1)

    fig.update_xaxes(title_text='Exp2 peak weight', row=4, col=1)
    fig.update_yaxes(title_text='Between_Burst peak weight', row=4, col=1)

# ── Legend & layout ───────────────────────────
fig.update_layout(
    title=dict(
        text='Weight distribution — Exp2 vs Between_Burst<br>'
             '<sup>w_ij(τ) = logit(p_ij(τ)) − logit(B_j) · off-diagonal pairs only</sup>',
        font=dict(size=14), x=0.0, xanchor='left',
    ),
    barmode='overlay',
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=1600,
    width=1400,
    legend=dict(font=dict(size=11), x=0.75, y=0.98,
                itemclick='toggle', itemdoubleclick='toggleothers'),
)

out = HERE / 'weight_distributions.html'
fig.write_html(out, include_plotlyjs='cdn')
print(f"Saved: {out}")
print("Open in any browser.")
