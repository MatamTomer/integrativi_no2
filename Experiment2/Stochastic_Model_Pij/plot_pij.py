"""
plot_pij.py
Visualise the p_ij distributions from Spike_Triggered_Average/p_ij.json files.

p_ij[i][j][k] = P(neuron j fires at lag bins[k] ms | neuron i spiked)
              = C_ij(tau) / N_i

Panels:
  Row 1 — Histogram: all off-diagonal p_ij, Exp2 vs Between_Burst (overlaid)
  Row 2 — Histogram per lag (tau = 1..5 ms) for each dataset
  Row 3 — Heatmap: peak p_ij matrix (max over lags) for each dataset
  Row 4 — Scatter: Exp2 peak p_ij vs Between_Burst peak p_ij per pair

Output: pij_distributions.html
"""

import json
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE        = Path(__file__).parent
EXPERIMENT2 = HERE.parent

PIJ_FILES = {
    'Exp2':          EXPERIMENT2 / 'Spike_Triggered_Average' / 'p_ij.json',
    'Between_Burst': EXPERIMENT2 / 'Between_Burst' / 'Spike_Triggered_Average' / 'p_ij.json',
}

COLORS = {
    'Exp2':          '#e6851e',
    'Between_Burst': '#3d9db3',
}

LAGS = [1, 2, 3, 4, 5]

# ─────────────────────────────────────────────
# Load p_ij
# ─────────────────────────────────────────────
datasets = {}
for label, path in PIJ_FILES.items():
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping {label}")
        continue
    print(f"Loading {label} from {path}...")
    with open(path) as f:
        d = json.load(f)

    pij  = np.array(d['p_ij'])      # (N, N, num_bins)
    bins = d['bins']
    tau_idx = [bins.index(t) for t in LAGS]
    pij = pij[:, :, tau_idx]        # (N, N, 5)

    N = pij.shape[0]
    mask = ~np.eye(N, dtype=bool)   # exclude self-pairs

    datasets[label] = {
        'pij':  pij,
        'mask': mask,
        'N':    N,
        'ids':  d['neuron_ids'],
    }
    print(f"  shape: {pij.shape}  max p_ij: {pij[mask].max():.4f}  "
          f"fraction zero: {(pij[mask] == 0).mean():.1%}")

labels = list(datasets.keys())

# ─────────────────────────────────────────────
# Build figure
# ─────────────────────────────────────────────
n_heatmap_cols = len(labels)
specs = (
    [[{'colspan': 5}, None, None, None, None]] +
    [[{}] * 5] +
    [[{}] * n_heatmap_cols + [None] * (5 - n_heatmap_cols)] +
    [[{'colspan': 3}, None, None, None, None]]
)

fig = make_subplots(
    rows=4, cols=5,
    specs=specs,
    subplot_titles=(
        ['All p_ij (off-diagonal, τ=1–5ms)', None, None, None, None] +
        [f'τ = {t} ms' for t in LAGS] +
        [f'Peak p_ij heatmap — {lb}' for lb in labels] + [None] * (5 - n_heatmap_cols) +
        ['Peak p_ij: Exp2 vs Between_Burst', None, None, None, None]
    ),
    vertical_spacing=0.09,
    horizontal_spacing=0.04,
)

# ── Row 1: overall histogram ──────────────────
for label in labels:
    d = datasets[label]
    p_flat = d['pij'][d['mask']].flatten()
    # clip extreme outliers for display (p_ij can occasionally be >1)
    p_flat = p_flat[p_flat <= np.percentile(p_flat, 99.5)]
    fig.add_trace(go.Histogram(
        x=p_flat,
        nbinsx=120,
        name=label,
        legendgroup=label,
        marker_color=COLORS[label],
        opacity=0.65,
        hovertemplate=f'<b>{label}</b><br>p_ij = %{{x:.4f}}<br>count = %{{y}}<extra></extra>',
    ), row=1, col=1)

fig.update_xaxes(title_text='p_ij', row=1, col=1)
fig.update_yaxes(title_text='count', row=1, col=1)

# ── Row 2: per-lag histograms ─────────────────
for tau_i, tau in enumerate(LAGS):
    for label in labels:
        d = datasets[label]
        p_lag = d['pij'][:, :, tau_i][d['mask']]
        p_lag = p_lag[p_lag <= np.percentile(p_lag, 99.5)]
        fig.add_trace(go.Histogram(
            x=p_lag,
            nbinsx=80,
            name=label,
            legendgroup=label,
            showlegend=False,
            marker_color=COLORS[label],
            opacity=0.65,
            hovertemplate=f'<b>{label} τ={tau}ms</b><br>p_ij=%{{x:.4f}}<br>count=%{{y}}<extra></extra>',
        ), row=2, col=tau_i + 1)
    fig.update_xaxes(title_text='p_ij', row=2, col=tau_i + 1)

# ── Row 3: peak p_ij heatmaps ────────────────
for col_i, label in enumerate(labels):
    d = datasets[label]
    p_peak = d['pij'].max(axis=2)    # (N, N)
    np.fill_diagonal(p_peak, np.nan)

    fig.add_trace(go.Heatmap(
        z=p_peak,
        colorscale='Viridis',
        text=[[f'N{d["ids"][i]}→N{d["ids"][j]}<br>p={p_peak[i,j]:.4f}'
               for j in range(d['N'])] for i in range(d['N'])],
        hovertemplate='%{text}<extra></extra>',
        colorbar=dict(title='peak p_ij', len=0.22, y=0.27,
                      yanchor='middle', x=0.42 + col_i * 0.55),
        showscale=True,
        showlegend=False,
    ), row=3, col=col_i + 1)
    fig.update_xaxes(title_text='target neuron j', row=3, col=col_i + 1)
    fig.update_yaxes(title_text='source neuron i', row=3, col=col_i + 1)

# ── Row 4: scatter Exp2 vs Between_Burst ─────
if 'Exp2' in datasets and 'Between_Burst' in datasets:
    dA = datasets['Exp2']
    dB = datasets['Between_Burst']
    mask = dA['mask']

    pA = dA['pij'].max(axis=2)[mask]
    pB = dB['pij'].max(axis=2)[mask]
    diff = pA - pB

    fig.add_trace(go.Scattergl(
        x=pA, y=pB,
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
        hovertemplate='Exp2: %{x:.4f}<br>Between: %{y:.4f}<extra></extra>',
        showlegend=False,
    ), row=4, col=1)

    # Diagonal reference
    lim = max(pA.max(), pB.max()) * 1.05
    fig.add_trace(go.Scatter(
        x=[0, lim], y=[0, lim],
        mode='lines', line=dict(color='gray', dash='dash', width=1),
        showlegend=False,
    ), row=4, col=1)

    # Baseline B_j reference line (horizontal) — average B across neurons
    import pandas as pd
    bi_df = pd.read_csv(EXPERIMENT2 / 'Configurations' / 'config_B_i.csv')
    B_mean = bi_df['B_i'].mean()
    fig.add_trace(go.Scatter(
        x=[0, lim], y=[B_mean, B_mean],
        mode='lines', line=dict(color='crimson', dash='dot', width=1),
        name=f'mean B_i = {B_mean:.4f}',
        showlegend=True,
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=[B_mean, B_mean], y=[0, lim],
        mode='lines', line=dict(color='crimson', dash='dot', width=1),
        showlegend=False,
    ), row=4, col=1)

    fig.update_xaxes(title_text='Exp2 peak p_ij', range=[0, lim], row=4, col=1)
    fig.update_yaxes(title_text='Between_Burst peak p_ij', range=[0, lim], row=4, col=1)

# ── Layout ────────────────────────────────────
fig.update_layout(
    title=dict(
        text='p_ij distribution — Exp2 vs Between_Burst<br>'
             '<sup>p_ij[i][j][τ] = P(j fires at lag τ ms | i spiked) · off-diagonal pairs only · τ = 1–5 ms</sup>',
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

out = HERE / 'pij_distributions.html'
fig.write_html(out, include_plotlyjs='cdn')
print(f"Saved: {out}")
print("Open in any browser.")
