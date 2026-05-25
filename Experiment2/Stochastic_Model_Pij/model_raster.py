import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit
import plotly.graph_objects as go
from plotly.subplots import make_subplots

EXPERIMENT2 = Path(__file__).parent.parent

CONFIG_ID  = 2
SEED       = 42
SPLIT      = 0.9
ACT_BIN_MS = 50     # ms window for population activity trace
BETA           = 10  # scaling parameter in σ_i(u) = 1 / (1 + exp(-(BETA*u - m_i)))
BASELINE_SCALE = 5   # baseline firing prob = B_i / BASELINE_SCALE  (σ_i(0) = B_i/k)

# ─────────────────────────────────────────────
# 1. Load Config data → binary spike matrix
# ─────────────────────────────────────────────
print("Loading neuron data...")
with open(EXPERIMENT2 / 'neuron_data.json') as f:
    neuron_data = json.load(f)

all_units = [neuron[0]['unit'] for neuron in neuron_data if neuron]
N = len(all_units)
unit_idx = {u: i for i, u in enumerate(all_units)}
print(f"  {N} neurons")

# Load config time bounds from Configurations/config_times.csv
cfg_times = pd.read_csv(EXPERIMENT2 / 'Configurations' / 'config_times.csv')
cfg_row   = cfg_times[cfg_times['config'] == CONFIG_ID].iloc[0]
t_start_cfg = float(cfg_row['t_start'])
t_end_cfg   = float(cfg_row['t_end'])
print(f"  Config {CONFIG_ID}: {t_start_cfg:.3f}s – {t_end_cfg:.3f}s")

print(f"Building binary matrix for Config {CONFIG_ID}...")
config_spikes = {}
for neuron in neuron_data:
    if not neuron:
        continue
    unit = neuron[0]['unit']
    all_spikes = np.array([s['spikeTime'] for s in neuron], dtype=np.float64)
    spikes = all_spikes[(all_spikes >= t_start_cfg) & (all_spikes <= t_end_cfg)]
    config_spikes[unit] = spikes

all_ms = np.concatenate([
    (spikes * 1000).astype(int)
    for spikes in config_spikes.values()
    if len(spikes) > 0
])
t_start_ms = int(all_ms.min())
t_end_ms   = int(all_ms.max())
T          = t_end_ms - t_start_ms + 1
t_start_s  = t_start_ms / 1000.0
print(f"  T = {T:,} ms bins  ({t_start_s:.1f}s – {t_end_ms/1000:.1f}s)")

X_real = np.zeros((N, T), dtype=np.uint8)
for unit, spikes in config_spikes.items():
    if len(spikes) == 0:
        continue
    bins = (spikes * 1000).astype(int) - t_start_ms
    bins = bins[(bins >= 0) & (bins < T)]
    X_real[unit_idx[unit], bins] = 1

print(f"  {X_real.sum():,} spikes loaded")

# ─────────────────────────────────────────────
# 2. B_i for this config → m_i
# ─────────────────────────────────────────────
print("Loading B_i...")
bi_df = pd.read_csv(EXPERIMENT2 / 'Configurations' / 'config_B_i.csv')
bi_cfg = bi_df[bi_df['config'] == CONFIG_ID].set_index('unit')['B_i']
B = np.array([float(bi_cfg.get(u, 1e-10)) for u in all_units])
B = np.clip(B, 1e-10, 1.0 - 1e-10)
m = np.log(BASELINE_SCALE / B - 1.0)   # σ_i(0) = B_i / BASELINE_SCALE
print(f"  B range: {B.min():.2e} – {B.max():.2e}")

# ─────────────────────────────────────────────
# 3. Weight tensor — w_ij(tau) = P_ij(tau) - B_j, tau = 1..5ms
# ─────────────────────────────────────────────
print("Loading weights...")
with open(Path(__file__).parent / 'weights_pij_all.json') as f:
    w_data = json.load(f)

w_full     = np.array(w_data['w_ij'])   # (N, N, 21)
bins_w     = w_data['bins']             # [-10..10]
# Extract tau = 1..5 ms — indices where bins == 1,2,3,4,5
tau_indices = [bins_w.index(tau) for tau in [1, 2, 3, 4, 5]]
TAU_MAX     = len(tau_indices)          # 5

# Align to all_units order
w_ids  = w_data['neuron_ids']
id2row = {nid: idx for idx, nid in enumerate(w_ids)}
order  = [id2row[u] for u in all_units]
W_causal = w_full[np.ix_(order, order, tau_indices)]  # (N, N, 5)
# W_causal[i, j, k] = w_ij at lag (k+1) ms
print(f"  W_causal shape: {W_causal.shape}, non-zero: {(W_causal != 0).sum()}")

# ─────────────────────────────────────────────
# 4. Random split
# ─────────────────────────────────────────────
rng  = np.random.default_rng(SEED)
perm = rng.permutation(N)
split_n    = int(N * SPLIT)
group1_idx = sorted(perm[:split_n].tolist())
group2_idx = sorted(perm[split_n:].tolist())
G1 = len(group1_idx)
G2 = len(group2_idx)
g2_arr = np.array(group2_idx)
print(f"  Group 1 (real):      {G1} neurons")
print(f"  Group 2 (simulated): {G2} neurons")

# W_sub_causal[g, i, k] = w_{all_units[i] -> group2[g]} at lag (k+1) ms
W_sub_causal = W_causal[group2_idx, :, :]   # (G2, N, 5)
m_sub        = m[group2_idx]                # (G2,)

# ─────────────────────────────────────────────
# 5. Model runner
#    x_j(t) = Bernoulli(σ_j(Σ_{i≠j} Σ_{tau=1}^{5} w_ij[tau] * x_i(t-tau)))
#    σ_j(u) = 1 / (1 + exp(-(BETA*u - m_j)))
#    History buffer x_hist: shape (N, 5), columns = x(t-1)..x(t-5)
# ─────────────────────────────────────────────
def run_model(X_g1_input, label):
    X_out  = X_real.copy()
    X_out[group1_idx, :] = X_g1_input        # fix Group 1 spikes
    rng_m  = np.random.default_rng(SEED + 1)

    # History buffer: columns 0..4 = x(t-1)..x(t-5), initialised to zero
    x_hist = np.zeros((N, TAU_MAX), dtype=np.float64)

    print(f"Running {label}...")
    for t in range(T - 1):
        # u_j = Σ_i Σ_{tau=1}^{5} W_sub_causal[j, i, tau-1] * x_hist[i, tau-1]
        # einsum: (G2, N, 5) x (N, 5) -> (G2,)
        u   = np.einsum('jit,it->j', W_sub_causal, x_hist)
        p   = expit(BETA * u - m_sub)
        new = (rng_m.random(G2) < p).astype(np.uint8)
        X_out[g2_arr, t + 1] = new

        # Build x(t) for all neurons — used to update history
        x_t                  = X_out[:, t + 1].astype(np.float64)
        x_t[group1_idx]      = X_g1_input[:, t + 1].astype(np.float64)

        # Shift history: drop oldest (col 4), prepend x(t) as col 0
        x_hist = np.roll(x_hist, 1, axis=1)
        x_hist[:, 0] = x_t

        if (t + 1) % 100_000 == 0:
            print(f"  t = {t+1:,} / {T:,}")

    print(f"  {label} done.")
    return X_out

# Real Group 1 input (G1, T)
g1_real_input = X_real[group1_idx, :]                             # (G1, T)

# Null Group 1 input: shuffle neuron identities within Group 1
# spike trains are real but who-drives-whom is randomized
shuffled_order = np.random.default_rng(SEED + 99).permutation(G1)
g1_null_input  = g1_real_input[shuffled_order, :]                 # (G1, T)

X_model = run_model(g1_real_input, 'model (real G1 input)')
X_null  = run_model(g1_null_input, 'null  (G1 identities shuffled)')

# ─────────────────────────────────────────────
# 6. Spike times for Plotly traces
# ─────────────────────────────────────────────
t_axis = t_start_s + np.arange(T) * 0.001

def spike_times(X, indices):
    return {i: t_axis[np.where(X[i])[0]] for i in indices}

g1_real_st  = spike_times(X_real,  group1_idx)
g2_real_st  = spike_times(X_real,  group2_idx)
g2_model_st = spike_times(X_model, group2_idx)
g2_null_st  = spike_times(X_null,  group2_idx)

# Population activity (ACT_BIN_MS bins)
T2    = (T // ACT_BIN_MS) * ACT_BIN_MS
t_act = t_start_s + (np.arange(T2 // ACT_BIN_MS) * ACT_BIN_MS + ACT_BIN_MS / 2) * 0.001
act_real  = X_real [:, :T2].reshape(N, -1, ACT_BIN_MS).sum(axis=(0, 2))
act_model = X_model[:, :T2].reshape(N, -1, ACT_BIN_MS).sum(axis=(0, 2))
act_null  = X_null [:, :T2].reshape(N, -1, ACT_BIN_MS).sum(axis=(0, 2))

# ─────────────────────────────────────────────
# 7. Build interactive raster
#    Section layout (y-axis):
#      Group 1 real      (steelblue)
#      Group 2 real      (crimson)      ← ground truth
#      Group 2 model     (darkorange)   ← real G1 input
#      Group 2 null      (mediumpurple) ← shifted G1 input
# ─────────────────────────────────────────────
print("Building plot...")

GAP     = 4
y0_g1   = 0
y0_g2r  = G1  + GAP
y0_g2m  = y0_g2r + G2 + GAP
y0_g2n  = y0_g2m + G2 + GAP
total_y = y0_g2n + G2

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.80, 0.20],
    vertical_spacing=0.03,
)

def add_raster_section(indices, y0, times_dict, color, legend_name, legend_group):
    for rank, idx in enumerate(indices):
        nid   = all_units[idx]
        times = times_dict[idx]
        fig.add_trace(go.Scattergl(
            x=times, y=np.full(len(times), y0 + rank),
            mode='markers',
            marker=dict(symbol='line-ns', size=3, color=color,
                        line=dict(width=1, color=color)),
            name=legend_name, legendgroup=legend_group,
            showlegend=(rank == 0),
            hovertemplate=f'<b>{legend_name} N{nid}</b><br>%{{x:.3f}} s<extra></extra>',
        ), row=1, col=1)

add_raster_section(group1_idx, y0_g1,  g1_real_st,  'steelblue',    'Group 1 (real)',    'g1')
add_raster_section(group2_idx, y0_g2r, g2_real_st,  'crimson',      'Group 2 (real)',    'g2r')
add_raster_section(group2_idx, y0_g2m, g2_model_st, 'darkorange',   'Group 2 (model)',   'g2m')
add_raster_section(group2_idx, y0_g2n, g2_null_st,  'mediumpurple', 'Group 2 (null)',    'g2n')

# ── Separator lines ──
t_min = t_start_s
t_max = t_start_s + T * 0.001
for y_sep in [y0_g2r - GAP/2, y0_g2m - GAP/2, y0_g2n - GAP/2]:
    fig.add_shape(type='line',
                  x0=t_min, x1=t_max, y0=y_sep, y1=y_sep,
                  xref='x', yref='y',
                  line=dict(color='rgba(120,120,120,0.6)', width=1.2, dash='dot'))

# ── Section annotations (right side) ──
section_labels = [
    (y0_g1  + G1/2, 'steelblue',    'GROUP 1',
     f'{G1} neurons — original spikes kept unchanged.<br>'
     'Fed as fixed input into both model and null at every time step.'),
    (y0_g2r + G2/2, 'crimson',      'GROUP 2 — Real',
     f'{G2} neurons — original spikes (ground truth).<br>'
     'Erased before running either model.'),
    (y0_g2m + G2/2, 'darkorange',   'GROUP 2 — Model',
     f'Stochastic model with <b>real</b> Group 1 input.<br>'
     'x<sub>i</sub>(t+1) ~ Bernoulli(σ<sub>i</sub>(Σ w<sub>ij</sub> x<sub>j</sub>(t)))<br>'
     f'σ<sub>i</sub>(u) = 1/(1+exp(−(β·u−m<sub>i</sub>))),  β={BETA},  '
     f'm<sub>i</sub>=ln({BASELINE_SCALE}/B<sub>i</sub>−1)  →  σ<sub>i</sub>(0)=B<sub>i</sub>/{BASELINE_SCALE}'),
    (y0_g2n + G2/2, 'mediumpurple', 'GROUP 2 — Null',
     'Same model with Group 1 neuron <b>identities shuffled</b>.<br>'
     'Spike trains are real but who-drives-whom is randomized.<br>'
     'Same weights, same β, same B<sub>i</sub> — only wiring identity broken.'),
]

for y_c, color, bold, sub in section_labels:
    fig.add_annotation(
        x=1.01, y=y_c,
        xref='paper', yref='y',
        text=(f'<b><span style="color:{color}">{bold}</span></b><br>'
              f'<span style="color:gray;font-size:9px">{sub}</span>'),
        xanchor='left', yanchor='middle',
        showarrow=False, font=dict(size=11), align='left',
    )

# ── Population activity ──
for y_act, color, name in [
    (act_real,  'steelblue',    'Activity – real'),
    (act_model, 'darkorange',   'Activity – model'),
    (act_null,  'mediumpurple', 'Activity – null'),
]:
    fig.add_trace(go.Scattergl(
        x=t_act, y=y_act,
        mode='lines', line=dict(color=color, width=1.2),
        fill='tozeroy',
        fillcolor=color.replace('steelblue',    'rgba(70,130,180,0.12)')
                       .replace('darkorange',   'rgba(255,140,0,0.12)')
                       .replace('mediumpurple', 'rgba(147,112,219,0.12)'),
        name=name,
        hovertemplate='Time: %{x:.3f}s<br>%{y:.0f} spikes/50ms<extra></extra>',
    ), row=2, col=1)

# ── Layout ──
tick_vals  = (list(range(y0_g1,  y0_g1  + G1)) +
              list(range(y0_g2r, y0_g2r + G2)) +
              list(range(y0_g2m, y0_g2m + G2)) +
              list(range(y0_g2n, y0_g2n + G2)))
tick_texts = ([f'N{all_units[i]}' for i in group1_idx] +
              [f'N{all_units[i]}' for i in group2_idx] * 3)

fig.update_layout(
    title=dict(
        text=(f'Stochastic network model — Config {CONFIG_ID}<br>'
              f'<sup>50/50 random split. Group 2 ground truth shown for comparison. '
              'Null model uses Group 1 with shuffled neuron identities — '
              'same spike trains, same weights and β, broken specific wiring.</sup>'),
        font=dict(size=14), x=0.0, xanchor='left',
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=1100,
    width=1600,
    margin=dict(r=420),
    legend=dict(font=dict(size=10), itemclick='toggle',
                itemdoubleclick='toggleothers'),
    hovermode='x',
)

fig.update_yaxes(
    tickvals=tick_vals, ticktext=tick_texts,
    showgrid=False, fixedrange=True,
    range=[-1, total_y + 1],
    tickfont=dict(size=5),
    title_text='Neuron',
    row=1, col=1,
)
fig.update_yaxes(
    showgrid=True, gridcolor='rgba(200,200,200,0.4)',
    fixedrange=True, title_text=f'Spikes / {ACT_BIN_MS}ms',
    row=2, col=1,
)
fig.update_xaxes(
    title_text='Time (s)',
    showgrid=True, gridcolor='rgba(200,200,200,0.3)',
    row=2, col=1,
)

out = Path(__file__).parent / f'model_raster_config{CONFIG_ID}.html'
fig.write_html(out, include_plotlyjs='cdn')
print(f"Saved: {out}")
print("Open in any browser — zoom/pan links both panels.")
