import json
import numpy as np
import pandas as pd
from scipy.special import expit
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CONFIG_ID  = 1
SEED       = 42
SPLIT      = 0.9
ACT_BIN_MS = 50     # ms window for population activity trace
BETA           = 20  # scaling parameter in σ_i(u) = 1 / (1 + exp(-(BETA*u - m_i)))
BASELINE_SCALE = 5   # baseline firing prob = B_i / BASELINE_SCALE  (σ_i(0) = B_i/k)
SHIFT_MS   = 50_000 # circular shift applied to Group 1 in the null model (ms)

# ─────────────────────────────────────────────
# 1. Load Config data → binary spike matrix
# ─────────────────────────────────────────────
print("Loading neuron data...")
with open('../neuron_data.json') as f:
    neuron_data = json.load(f)

all_units = [neuron[0]['unit'] for neuron in neuron_data if neuron]
N = len(all_units)
unit_idx = {u: i for i, u in enumerate(all_units)}
print(f"  {N} neurons")

print(f"Building binary matrix for Config {CONFIG_ID}...")
config_spikes = {}
for neuron in neuron_data:
    if not neuron:
        continue
    unit = neuron[0]['unit']
    spikes = np.array([s['spikeTime'] for s in neuron if s['Config'] == CONFIG_ID])
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
bi_df = pd.read_csv('../Configurations/config_B_i.csv')
bi_cfg = bi_df[bi_df['config'] == CONFIG_ID].set_index('unit')['B_i']
B = np.array([float(bi_cfg.get(u, 1e-10)) for u in all_units])
B = np.clip(B, 1e-10, 1.0 - 1e-10)
m = np.log(BASELINE_SCALE / B - 1.0)   # σ_i(0) = B_i / BASELINE_SCALE
print(f"  B range: {B.min():.2e} – {B.max():.2e}")

# ─────────────────────────────────────────────
# 3. Weight matrix
# ─────────────────────────────────────────────
print("Loading weights...")
W_df = pd.read_csv('weights_wide.csv', index_col=0)
W_df.index   = W_df.index.astype(int)
W_df.columns = W_df.columns.astype(int)
W = W_df.reindex(index=all_units, columns=all_units).fillna(0.0).values  # (N, N)
print(f"  W shape: {W.shape}, non-zero: {(W != 0).sum()}")

# ─────────────────────────────────────────────
# 4. Random 50/50 split
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

W_sub = W[group2_idx, :]   # (G2, N)
m_sub = m[group2_idx]      # (G2,)

# ─────────────────────────────────────────────
# 5. Model runner
#    x_i(t+1) = Bernoulli(σ_i(Σ_j w_ij x_j(t)))
#    σ_i(u)   = 1 / (1 + exp(-(BETA*u - m_i)))
#    Group 1 input is taken from X_g1_input (real or shifted).
#    Both model and null use the same SEED so random draws are identical —
#    the only difference is which Group 1 spike train drives them.
# ─────────────────────────────────────────────
def run_model(X_g1_input, label):
    X_out  = X_real.copy()
    X_out[group1_idx, :] = X_g1_input     # overwrite Group 1, shape (G1, T)
    x_cur  = X_out[:, 0].astype(np.float64)
    rng_m  = np.random.default_rng(SEED + 1)   # fresh, same seed for fair comparison

    x_prev = np.zeros(N, dtype=np.float64)   # x(t-1), zero before recording starts

    print(f"Running {label}...")
    for t in range(T - 1):
        u   = W_sub @ (x_cur + x_prev)
        p   = expit(BETA * u - m_sub)
        new = (rng_m.random(G2) < p).astype(np.uint8)
        X_out[g2_arr, t + 1] = new
        x_prev               = x_cur.copy()
        x_cur[g2_arr]        = new.astype(np.float64)
        x_cur[group1_idx]    = X_g1_input[:, t + 1].astype(np.float64)
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
    (act_null,  'mediumpurple', f'Activity – null (+{SHIFT_MS//1000}s shift)'),
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

out = f'model_raster_config{CONFIG_ID}.html'
fig.write_html(out, include_plotlyjs='cdn')
print(f"Saved: {out}")
print("Open in any browser — zoom/pan links both panels.")
