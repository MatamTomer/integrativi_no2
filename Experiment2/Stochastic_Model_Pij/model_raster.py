import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, NullFormatter

# Journal-figure-scale fonts for the matplotlib stats-comparison figure
plt.rcParams.update({
    'font.size':        9,
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'legend.fontsize':  8,
})

EXPERIMENT2 = Path(__file__).parent.parent

CONFIG_ID      = 2
SEED           = 42
SPLIT          = 0.9
ACT_BIN_MS     = 10     # ms window for population activity trace
GAMMA_DELTA_MS = 5      # ms coincidence window for spike-timing reliability (Γ)
BETA           = 20   # scaling factor r in σ_j(u) = 1 / (1 + exp(-(r*u - m_j)))
BASELINE_SCALE = 5      # baseline firing prob = B_i / BASELINE_SCALE  (σ_i(0) = B_i/k)
WEIGHT_TOP_PCT = 10     # keep only the top N% of weights; set to None to use all

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

# Load config time bounds
cfg_times   = pd.read_csv(EXPERIMENT2 / 'Configurations' / 'config_times.csv')
cfg_row     = cfg_times[cfg_times['config'] == CONFIG_ID].iloc[0]
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
bi_df  = pd.read_csv(EXPERIMENT2 / 'Configurations' / 'config_B_i.csv')
bi_cfg = bi_df[bi_df['config'] == CONFIG_ID].set_index('unit')['B_i']
B = np.array([float(bi_cfg.get(u, 1e-10)) for u in all_units])
B = np.clip(B, 1e-10, 1.0 - 1e-10)
m = np.log(BASELINE_SCALE / B - 1.0)   # σ_i(0) = B_i / BASELINE_SCALE
print(f"  B range: {B.min():.2e} – {B.max():.2e}")

# ─────────────────────────────────────────────
# 3. Load weight tensors
#    Three sources: Experiment2 (all), Between_Burst, Within_Burst
# ─────────────────────────────────────────────
def load_weights(weights_path, label):
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found for {label}: {weights_path}\n"
            f"Run compute_weights_pij.py in that folder first."
        )
    print(f"Loading {label} weights from {weights_path}...")
    with open(weights_path) as f:
        w_data = json.load(f)

    w_full  = np.array(w_data['w_ij'])   # (N, N, 21)
    bins_w  = w_data['bins']             # [-10..10]
    tau_indices = [bins_w.index(tau) for tau in [1, 2, 3, 4, 5]]

    # Align to all_units order
    w_ids  = w_data['neuron_ids']
    id2row = {nid: idx for idx, nid in enumerate(w_ids)}
    order  = [id2row[u] for u in all_units]
    W = w_full[np.ix_(order, order, tau_indices)]   # (N, N, 5)
    print(f"  shape: {W.shape}, non-zero: {(W != 0).sum()}")
    return W

TAU_MAX = 5

def apply_weight_threshold(W, label):
    """Zero out all weights below the (100 - WEIGHT_TOP_PCT)th percentile."""
    if WEIGHT_TOP_PCT is None:
        return W
    N = W.shape[0]
    mask = ~np.eye(N, dtype=bool)
    # compute threshold on off-diagonal values across all lags
    off_diag = W[mask].flatten()
    thresh = np.percentile(off_diag, 100 - WEIGHT_TOP_PCT)
    W_thresh = np.where(W >= thresh, W, 0.0)
    kept = int((W_thresh[mask] != 0).sum())
    total = int(mask.sum() * W.shape[2])
    print(f"  [{label}] threshold={thresh:.3f}  kept {kept}/{total} weights "
          f"({100*kept/total:.1f}%) — top {WEIGHT_TOP_PCT}%")
    return W_thresh

W_exp2    = load_weights(Path(__file__).parent / 'weights_pij_excess_per_config_1.json',                              'Experiment2 (excess, per_config_1)')
W_between = load_weights(EXPERIMENT2 / 'Between_Burst' / 'Stochastic_Model_Pij' / 'weights_pij_excess_all.json', 'Between_Burst (excess)')

W_exp2    = apply_weight_threshold(W_exp2,    'Exp2')
W_between = apply_weight_threshold(W_between, 'Between_Burst')

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

m_sub = m[group2_idx]   # (G2,) — shared across all model runs

# ─────────────────────────────────────────────
# 5. Model runner
#    Accepts W_sub_causal as parameter so any weight set can be used.
#    x_j(t) = Bernoulli(σ_j(Σ_{i≠j} Σ_{tau=1}^{5} w_ij[tau] * x_i(t-tau)))
#    σ_j(u) = 1 / (1 + exp(-(BETA*u - m_j)))
# ─────────────────────────────────────────────
def run_model(X_g1_input, W_full, label, run_seed=None):
    """
    X_g1_input : (G1, T) spike matrix for Group 1 (fixed input)
    W_full     : (N, N, 5) weight tensor — will be sliced to Group 2 rows
    label      : string for progress printing
    run_seed   : seed for this run's RNG (must differ across runs!)
    """
    # Feedforward only: G1→G2 weights, no G2→G2 recurrence
    W_sub = W_full[group1_idx, :, :][:, group2_idx, :]  # (G1, G2, 5)

    X_out = X_real.copy()
    X_out[group1_idx, :] = X_g1_input
    rng_m = np.random.default_rng(run_seed)

    x_hist = np.zeros((G1, TAU_MAX), dtype=np.float64)  # only G1 history needed

    print(f"Running {label}...")
    for t in range(T - 1):
        u   = np.einsum('ijt,it->j', W_sub, x_hist)
        n_active = (x_hist > 0).any(axis=1).sum()
        u  /= max(1.0, np.sqrt(n_active))
        p   = expit(BETA * u - m_sub)
        new = (rng_m.random(G2) < p).astype(np.uint8)
        X_out[g2_arr, t + 1] = new

        # Advance G1 history only (G2 spikes never feed back)
        x_hist = np.roll(x_hist, 1, axis=1)
        x_hist[:, 0] = X_g1_input[:, t + 1].astype(np.float64)

        if (t + 1) % 100_000 == 0:
            print(f"  t = {t+1:,} / {T:,}")

    print(f"  {label} done.")
    return X_out

# ─────────────────────────────────────────────
# 6. Run all four models
# ─────────────────────────────────────────────
g1_real_input = X_real[group1_idx, :]                              # (G1, T)

# Null: same Experiment2 weights, but G1 neuron identities shuffled
shuffled_order = np.random.default_rng(SEED + 99).permutation(G1)
g1_null_input  = g1_real_input[shuffled_order, :]                  # (G1, T)

X_model   = run_model(g1_real_input, W_exp2,    'model   (Exp2 weights,    real G1)',    run_seed=SEED + 1)
X_between = run_model(g1_real_input, W_between, 'between (Between_Burst weights, real G1)', run_seed=SEED + 2)
X_null    = run_model(g1_null_input, W_exp2,    'null    (Exp2 weights,    shuffled G1)',  run_seed=SEED + 4)

# ─────────────────────────────────────────────
# 6b. Per-neuron statistical comparison (Group 2): firing rate, ISI, pairwise corr, Γ
# ─────────────────────────────────────────────
def compute_gamma(data_row, model_row, delta_ms, bin_ms=1):
    """
    Coincidence factor Γ (Kistler et al. 1997 / Jolivet et al. 2008), with
    one-to-one greedy spike matching (each real spike can be claimed by at
    most one generated spike within ±delta_ms, and vice versa) so that
    padding extra spikes near a real one cannot inflate the score.
    Γ = 1  → perfect reproduction.  Γ = 0 → no better than a matched-rate Poisson process.
    """
    T_local = len(data_row)
    data_times  = np.flatnonzero(data_row)
    model_times = np.flatnonzero(model_row)
    Ndata  = len(data_times)
    Nmodel = len(model_times)
    if Ndata == 0 or (Ndata + Nmodel) == 0:
        return np.nan

    v = Ndata / (T_local * bin_ms / 1000.0)   # data firing rate, Hz
    delta_s = delta_ms / 1000.0
    Nnorm = 1 - 2 * v * delta_s
    if Nnorm <= 0:
        return np.nan

    # Greedy one-to-one matching on sorted spike times (optimal for this
    # windowed line-matching problem): each spike used at most once.
    delta_bins = delta_ms / bin_ms
    i = j = 0
    Ncoinc = 0
    while i < Ndata and j < Nmodel:
        diff = model_times[j] - data_times[i]
        if abs(diff) <= delta_bins:
            Ncoinc += 1
            i += 1
            j += 1
        elif data_times[i] < model_times[j] - delta_bins:
            i += 1
        else:
            j += 1
    Ncoinc_exp = 2 * v * delta_s * Ndata

    return ((Ncoinc - Ncoinc_exp) / (0.5 * (Ndata + Nmodel))) / Nnorm


def plot_stats_comparison(X_gt, variants, indices, out_path):
    """
    variants: list of (X_gen, label, color) tuples, all overlaid on the same
    three panels (mean firing rate, mean ISI, spike-timing reliability Γ).
    """
    idx = np.asarray(indices)
    spikes_gt  = X_gt[idx]   # (G2, T) binary
    duration_s = T / 1000.0

    def mean_isi(spikes_row):
        times = np.where(spikes_row)[0] * 0.001
        if len(times) < 2:
            return np.nan
        return np.diff(times).mean()

    means_GT     = spikes_gt.sum(axis=1) / duration_s
    means_isi_GT = np.array([mean_isi(r) for r in spikes_gt])

    per_variant = []   # (label, color, means_gen, means_isi_gen, isi_mask, gammas, gamma_mask)
    for X_gen, label, color in variants:
        spikes_gen    = X_gen[idx]
        means_gen     = spikes_gen.sum(axis=1) / duration_s
        means_isi_gen = np.array([mean_isi(r) for r in spikes_gen])
        isi_mask      = np.isfinite(means_isi_GT) & np.isfinite(means_isi_gen)
        gammas = np.array([
            compute_gamma(spikes_gt[k], spikes_gen[k], GAMMA_DELTA_MS)
            for k in range(len(idx))
        ])
        gamma_mask = np.isfinite(gammas)
        per_variant.append((label, color, means_gen, means_isi_gen, isi_mask, gammas, gamma_mask))
        print(f"  [{label}] mean Γ = {gammas[gamma_mask].mean():.3f}  (n={gamma_mask.sum()})")

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.6))

    # --- mean firing rates ---
    ax = axes[0]
    vmax = means_GT.max()
    for label, color, means_gen, *_ in per_variant:
        ax.scatter(means_GT, means_gen, color=color, alpha=0.7, s=18, linewidths=0, label=label)
        vmax = max(vmax, means_gen.max())
    vmax *= 1.1
    ax.plot([0, vmax], [0, vmax], color="grey", ls="--", lw=1, zorder=-1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Data"); ax.set_ylabel("Generated")
    ax.set_title("Mean firing rate (Hz)")
    ax.set_aspect(1)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())

    # --- mean ISI (finite neurons only) ---
    ax = axes[1]
    vmax = 0.0
    n_isi = 0
    for label, color, _, means_isi_gen, isi_mask, *_ in per_variant:
        ax.scatter(means_isi_GT[isi_mask], means_isi_gen[isi_mask],
                   color=color, alpha=0.7, s=18, linewidths=0, label=label)
        if isi_mask.any():
            vmax = max(vmax, means_isi_GT[isi_mask].max(), means_isi_gen[isi_mask].max())
        n_isi = max(n_isi, isi_mask.sum())
    vmax *= 1.1
    ax.plot([0, vmax], [0, vmax], color="grey", ls="--", lw=1, zorder=-1)
    ax.set_xlabel("Data"); ax.set_ylabel("Generated")
    ax.set_title(f"Mean ISI, s  (n≤{n_isi})")
    ax.set_xlim(0); ax.set_ylim(0)
    ax.set_aspect(1, adjustable="datalim")

    # --- spike-timing reliability Γ (finite neurons only) ---
    ax = axes[2]
    n_gamma = 0
    for label, color, *_rest, gammas, gamma_mask in per_variant:
        gamma_sorted = np.sort(gammas[gamma_mask])
        ax.scatter(np.arange(len(gamma_sorted)), gamma_sorted,
                   color=color, alpha=0.7, s=18, linewidths=0, label=label)
        n_gamma = max(n_gamma, gamma_mask.sum())
    ax.axhline(0, color="grey", ls="--", lw=1, zorder=-1)
    ax.axhline(1, color="grey", ls=":", lw=1, zorder=-1)
    ax.set_xlabel("Neuron (sorted by Γ)"); ax.set_ylabel("Γ")
    ax.set_title(f"Spike-timing reliability Γ\n(Δ={GAMMA_DELTA_MS}ms, n≤{n_gamma})")

    plt.tight_layout()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(variants),
               bbox_to_anchor=(0.5, 1.12), frameon=False, markerscale=1.6)

    def savefig_with_retry(path, **kwargs):
        for attempt in range(5):
            try:
                fig.savefig(path, **kwargs)
                return
            except OSError as e:
                print(f"  Warning: save to {path} failed ({e}); retrying "
                      f"({attempt + 1}/5)...")
                time.sleep(3)
        fig.savefig(path, **kwargs)   # final attempt, let it raise if it fails

    savefig_with_retry(out_path, dpi=300, bbox_inches="tight")
    savefig_with_retry(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print(f"  Saved: {out_path.with_suffix('.svg')}")

print("Building stats-comparison plot...")
STATS_DIR = Path(__file__).parent
plot_stats_comparison(
    X_real,
    [
        (X_model,   'Trained on full data',         'darkorange'),
        (X_between, 'Trained on between-burst data', 'cadetblue'),
        (X_null,    'Null (shuffled input)',         'mediumpurple'),
    ],
    group2_idx,
    STATS_DIR / f'stats_comparison_config{CONFIG_ID}.png',
)

for stale in ['Exp2', 'BetweenBurst', 'Null']:
    stale_path = STATS_DIR / f'stats_comparison_{stale}_config{CONFIG_ID}.png'
    if stale_path.exists():
        stale_path.unlink()

# ─────────────────────────────────────────────
# 7. Pearson correlation of G2 activity A(t)
#    A(t) = number of G2 spikes in ±25 ms window around t
# ─────────────────────────────────────────────
HALF_WIN = 25
window   = np.ones(2 * HALF_WIN + 1)   # 51-bin boxcar

def g2_activity(X):
    """Sum G2 spikes across neurons, then smooth with ±25ms window."""
    g2_sum = X[group2_idx, :].sum(axis=0).astype(np.float64)   # (T,)
    return np.convolve(g2_sum, window, mode='same')              # (T,)

A_real    = g2_activity(X_real)
A_model   = g2_activity(X_model)
A_between = g2_activity(X_between)
A_null    = g2_activity(X_null)

# Trim edges to avoid zero-padding artifacts
trim = HALF_WIN
r_model   = np.corrcoef(A_real[trim:-trim], A_model[trim:-trim])[0, 1]
r_between = np.corrcoef(A_real[trim:-trim], A_between[trim:-trim])[0, 1]
r_null    = np.corrcoef(A_real[trim:-trim], A_null[trim:-trim])[0, 1]

print(f"\nPearson r — G2 activity A(t) = spikes in ±{HALF_WIN}ms window:")
print(f"  Real vs Exp2 weights:          r = {r_model:.4f}")
print(f"  Real vs Between_Burst weights: r = {r_between:.4f}")
print(f"  Real vs Null:                  r = {r_null:.4f}\n")

# ─────────────────────────────────────────────
# 8. Spike times for Plotly traces
# ─────────────────────────────────────────────
t_axis = t_start_s + np.arange(T) * 0.001

def spike_times_dict(X, indices):
    return {i: t_axis[np.where(X[i])[0]] for i in indices}

g1_real_st    = spike_times_dict(X_real,    group1_idx)
g2_real_st    = spike_times_dict(X_real,    group2_idx)
g2_model_st   = spike_times_dict(X_model,   group2_idx)
g2_between_st = spike_times_dict(X_between, group2_idx)
g2_null_st    = spike_times_dict(X_null,    group2_idx)

# Population activity (ACT_BIN_MS bins)
T2    = (T // ACT_BIN_MS) * ACT_BIN_MS
t_act = t_start_s + (np.arange(T2 // ACT_BIN_MS) * ACT_BIN_MS + ACT_BIN_MS / 2) * 0.001
act_real    = X_real   [:, :T2].reshape(N, -1, ACT_BIN_MS).sum(axis=(0, 2))
act_model   = X_model  [:, :T2].reshape(N, -1, ACT_BIN_MS).sum(axis=(0, 2))
act_between = X_between[:, :T2].reshape(N, -1, ACT_BIN_MS).sum(axis=(0, 2))
act_null    = X_null   [:, :T2].reshape(N, -1, ACT_BIN_MS).sum(axis=(0, 2))

# ─────────────────────────────────────────────
# 8. Build interactive raster
#    Section layout (y-axis, top → bottom):
#      Group 1 real        (steelblue)
#      Group 2 real        (crimson)        ← ground truth
#      Group 2 model       (darkorange)     ← Exp2 weights,    real G1
#      Group 2 between     (cadetblue)      ← Between_Burst weights, real G1
#      Group 2 null        (mediumpurple)   ← Exp2 weights,    shuffled G1
# ─────────────────────────────────────────────
print("Building plot...")

# Print-ready font sizes (used below and again in the "Layout" section for
# the static exports -- legible on-screen either way, just larger than the
# original screen-only sizes).
TITLE_FONT_SIZE  = 26
ANNOT_FONT_SIZE  = 20
AXIS_TITLE_SIZE  = 20
AXIS_TICK_SIZE   = 16

GAP       = 4
y0_g1     = 0
y0_g2r    = G1  + GAP
y0_g2m    = y0_g2r  + G2 + GAP
y0_g2b    = y0_g2m  + G2 + GAP
y0_g2n    = y0_g2b  + G2 + GAP
total_y   = y0_g2n  + G2

def build_figure(use_webgl):
    """
    Build the full figure. use_webgl=True (Scattergl) for the interactive
    HTML, where WebGL performance matters. use_webgl=False (plain SVG
    Scatter) for static image export -- Kaleido's headless-browser
    screenshot is frequently taken before a WebGL canvas finishes painting,
    silently producing a blank plot, so static exports must avoid GL traces.
    """
    scatter_cls = go.Scattergl if use_webgl else go.Scatter

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.80, 0.20],
        vertical_spacing=0.03,
    )

    def add_raster_section(indices, y0, times_dict, color, legend_name, legend_group):
        """Merge all neurons into one trace (None-separated) for fast rendering."""
        xs, ys = [], []
        for rank, idx in enumerate(indices):
            times = times_dict[idx]
            if len(times):
                xs.append(times)
                ys.append(np.full(len(times), y0 + rank))
                xs.append([None])
                ys.append([None])
        if not xs:
            return
        fig.add_trace(scatter_cls(
            x=np.concatenate(xs),
            y=np.concatenate(ys),
            mode='markers',
            marker=dict(symbol='line-ns', size=3, color=color,
                        line=dict(width=1, color=color)),
            name=legend_name, legendgroup=legend_group,
            showlegend=False,
            hovertemplate=f'<b>{legend_name}</b><br>%{{x:.3f}} s<extra></extra>',
        ), row=1, col=1)

    add_raster_section(group1_idx, y0_g1,  g1_real_st,    'steelblue',      'Group 1 (real)',              'g1')
    add_raster_section(group2_idx, y0_g2r, g2_real_st,    'crimson',        'Group 2 (real)',              'g2r')
    add_raster_section(group2_idx, y0_g2m, g2_model_st,   'darkorange',     'Group 2 (Exp2 weights)',      'g2m')
    add_raster_section(group2_idx, y0_g2b, g2_between_st, 'cadetblue',      'Group 2 (Between_Burst w)',   'g2b')
    add_raster_section(group2_idx, y0_g2n, g2_null_st,    'mediumpurple',   'Group 2 (null)',              'g2n')

    # ── Separator lines ──
    t_min = t_start_s
    t_max = t_start_s + T * 0.001
    for y_sep in [y0_g2r - GAP/2, y0_g2m - GAP/2, y0_g2b - GAP/2, y0_g2n - GAP/2]:
        fig.add_shape(type='line',
                      x0=t_min, x1=t_max, y0=y_sep, y1=y_sep,
                      xref='x', yref='y',
                      line=dict(color='rgba(120,120,120,0.6)', width=1.2, dash='dot'))

    # ── Section annotations (right side) ──
    section_labels = [
        (y0_g1  + G1/2, 'steelblue',      'GROUP 1',
         f'{G1} neurons — original spikes kept unchanged.<br>'
         'Fed as fixed real input into all model runs.'),
        (y0_g2r + G2/2, 'crimson',        'GROUP 2 — Real',
         f'{G2} neurons — original spikes (ground truth).<br>'
         'Erased before running any model.'),
        (y0_g2m + G2/2, 'darkorange',     'GROUP 2 — Exp2 weights',
         'Stochastic model with <b>Experiment2</b> (all-spikes) weights.<br>'
         'x<sub>i</sub>(t+1) ~ Bernoulli(σ<sub>i</sub>(Σ w<sub>ij</sub> x<sub>j</sub>(t)))<br>'
         f'β={BETA},  m<sub>i</sub>=ln({BASELINE_SCALE}/B<sub>i</sub>−1)'),
        (y0_g2b + G2/2, 'cadetblue',      'GROUP 2 — Between_Burst weights',
         'Same model with weights derived from <b>between-burst</b> spike-triggered histograms.<br>'
         'Same real G1 input, same β and B<sub>i</sub> — only connectivity pattern differs.'),
        (y0_g2n + G2/2, 'mediumpurple',   'GROUP 2 — Null',
         'Exp2 weights with Group 1 neuron <b>identities shuffled</b>.<br>'
         'Spike trains are real but who-drives-whom is randomized.<br>'
         'Same weights, same β, same B<sub>i</sub> — only wiring identity broken.'),
    ]

    for y_c, color, bold, sub in section_labels:
        fig.add_annotation(
            x=1.01, y=y_c,
            xref='paper', yref='y',
            text=f'<b><span style="color:{color}">{bold}</span></b>',
            xanchor='left', yanchor='middle',
            showarrow=False, font=dict(size=ANNOT_FONT_SIZE), align='left',
        )

    # ── Population activity ──
    for y_act, color, name in [
        (act_real,    'steelblue',      'Activity – real'),
        (act_model,   'darkorange',     'Activity – Exp2 weights'),
        (act_between, 'cadetblue',      'Activity – Between_Burst weights'),
        (act_null,    'mediumpurple',   'Activity – null'),
    ]:
        fig.add_trace(scatter_cls(
            x=t_act, y=y_act,
            mode='lines', line=dict(color=color, width=1.2),
            fill='tozeroy',
            fillcolor=(color
                       .replace('steelblue',      'rgba(70,130,180,0.12)')
                       .replace('darkorange',      'rgba(255,140,0,0.12)')
                       .replace('cadetblue',       'rgba(95,158,160,0.12)')
                       .replace('mediumseagreen',  'rgba(60,179,113,0.12)')
                       .replace('mediumpurple',    'rgba(147,112,219,0.12)')),
            name=name,
            hovertemplate='Time: %{x:.3f}s<br>%{y:.0f} spikes/50ms<extra></extra>',
        ), row=2, col=1)

    # ── Layout ──
    fig.update_layout(
        title=dict(
            text=(f'Stochastic network model — Config {CONFIG_ID}<br>'
                  f'<sup>90/10 random split · Exp2 / Between_Burst weights compared · '
                  f'Null = shuffled G1 identity<br>'
                  f'Pearson r(G2 activity, ±{HALF_WIN}ms):  '
                  f'Exp2 = {r_model:.4f} · '
                  f'Between_Burst = {r_between:.4f} · '
                  f'Null = {r_null:.4f}</sup>'),
            font=dict(size=TITLE_FONT_SIZE), x=0.0, xanchor='left',
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=1400,
        width=1600,
        margin=dict(r=460),
        showlegend=False,
        hovermode='x',
    )

    fig.update_yaxes(
        showgrid=False, fixedrange=True,
        range=[-1, total_y + 1],
        showticklabels=False, ticks='',
        title_text='Neuron', title_font=dict(size=AXIS_TITLE_SIZE),
        row=1, col=1,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor='rgba(200,200,200,0.4)',
        fixedrange=True, title_text=f'Spikes / {ACT_BIN_MS}ms',
        title_font=dict(size=AXIS_TITLE_SIZE), tickfont=dict(size=AXIS_TICK_SIZE),
        row=2, col=1,
    )
    fig.update_xaxes(
        title_text='Time (s)', title_font=dict(size=AXIS_TITLE_SIZE),
        tickfont=dict(size=AXIS_TICK_SIZE),
        showgrid=True, gridcolor='rgba(200,200,200,0.3)',
        row=2, col=1,
    )
    return fig

fig = build_figure(use_webgl=True)
out = Path(__file__).parent / f'model_raster_config{CONFIG_ID}.html'
fig.write_html(out, include_plotlyjs='cdn')
print(f"Saved: {out}")
print("Open in any browser — zoom/pan links both panels.")

# ─────────────────────────────────────────────
# 9. Static, print-quality exports (≥300 DPI) for the paper
#    - Full-duration overview
#    - Zoomed excerpt centered on the busiest population-activity window
# ─────────────────────────────────────────────
print("\nBuilding static exports for the paper...")
PRINT_DIR = EXPERIMENT2 / 'Presentation_Pictures'
PRINT_DIR.mkdir(exist_ok=True)

# scale=3 on the 1600x1400 layout -> 4800x4200px (~20MP), i.e. 300+ DPI at
# up to a 16x14" print size -- ample headroom, easy to downscale in the paper.
EXPORT_SCALE = 3

def export_static(figure, path, retries=5, delay=5, **kwargs):
    for attempt in range(retries):
        try:
            figure.write_image(str(path), **kwargs)
            print(f"  Saved: {path}")
            return
        except Exception as e:
            print(f"  Warning: export to {path} failed ({e}); retrying "
                  f"({attempt + 1}/{retries})...")
            time.sleep(delay)
    figure.write_image(str(path), **kwargs)   # final attempt, let it raise

fig_print = build_figure(use_webgl=False)   # SVG traces -- required for Kaleido to render data at all
export_static(fig_print, PRINT_DIR / f'model_raster_config{CONFIG_ID}_full.png',
              width=1600, height=1400, scale=EXPORT_SCALE)

# Zoom window: centered on the busiest 10ms bin of total population activity
zoom_half_s   = 7.5
peak_idx_act  = int(np.argmax(act_real))
zoom_center_s = float(t_act[peak_idx_act])
zoom_t0       = max(t_start_s, zoom_center_s - zoom_half_s)
zoom_t1       = min(t_start_s + T * 0.001, zoom_center_s + zoom_half_s)

fig_zoom = go.Figure(fig_print)
fig_zoom.update_xaxes(range=[zoom_t0, zoom_t1], row=1, col=1)
fig_zoom.update_xaxes(range=[zoom_t0, zoom_t1], row=2, col=1)
fig_zoom.update_layout(title=dict(
    text=(f'Stochastic network model — Config {CONFIG_ID}  '
          f'(zoomed excerpt: {zoom_t0:.1f}s–{zoom_t1:.1f}s, '
          f'busiest population window)<br>'
          f'<sup>90/10 random split · Exp2 / Between_Burst weights compared · '
          f'Null = shuffled G1 identity</sup>'),
    font=dict(size=TITLE_FONT_SIZE), x=0.0, xanchor='left',
))
export_static(fig_zoom, PRINT_DIR / f'model_raster_config{CONFIG_ID}_zoom.png',
              width=1600, height=1400, scale=EXPORT_SCALE)

print("Done.")
