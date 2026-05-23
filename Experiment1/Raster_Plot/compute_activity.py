import json
import numpy as np
from scipy.ndimage import uniform_filter1d

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────
BIN_SIZE_MS     = 1      # each time bin is 1 ms
WINDOW_MS            = 50    # sliding window for spike count
WINDOW_MS_NEURONS    = 200   # sliding window for active neuron count
DOWNSAMPLE      = 10     # save every Nth bin to reduce output size (10ms resolution)

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

N            = len(neuron_ids)
total_spikes = sum(len(t) for t in spike_times)
print(f"  {N} neurons, {total_spikes:,} total spikes")

# ─────────────────────────────────────────────
# 2. Define the 1ms time bins
#
#    The bin edges go from the earliest spike to the latest spike.
#    np.arange gives us one edge per ms, so there are n_bins bins.
# ─────────────────────────────────────────────
all_spikes = np.concatenate(spike_times)
t_start    = np.floor(all_spikes.min())     # seconds, rounded down
t_end      = np.ceil(all_spikes.max())      # seconds, rounded up

bin_size_sec = BIN_SIZE_MS / 1000.0
bin_edges    = np.arange(t_start, t_end + bin_size_sec, bin_size_sec)
n_bins       = len(bin_edges) - 1

print(f"  Recording: {t_start:.0f}s to {t_end:.0f}s  ({(t_end - t_start)/3600:.1f} hours)")
print(f"  Number of 1ms bins: {n_bins:,}")
print(f"  Memory for spike count array: ~{n_bins * 2 / 1e9:.2f} GB (int16)")

# ─────────────────────────────────────────────
# 3. Count spikes per 1ms bin, summed across all neurons
#
#    We accumulate into a single int16 array.
#    int16 can hold up to 32,767 spikes per ms — more than enough.
#    We add one neuron at a time to avoid allocating a large 2D array.
# ─────────────────────────────────────────────
print("\nCounting spikes per 1ms bin...")

# population_counts[t] = total spikes across all neurons in bin t
population_counts = np.zeros(n_bins, dtype=np.int16)

# neuron_fired[n][t] = 1 if neuron n fired at least once in bin t, else 0
# We keep one row per neuron so we can later ask "how many neurons fired
# at least once in the 50ms window around t" (Option B)
neuron_fired = np.zeros((N, n_bins), dtype=np.int8)

for idx, (nid, times) in enumerate(zip(neuron_ids, spike_times)):
    counts, _ = np.histogram(times, bins=bin_edges)
    counts = counts.astype(np.int16)
    population_counts  += counts
    neuron_fired[idx]   = (counts > 0).astype(np.int8)
    print(f"  [{idx+1}/{N}] Neuron {nid}: {len(times):,} spikes added")

print(f"  Total spike count in array: {population_counts.sum():,}")

# ─────────────────────────────────────────────
# 4. Apply the 50ms sliding window via convolution
#
#    uniform_filter1d computes the MEAN over a window of size WINDOW_MS.
#    Multiplying by WINDOW_MS converts mean → sum, giving us the total
#    spike count in a 50ms window centered on each bin.
#
#    This is equivalent to sliding a boxcar of [1,1,...,1] (50 ones)
#    across the array, but runs in O(n) time instead of O(n * window).
# ─────────────────────────────────────────────
print("\nApplying 50ms sliding window...")

# Spike count: total spikes in 50ms window
smoothed_spikes = uniform_filter1d(
    population_counts.astype(np.float32),
    size=WINDOW_MS,
    mode='nearest'
) * WINDOW_MS        # convert mean → sum

# Active neurons (Option B): how many distinct neurons fired at least once
# in the 100ms window around each bin.
# For each neuron, convolve its binary fired array with a boxcar of WINDOW_MS_NEURONS ones.
# Result per neuron: > 0 means it fired somewhere in the window.
# Sum across neurons gives the count of active neurons per bin.
neuron_active_in_window = np.zeros(n_bins, dtype=np.int16)
for n in range(N):
    fired_smoothed = uniform_filter1d(
        neuron_fired[n].astype(np.float32),
        size=WINDOW_MS_NEURONS,
        mode='nearest'
    )
    neuron_active_in_window += (fired_smoothed > 0).astype(np.int16)

smoothed_neurons = neuron_active_in_window.astype(np.float32)

print(f"  Spike count  — max: {smoothed_spikes.max():.0f},  mean: {smoothed_spikes.mean():.1f}")
print(f"  Active neurons — max: {smoothed_neurons.max():.0f}, mean: {smoothed_neurons.mean():.1f}")

# ─────────────────────────────────────────────
# 5. Downsample from 1ms to 10ms resolution
#
#    537 million points is too large to store or plot efficiently.
#    Taking every 10th bin reduces to ~54 million points,
#    which is still fine for zooming in the interactive plot.
# ─────────────────────────────────────────────
print(f"\nDownsampling by {DOWNSAMPLE}x ({BIN_SIZE_MS}ms -> {BIN_SIZE_MS * DOWNSAMPLE}ms resolution)...")

# Time axis: center of each bin, in seconds
bin_centers    = bin_edges[:-1] + bin_size_sec / 2
time_ds         = bin_centers[::DOWNSAMPLE]
spike_count_ds  = smoothed_spikes[::DOWNSAMPLE]
active_neurons_ds = smoothed_neurons[::DOWNSAMPLE]

print(f"  Downsampled array size: {len(time_ds):,} points")

# ─────────────────────────────────────────────
# 6. Save to disk
# ─────────────────────────────────────────────
print("\nSaving...")

np.savez_compressed(
    'activity.npz',
    time_s         = time_ds,           # time axis in seconds
    activity       = spike_count_ds,    # total spikes per 50ms window
    active_neurons = active_neurons_ds, # active neuron count per 50ms window
)
print("  Saved: activity.npz (time_s, activity, active_neurons)")
print("\nDone.")
