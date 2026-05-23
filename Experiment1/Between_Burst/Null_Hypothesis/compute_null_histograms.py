import json
import numpy as np
from numba import njit, prange
from pathlib import Path

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────
WINDOW_MS   = 10
WINDOW_SEC  = WINDOW_MS / 1000.0
BIN_EDGES   = list(range(-WINDOW_MS, 0)) + list(range(1, WINDOW_MS + 1))
NUM_BINS    = len(BIN_EDGES)   # 20

N_BOOTSTRAP = 10_000
JITTER_MIN  = 20 / 1000.0     # 20ms in seconds
JITTER_MAX  = 50 / 1000.0     # 50ms in seconds


# ─────────────────────────────────────────────
# Numba JIT kernel
# ─────────────────────────────────────────────
@njit(parallel=True)
def compute_all_bootstraps(trigger_spikes, jitter_matrix, window_sec, window_ms, num_bins):
    """
    Compute histograms for all N_BOOTSTRAP versions of neuron j simultaneously.
    Compiled to machine code by Numba — simple loops run at C speed.
    Parallelized across the bootstrap dimension via prange (OpenMP threads).

    Args:
        trigger_spikes : sorted 1D array of shape (N_i,)
        jitter_matrix  : 2D array of shape (N_BOOTSTRAP, N_j) — each row is one
                         jittered version of neuron j's spike train, pre-sorted
        window_sec     : float, ±window in seconds
        window_ms      : int, ±window in ms (= 10)
        num_bins       : int (= 20)

    Returns:
        histograms : 2D int array of shape (N_BOOTSTRAP, num_bins)
    """
    n_bootstrap = jitter_matrix.shape[0]
    N_j         = jitter_matrix.shape[1]
    N_i         = len(trigger_spikes)
    histograms  = np.zeros((n_bootstrap, num_bins), dtype=np.int32)

    # Each bootstrap is fully independent — parallelize across them
    for b in prange(n_bootstrap):

        for k in range(N_j):
            t_j = jitter_matrix[b, k]

            # Binary search: find trigger spikes within [t_j - window, t_j + window]
            # Manual searchsorted (required inside @njit)
            lo = 0
            hi_bound = N_i
            target_lo = t_j - window_sec
            while lo < hi_bound:
                mid = (lo + hi_bound) // 2
                if trigger_spikes[mid] < target_lo:
                    lo = mid + 1
                else:
                    hi_bound = mid

            hi = lo
            hi_bound = N_i
            target_hi = t_j + window_sec
            while hi < hi_bound:
                mid = (hi + hi_bound) // 2
                if trigger_spikes[mid] <= target_hi:
                    hi = mid + 1
                else:
                    hi_bound = mid

            # Loop over nearby trigger spikes and bin the offset
            for idx in range(lo, hi):
                offset_ms = int(round((t_j - trigger_spikes[idx]) * 1000))

                if offset_ms == 0:
                    continue

                if -window_ms <= offset_ms <= -1:
                    bin_idx = offset_ms + window_ms        # -10→0, -1→9
                elif 1 <= offset_ms <= window_ms:
                    bin_idx = offset_ms + window_ms - 1   # 1→10, 10→19
                else:
                    continue

                histograms[b, bin_idx] += 1

    return histograms


# ─────────────────────────────────────────────
# Process one neuron pair
# ─────────────────────────────────────────────
def process_pair(trigger_spikes, other_spikes, original_hist, rng):
    """
    Run all 10,000 bootstraps for one (i, j) pair.
    Returns: null_mean, null_std, p_high, p_low — each shape (NUM_BINS,)
    """
    N_j = len(other_spikes)

    CHUNK_SIZE = 1000

    mean      = np.zeros(NUM_BINS, dtype=np.float64)
    M2        = np.zeros(NUM_BINS, dtype=np.float64)
    count_gte = np.zeros(NUM_BINS, dtype=np.int64)
    count_lte = np.zeros(NUM_BINS, dtype=np.int64)
    n_done    = 0

    for chunk_start in range(0, N_BOOTSTRAP, CHUNK_SIZE):
        chunk = min(CHUNK_SIZE, N_BOOTSTRAP - chunk_start)

        # Generate jitter matrix for this chunk only: shape (chunk, N_j)
        magnitudes    = rng.uniform(JITTER_MIN, JITTER_MAX, size=(chunk, N_j))
        directions    = rng.integers(0, 2, size=(chunk, N_j)) * 2 - 1
        jitter_matrix = other_spikes[np.newaxis, :] + directions * magnitudes
        jitter_matrix.sort(axis=1)

        chunk_hists = compute_all_bootstraps(
            trigger_spikes, jitter_matrix,
            WINDOW_SEC, WINDOW_MS, NUM_BINS
        )

        # Welford update and p-value counts
        for b in range(chunk):
            hist    = chunk_hists[b].astype(np.float64)
            n_done += 1
            delta   = hist - mean
            mean   += delta / n_done
            delta2  = hist - mean
            M2     += delta * delta2
            count_gte += (chunk_hists[b] >= original_hist)
            count_lte += (chunk_hists[b] <= original_hist)

    std    = np.sqrt(M2 / N_BOOTSTRAP)
    p_high = count_gte / N_BOOTSTRAP
    p_low  = count_lte / N_BOOTSTRAP

    return mean, std, p_high, p_low


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':

    print("Loading neuron data...")
    with open('../neuron_data.json', 'r') as f:
        neuron_data = json.load(f)

    N = len(neuron_data)
    spike_times = []
    for neuron in neuron_data:
        times = np.array([spike['spikeTime'] for spike in neuron], dtype=np.float64)
        times.sort()
        spike_times.append(times)

    neuron_ids = [neuron[0]['unit'] for neuron in neuron_data if neuron]
    print(f"  {N} neurons, {sum(len(t) for t in spike_times):,} total spikes")

    print("Loading original histograms...")
    with open('../Spike_Triggered_Average/histogram_data.json', 'r') as f:
        hist_data = json.load(f)
    original_histograms = np.array(hist_data['histograms'], dtype=np.int32)
    print(f"  Loaded histograms of shape {original_histograms.shape}")

    # Warmup Numba JIT compiler with a tiny dummy call
    print("\nWarming up Numba JIT compiler...")
    dummy_trigger = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    dummy_jitter  = np.array([[0.05, 0.15], [0.06, 0.16]], dtype=np.float64)
    compute_all_bootstraps(dummy_trigger, dummy_jitter, WINDOW_SEC, WINDOW_MS, NUM_BINS)
    print("  Done.\n")

    # Allocate output arrays
    null_mean_all = np.zeros((N, N, NUM_BINS), dtype=np.float64)
    null_std_all  = np.zeros((N, N, NUM_BINS), dtype=np.float64)
    p_high_all    = np.zeros((N, N, NUM_BINS), dtype=np.float64)
    p_low_all     = np.zeros((N, N, NUM_BINS), dtype=np.float64)

    total_pairs = N * (N - 1)
    done        = 0
    rng         = np.random.default_rng(42)

    print(f"Processing {total_pairs} pairs ({N_BOOTSTRAP:,} bootstraps each)...\n")

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if len(spike_times[i]) == 0 or len(spike_times[j]) == 0:
                continue

            null_mean, null_std, p_high, p_low = process_pair(
                spike_times[i], spike_times[j],
                original_histograms[i][j],
                rng
            )

            null_mean_all[i][j] = null_mean
            null_std_all[i][j]  = null_std
            p_high_all[i][j]    = p_high
            p_low_all[i][j]     = p_low

            done += 1
            print(f"  [{done}/{total_pairs}] Neuron {neuron_ids[i]} -> {neuron_ids[j]} done")

    print(f"\nAll pairs done.")

    def save_json(array, path, description):
        data = {
            'neuron_ids':    neuron_ids,
            'bins':          BIN_EDGES,
            'window_ms':     WINDOW_MS,
            'n_bootstrap':   N_BOOTSTRAP,
            'jitter_min_ms': int(JITTER_MIN * 1000),
            'jitter_max_ms': int(JITTER_MAX * 1000),
            'description':   description,
            'data':          array.tolist()
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {path}")

    save_json(
        null_mean_all, Path('null_histograms.json'),
        'null_histograms[i][j][k] = mean count across 10000 jittered bootstraps '
        'for neuron j spikes at bin k relative to neuron i triggers'
    )
    save_json(
        null_std_all, Path('null_std.json'),
        'null_std[i][j][k] = std of null histogram counts across 10000 bootstraps '
        'for pair (i,j) at bin k'
    )
    save_json(
        p_high_all, Path('p_high.json'),
        'p_high[i][j][k] = fraction of bootstrap histograms >= observed histogram '
        'at bin k for pair (i,j). Small value = significant excitation.'
    )
    save_json(
        p_low_all, Path('p_low.json'),
        'p_low[i][j][k] = fraction of bootstrap histograms <= observed histogram '
        'at bin k for pair (i,j). Small value = significant inhibition.'
    )

    print("\nDone.")
