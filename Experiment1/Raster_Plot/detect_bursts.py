import json
import numpy as np

# ─────────────────────────────────────────────
# Parameters — tune these to adjust burst detection
# ─────────────────────────────────────────────
THRESHOLD_K           = 3     # threshold = median + k * std (unused when MANUAL_THRESHOLD is set)
MANUAL_THRESHOLD      = 1     # set to None to use adaptive threshold
MIN_ACTIVE_NEURONS    = 4     # burst must have at least this many distinct neurons active in window
MIN_BURST_DURATION_MS = 200   # discard bursts shorter than this (ms)
MERGE_GAP_MS          = 50    # merge two bursts if the gap between them is shorter than this (ms)

# ─────────────────────────────────────────────
# 1. Load activity data
# ─────────────────────────────────────────────
print("Loading activity data...")

data           = np.load('activity.npz')
time_s         = data['time_s']          # time axis in seconds, 10ms resolution
activity       = data['activity']        # spikes per 50ms window
active_neurons = data['active_neurons']  # distinct neurons active in 50ms window

BIN_SIZE_MS = round((time_s[1] - time_s[0]) * 1000)   # should be 10ms
print(f"  {len(time_s):,} time bins at {BIN_SIZE_MS}ms resolution")
print(f"  Activity range: {activity.min():.0f} – {activity.max():.0f} spikes/50ms")
print(f"  Active neurons range: {active_neurons.min():.0f} – {active_neurons.max():.0f}")

# ─────────────────────────────────────────────
# 2. Compute adaptive threshold
#
#    Most bins are zero (silence between bursts), so using the full
#    distribution gives median=0 which is uninformative.
#    Instead we compute baseline and noise from non-zero bins only —
#    these represent the actual active periods of the recording.
# ─────────────────────────────────────────────
if MANUAL_THRESHOLD is not None:
    threshold = MANUAL_THRESHOLD
    print(f"\n  Using manual threshold: {threshold} spikes/50ms")
else:
    active_bins = activity[activity > 0]
    baseline    = np.median(active_bins)
    noise       = np.std(active_bins)
    threshold   = baseline + THRESHOLD_K * noise
    print(f"\n  Non-zero bins: {len(active_bins):,} ({100*len(active_bins)/len(activity):.1f}% of recording)")
    print(f"  Baseline (median of non-zero): {baseline:.2f} spikes/50ms")
    print(f"  Noise (std of non-zero):       {noise:.2f} spikes/50ms")
    print(f"  Threshold (k={THRESHOLD_K}):              {threshold:.2f} spikes/50ms")

# ─────────────────────────────────────────────
# 3. Find contiguous segments above threshold
# ─────────────────────────────────────────────
above = (activity > threshold) & (active_neurons >= MIN_ACTIVE_NEURONS)

# Detect rising edges (False→True) and falling edges (True→False)
rising  = np.where(np.diff(above.astype(int)) == +1)[0] + 1
falling = np.where(np.diff(above.astype(int)) == -1)[0] + 1

# Handle edge cases: recording starts or ends mid-burst
if above[0]:
    rising = np.concatenate([[0], rising])
if above[-1]:
    falling = np.concatenate([falling, [len(above)]])

raw_bursts = list(zip(rising, falling))   # list of (start_bin, end_bin) pairs
print(f"\n  Raw segments above threshold: {len(raw_bursts)}")

# ─────────────────────────────────────────────
# 4. Merge bursts that are too close together
#
#    If the gap between two consecutive bursts is less than
#    MERGE_GAP_MS, they are likely one burst with a brief dip.
# ─────────────────────────────────────────────
MERGE_GAP_BINS = MERGE_GAP_MS // BIN_SIZE_MS

merged_bursts = []
for start, end in raw_bursts:
    if merged_bursts and (start - merged_bursts[-1][1]) <= MERGE_GAP_BINS:
        # extend the previous burst to absorb this one
        merged_bursts[-1] = (merged_bursts[-1][0], end)
    else:
        merged_bursts.append((start, end))

print(f"  After merging (gap < {MERGE_GAP_MS}ms): {len(merged_bursts)}")

# ─────────────────────────────────────────────
# 5. Filter by minimum duration
# ─────────────────────────────────────────────
MIN_BURST_BINS = MIN_BURST_DURATION_MS // BIN_SIZE_MS

bursts = [(s, e) for s, e in merged_bursts if (e - s) >= MIN_BURST_BINS]
print(f"  After min duration filter (>= {MIN_BURST_DURATION_MS}ms): {len(bursts)} bursts")

# ─────────────────────────────────────────────
# 6. Compute burst properties and print summary
# ─────────────────────────────────────────────
print(f"\n{'#':>4}  {'Start (s)':>10}  {'End (s)':>10}  {'Duration (ms)':>14}  {'Peak (spk/50ms)':>16}")
print("-" * 62)

burst_records = []
for idx, (s, e) in enumerate(bursts):
    t_start   = float(time_s[s])
    t_end     = float(time_s[e - 1])
    duration  = (e - s) * BIN_SIZE_MS
    peak      = float(activity[s:e].max())

    print(f"  {idx+1:>3}  {t_start:>10.2f}  {t_end:>10.2f}  {duration:>13}  {peak:>15.0f}")

    burst_records.append({
        'burst_id':    int(idx + 1),
        't_start_s':  float(t_start),
        't_end_s':    float(t_end),
        'duration_ms': int(duration),
        'peak_activity': float(peak),
    })

total_burst_time = sum(r['duration_ms'] for r in burst_records) / 1000
recording_time   = float(time_s[-1] - time_s[0])
print(f"\n  Total burst time: {total_burst_time:.1f}s  ({100*total_burst_time/recording_time:.1f}% of recording)")

# ─────────────────────────────────────────────
# 7. Save bursts to JSON
# ─────────────────────────────────────────────
output = {
    'threshold':            float(threshold),
    'threshold_k':          THRESHOLD_K,
    'min_burst_duration_ms': MIN_BURST_DURATION_MS,
    'merge_gap_ms':         MERGE_GAP_MS,
    'n_bursts':             len(burst_records),
    'bursts':               burst_records,
}

with open('bursts.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved: bursts.json")
print("Done.")
