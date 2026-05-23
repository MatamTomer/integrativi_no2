import json
import numpy as np
from scipy.ndimage import gaussian_filter1d

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────
GAUSSIAN_SIGMA_MS   = 200   # sigma of Gaussian smoothing kernel (ms)
THRESHOLD_GAUSSIAN  = 4     # Gaussian-smoothed activity must be > this (spikes/50ms)
MIN_ACTIVE_NEURONS  = 4     # distinct neurons active in 200ms window must be > this

# ─────────────────────────────────────────────
# 1. Load activity data
# ─────────────────────────────────────────────
print("Loading activity data...")

data           = np.load('activity.npz')
time_s         = data['time_s']          # time axis in seconds, 10ms resolution
activity       = data['activity']        # spikes per 50ms window
active_neurons = data['active_neurons']  # distinct neurons active in 200ms window

BIN_SIZE_MS = round((time_s[1] - time_s[0]) * 1000)   # 10ms
print(f"  {len(time_s):,} time bins at {BIN_SIZE_MS}ms resolution")
print(f"  Activity range: {activity.min():.0f} - {activity.max():.0f} spikes/50ms")

# ─────────────────────────────────────────────
# 2. Apply Gaussian smoothing
#
#    sigma in bins = sigma_ms / bin_size_ms
# ─────────────────────────────────────────────
sigma_bins = GAUSSIAN_SIGMA_MS / BIN_SIZE_MS   # 200ms / 10ms = 20 bins
smoothed   = gaussian_filter1d(activity.astype(np.float64), sigma=sigma_bins)

print(f"\n  Gaussian sigma: {GAUSSIAN_SIGMA_MS}ms ({sigma_bins:.0f} bins)")
print(f"  Smoothed range: {smoothed.min():.2f} - {smoothed.max():.2f}")

# ─────────────────────────────────────────────
# 3. Apply two-condition threshold
#
#    Both conditions must be true simultaneously:
#      - raw activity >= THRESHOLD_RAW  (at least 2 spikes in 50ms window)
#      - smoothed     >  THRESHOLD_GAUSSIAN  (sustained activity above baseline)
# ─────────────────────────────────────────────
above = (smoothed > THRESHOLD_GAUSSIAN) & (active_neurons > MIN_ACTIVE_NEURONS)

print(f"\n  Threshold Gaussian: >  {THRESHOLD_GAUSSIAN} spikes/50ms (smoothed)")
print(f"  Min active neurons: >  {MIN_ACTIVE_NEURONS}")
print(f"  Bins passing both:  {above.sum():,} ({100*above.sum()/len(above):.1f}% of recording)")

# ─────────────────────────────────────────────
# 4. Find contiguous segments above threshold
#    (no merging, no min duration)
# ─────────────────────────────────────────────
rising  = np.where(np.diff(above.astype(int)) == +1)[0] + 1
falling = np.where(np.diff(above.astype(int)) == -1)[0] + 1

if above[0]:
    rising = np.concatenate([[0], rising])
if above[-1]:
    falling = np.concatenate([falling, [len(above)]])

raw_segments = list(zip(rising, falling))
print(f"\n  Raw burst segments: {len(raw_segments)}")

# ─────────────────────────────────────────────
# 5. Compute burst properties and print summary
# ─────────────────────────────────────────────
print(f"\n{'#':>4}  {'Start (s)':>10}  {'End (s)':>10}  {'Duration (ms)':>14}  {'Peak (spk/50ms)':>16}")
print("-" * 62)

burst_records = []
for idx, (s, e) in enumerate(raw_segments):
    t_start  = float(time_s[s])
    t_end    = float(time_s[e - 1])
    duration = (e - s) * BIN_SIZE_MS
    peak     = float(activity[s:e].max())

    print(f"  {idx+1:>3}  {t_start:>10.2f}  {t_end:>10.2f}  {duration:>13}  {peak:>15.0f}")

    burst_records.append({
        'burst_id':      int(idx + 1),
        't_start_s':     float(t_start),
        't_end_s':       float(t_end),
        'duration_ms':   int(duration),
        'peak_activity': float(peak),
    })

total_burst_time = sum(r['duration_ms'] for r in burst_records) / 1000
recording_time   = float(time_s[-1] - time_s[0])
print(f"\n  Total burst time: {total_burst_time:.1f}s  ({100*total_burst_time/recording_time:.1f}% of recording)")

# ─────────────────────────────────────────────
# 6. Save to bursts_gaussian.json
# ─────────────────────────────────────────────
output = {
    'method':             'gaussian',
    'gaussian_sigma_ms':  GAUSSIAN_SIGMA_MS,
    'threshold_gaussian': THRESHOLD_GAUSSIAN,
    'min_active_neurons': MIN_ACTIVE_NEURONS,
    'n_bursts':           len(burst_records),
    'bursts':             burst_records,
}

with open('bursts_gaussian.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved: bursts_gaussian.json")
print("Done.")
