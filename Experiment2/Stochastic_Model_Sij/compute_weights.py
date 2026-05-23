import pandas as pd
import numpy as np

print("Loading sij scores...")
df = pd.read_csv('../Significance_Analysis/sij_v2_scores.csv')

all_units = sorted(set(df['trigger_unit']) | set(df['response_unit']))
N = len(all_units)
unit_idx = {u: i for i, u in enumerate(all_units)}
print(f"  {N} neurons, {len(df)} pairs")

# ─────────────────────────────────────────────
# Denominator: sum of |S_ki| over ALL k != i
# (trigger_unit=k, response_unit=i, all pairs)
# ─────────────────────────────────────────────
denom = df[df['S_ij'].abs() > 10].groupby('response_unit')['S_ij'].apply(lambda x: x.abs().sum())
denom = denom.reindex(all_units, fill_value=0.0)

print(f"  Denominator range: {denom[denom > 0].min():.2f} – {denom.max():.2f}")
print(f"  Neurons with zero total incoming S: {(denom == 0).sum()}")

# ─────────────────────────────────────────────
# Weights: w_ij = S_ji / denom_i  if |S_ji| > 10, else 0
# (trigger_unit=j, response_unit=i)
# ─────────────────────────────────────────────
sig = df[df['S_ij'].abs() > 10].copy()

sig['w_ij'] = sig.apply(
    lambda r: r['S_ij'] / denom[r['response_unit']]
    if denom[r['response_unit']] > 0 else 0.0,
    axis=1
)

sparse = sig[['trigger_unit', 'response_unit', 'S_ij', 'w_ij']].copy()
sparse.columns = ['source_unit', 'target_unit', 'S_ji', 'w_ij']
sparse = sparse.sort_values(['target_unit', 'source_unit']).reset_index(drop=True)
sparse.to_csv('weights.csv', index=False)
print("Saved: weights.csv")

# ─────────────────────────────────────────────
# Dense matrix W[i, j] = weight from source j to target i
# ─────────────────────────────────────────────
W = np.zeros((N, N), dtype=np.float64)
for _, row in sig.iterrows():
    j = unit_idx[int(row['trigger_unit'])]
    i = unit_idx[int(row['response_unit'])]
    W[i, j] = row['w_ij']

wide = pd.DataFrame(W, index=all_units, columns=all_units)
wide.index.name = 'target_unit'
wide.to_csv('weights_wide.csv')
print("Saved: weights_wide.csv")

# Summary
excit = (W > 0).sum()
inhib = (W < 0).sum()
nonzero = excit + inhib
print(f"\n  Non-zero weights : {nonzero}")
print(f"  Excitatory (w>0) : {excit}")
print(f"  Inhibitory (w<0) : {inhib}")
print(f"  w range          : {W[W != 0].min():.6f} – {W[W != 0].max():.6f}")
row_abs_sum = np.abs(W).sum(axis=1)
print(f"  |w| row-sum range: {row_abs_sum[row_abs_sum > 0].min():.4f} – {row_abs_sum.max():.4f}")
print("Done.")
