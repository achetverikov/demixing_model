"""Checks for the corrected empirical SD and the bin-pooled model SD."""
import numpy as np
import jax.numpy as jnp

from shared.config import config
from shared.mu1_axis import mu1_cell_width
from model_fit_to_data.create_unified_subject_plots import (
    SD_N_BINS, compute_empirical_sd_curve, compute_feat_bin_weights,
    compute_predicted_sd_curves_batch, compute_predicted_sd_curves_batch_pooled,
)

rng = np.random.default_rng(0)
feat_vals = np.arange(config.feat_diff_range[0], config.feat_diff_range[1] + 1, 2)


def wrapped_normal_surface(sds_per_column, means_per_column):
    """Log densities: one von-Mises-ish column per feature difference."""
    grid = np.asarray(config.create_grid('mu1_bias'), dtype=float)
    d = np.radians((grid[:, None] - means_per_column[None, :] + 180) % 360 - 180)
    sd_rad = np.radians(sds_per_column)[None, :]
    logp = -0.5 * (d / sd_rad) ** 2
    p = np.exp(logp)
    p /= p.sum(axis=0, keepdims=True) * mu1_cell_width()
    return np.log(p)[None, ...]


# --- 1. small-sample correction removes the downward bias -------------------
print("small-sample correction (true SD 30 deg, mean over 4000 replicates):")
for n in (3, 10, 50):
    old, new = [], []
    for _ in range(4000):
        bias = rng.normal(0, 30, size=n)
        fd = np.full(n, 50.0)
        a = np.radians(bias)
        r = np.hypot(np.mean(np.cos(a)), np.mean(np.sin(a)))
        old.append(np.degrees(np.sqrt(-2 * np.log(max(r, 1e-10)))))
        _, sds, counts = compute_empirical_sd_curve(fd, bias)
        new.append(np.nanmean(sds))
        assert counts.sum() == n
    print(f"  n={n:>3}  uncorrected {np.mean(old):5.1f}   corrected {np.nanmean(new):5.1f}")

# --- 2. degenerate bins are NaN, not 389 deg -------------------------------
_, sds, counts = compute_empirical_sd_curve(np.array([50.0, 50.0]),
                                            np.array([0.0, 180.0]))
assert np.all(np.isnan(sds)), sds
print("\nantipodal pair -> NaN (not a clamped spike): ok")

# --- 3. weights: discrete design collapses onto one column -----------------
discrete_fd = np.repeat([30.0, 70.0, 110.0], 200)
w = compute_feat_bin_weights(discrete_fd, feat_vals)
occupied = np.where(w.sum(axis=1) > 0)[0]
assert len(occupied) == 3, occupied
for b in occupied:
    assert np.count_nonzero(w[b]) == 1, (b, np.count_nonzero(w[b]))
print("discrete (Moors-like) design -> 1 column per occupied bin: ok")

cont_fd = rng.uniform(2, 180, size=5000)
wc = compute_feat_bin_weights(cont_fd, feat_vals)
assert np.all(np.abs(wc.sum(axis=1) - 1) < 1e-12)
print(f"continuous design -> {np.count_nonzero(wc[5])} columns in bin 5, rows sum to 1: ok")

# --- 4. pooled == unpooled when the bin holds a single column --------------
means = np.linspace(-40, 40, len(feat_vals))
surf = wrapped_normal_surface(np.full(len(feat_vals), 25.0), means)
fine = np.asarray(compute_predicted_sd_curves_batch(jnp.asarray(surf), feat_vals))[0]
pooled = np.asarray(compute_predicted_sd_curves_batch_pooled(
    jnp.asarray(surf), jnp.asarray(w[None, ...])))[0]
for b in occupied:
    col = int(np.argmax(w[b]))
    assert abs(pooled[b] - fine[col]) < 1e-3, (b, pooled[b], fine[col])
empty = [b for b in range(SD_N_BINS) if b not in occupied]
assert np.all(np.isnan(pooled[empty]))
print("pooled == fine-grid on delta weights; empty bins NaN: ok")

# --- 5. pooling a moving-mean bin broadens the model SD --------------------
pooled_cont = np.asarray(compute_predicted_sd_curves_batch_pooled(
    jnp.asarray(surf), jnp.asarray(wc[None, ...])))[0]
nearest = np.array([fine[int(np.argmin(np.abs(feat_vals - c)))]
                    for c in (np.linspace(2, 180, SD_N_BINS + 1)[:-1]
                              + np.linspace(2, 180, SD_N_BINS + 1)[1:]) / 2])
print(f"\nmoving-mean surface, bin 8: fine {nearest[8]:.2f} -> pooled {pooled_cont[8]:.2f} deg")
assert np.all(pooled_cont > nearest - 1e-6)
print("pooling broadens (never narrows) the model SD: ok")

# --- 6. flat-mean surface: pooling changes nothing -------------------------
flat = wrapped_normal_surface(np.full(len(feat_vals), 25.0), np.zeros(len(feat_vals)))
fine_flat = np.asarray(compute_predicted_sd_curves_batch(jnp.asarray(flat), feat_vals))[0]
pooled_flat = np.asarray(compute_predicted_sd_curves_batch_pooled(
    jnp.asarray(flat), jnp.asarray(wc[None, ...])))[0]
assert np.nanmax(np.abs(pooled_flat - fine_flat.mean())) < 1e-3
print("constant-mean surface: pooled == unpooled: ok")

print("\nall checks passed")
