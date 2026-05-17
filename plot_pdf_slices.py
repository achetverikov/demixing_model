"""
Plot PDF slices at small feature differences to illustrate why density fitting
can yield negative E[mu1_bias] while asymmetry is positive.
"""

import sys
sys.path.insert(0, '/workspaces/demixing_model')
sys.path.insert(0, '/workspaces/demixing_model/neural_network_optimization')

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from shared.utils import load_checkpoint
from shared.config import config

CHECKPOINT   = '/workspaces/demixing_model/pretrained/model_epoch_1500.pkl'
PARAMS_CSV   = '/workspaces/demixing_model/results/fritsche/csv_exports/fitted_parameters.csv'
DATA_CSV     = '/workspaces/demixing_model/example_data/fritsche_prepared.csv'
WEIGHTS_SD   = 20  # Gaussian SD (degrees) for feat_diff weighting — same for model and empirical

# Feature difference slices to show (degrees)
FEAT_DIFFS = [4, 10, 18, 30]

# ── load model ────────────────────────────────────────────────────────────────
state, _ = load_checkpoint(CHECKPOINT)

def predict(sd_feat1, sd_feat2, sd_spat):
    """Return log-surface (n_mu1, n_feat_diff) for one parameter set."""
    params = jnp.array([[sd_feat1, sd_feat2, sd_spat]])
    return state.apply_fn(state.params, params)[0]  # shape (n_mu1, n_feat_diff)

# ── grids ──────────────────────────────────────────────────────────────────────
mu1_grid  = np.array(config.create_grid('mu1_bias'))   # -180..180, step 2
feat_grid = np.array(config.create_grid('feat_diff'))  # 0..180,   step 2

# ── load empirical data ────────────────────────────────────────────────────────
raw = pd.read_csv(DATA_CSV)
raw_e1 = raw[(raw['expName'] == 'Fritsche_E1') & (raw['condition'] == 1) &
             (raw['is_outlier'] == 0)].copy()

def empirical_slice_density(subject_id, target_fd, bias_grid, weights_sd=WEIGHTS_SD):
    """Weighted KDE of bias values for one subject around a target feat_diff."""
    d = raw_e1[raw_e1['subject'] == subject_id]
    fd   = d['abs_td_dist'].values
    bias = d['bias_to_distr_corr'].values

    # Gaussian weights in feat_diff dimension, normalised to sum=1
    w = np.exp(-0.5 * ((fd - target_fd) / weights_sd) ** 2)
    if w.sum() == 0:
        return np.zeros_like(bias_grid, dtype=float)
    w = w / w.sum()

    # Bandwidth via Silverman's rule on the (unweighted) bias distribution
    bias_std = bias.std()
    iqr      = np.percentile(bias, 75) - np.percentile(bias, 25)
    bw = 0.9 * min(bias_std, iqr / 1.34) * len(bias) ** (-0.2)
    bw = max(bw, 1.0)  # floor to avoid degenerate kernel

    # Weighted KDE evaluated on bias_grid
    diff    = bias_grid[:, None] - bias[None, :]       # (n_grid, n_trials)
    kernels = np.exp(-0.5 * (diff / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
    density = (kernels * w[None, :]).sum(axis=1)       # weighted sum
    return density

# ── pick representative subjects from Fritsche E1, density fit ────────────────
df = pd.read_csv(PARAMS_CSV)
e1d = df[(df['experiment'] == 'Fritsche_E1') & (df['optimizer'] == 'density') &
         (df['condition'] == 1)].copy()

# Use subjects that actually show the dissociation: negative E but positive asym at small fd
# (subjects 21, 22, 23 identified from fitted_curves.csv)
target_subjs = [22, 23, 21]

def row_label(r):
    return (f"Subj {int(r.subject)}\n"
            f"(sd1={r.sd_feat1:.0f}, sd2={r.sd_feat2:.0f}, sdS={r.sd_spat:.0f})")

cases = {
    row_label(e1d[e1d['subject'] == s].iloc[0]):
        (s, tuple(e1d[e1d['subject'] == s][['sd_feat1', 'sd_feat2', 'sd_spat']].iloc[0]))
    for s in target_subjs
}

# ── compute surfaces + summaries ───────────────────────────────────────────────
# bias_grid for empirical KDE — finer than mu1_grid, same range
bias_plot_grid = np.linspace(-90, 90, 361)

def get_slices(subj_id, sd1, sd2, sdS, feat_diffs):
    log_surf = np.array(predict(sd1, sd2, sdS))  # (n_mu1, n_feat_diff)
    # Convert to normalised probability columns once
    surf = np.exp(log_surf - log_surf.max(axis=0, keepdims=True))
    surf = surf / (surf.sum(axis=0, keepdims=True) * config.mu1_bias_step)

    slices = {}
    for fd in feat_diffs:
        # Gaussian weights over feat_diff axis — same weighting as empirical
        w = np.exp(-0.5 * ((feat_grid - fd) / WEIGHTS_SD) ** 2)
        w = w / w.sum()
        prob = surf @ w   # weighted average of surface columns, shape (n_mu1,)

        E     = np.sum(mu1_grid * prob) * config.mu1_bias_step
        p_pos = np.sum(prob[mu1_grid > 0]) * config.mu1_bias_step
        p_neg = np.sum(prob[mu1_grid < 0]) * config.mu1_bias_step
        asym  = p_pos - p_neg
        emp   = empirical_slice_density(subj_id, fd, bias_plot_grid)
        slices[fd] = dict(prob=prob, E=E, asym=asym, actual_fd=fd, emp=emp)
    return slices

all_slices = {label: get_slices(subj_id, *params, FEAT_DIFFS)
              for label, (subj_id, params) in cases.items()}

# ── plot ───────────────────────────────────────────────────────────────────────
n_cases = len(cases)
n_fd    = len(FEAT_DIFFS)

fig = plt.figure(figsize=(4 * n_fd, 3.5 * n_cases))
gs  = gridspec.GridSpec(n_cases, n_fd, hspace=0.55, wspace=0.35)

cmap = plt.cm.Blues

for row, (label, slices) in enumerate(all_slices.items()):
    for col, fd in enumerate(FEAT_DIFFS):
        ax = fig.add_subplot(gs[row, col])
        s  = slices[fd]

        color_fill = cmap(0.45)
        ax.fill_between(mu1_grid, s['prob'], alpha=0.6, color=color_fill)
        ax.plot(mu1_grid, s['prob'], color=cmap(0.8), lw=1.2)

        # Shade positive / negative regions
        ax.fill_between(mu1_grid, s['prob'],
                        where=(mu1_grid > 0), alpha=0.35, color='forestgreen',
                        label='bias > 0')
        ax.fill_between(mu1_grid, s['prob'],
                        where=(mu1_grid < 0), alpha=0.35, color='tomato',
                        label='bias < 0')

        # Empirical density (scale to match model density height)
        emp      = s['emp']
        mod_prob = s['prob']
        if emp.max() > 0 and mod_prob.max() > 0:
            emp_scaled = emp * (mod_prob.max() / emp.max())
        else:
            emp_scaled = emp
        ax.plot(bias_plot_grid, emp_scaled, color='darkorange', lw=1.5,
                linestyle='-', alpha=0.85, label='data (KDE, scaled)')

        # Mark E[mu1_bias]
        ax.axvline(s['E'], color='black', lw=1.5, linestyle='--', label=f"E = {s['E']:.1f}°")
        ax.axvline(0,      color='gray',  lw=0.8, linestyle=':')

        ax.set_xlim(-90, 90)
        ax.set_xlabel('mu1 bias (°)', fontsize=9)
        if col == 0:
            ax.set_ylabel('Density', fontsize=9)
            # Case label on the left
            ax.annotate(label, xy=(-0.38, 0.5), xycoords='axes fraction',
                        fontsize=8, ha='center', va='center', rotation=90,
                        annotation_clip=False)

        sign = '+' if s['asym'] >= 0 else ''
        dissoc = (s['E'] < 0 and s['asym'] > 0) or (s['E'] > 0 and s['asym'] < 0)
        marker = '  ★' if dissoc else ''
        title = (f"feat_diff ≈ {int(s['actual_fd'])}°\n"
                 f"E = {s['E']:.1f}°  |  asym = {sign}{s['asym']:.3f}{marker}")
        ax.set_title(title, fontsize=8.5,
                     color='darkred' if dissoc else 'black')

        ax.tick_params(labelsize=8)
        if col == 0 and row == 0:
            ax.legend(fontsize=7, loc='upper right', framealpha=0.7)

        # On the first dissociation cell (row=0, col=1), annotate the key mechanism
        if row == 0 and col == 1 and dissoc:
            ymax = ax.get_ylim()[1]
            ax.annotate('most mass\n(asym > 0)', xy=(6, ymax * 0.55),
                        fontsize=7, color='darkgreen', ha='left')
            ax.annotate('', xy=(s['E'], ymax * 0.35),
                        xytext=(s['E'] - 18, ymax * 0.5),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
            ax.text(s['E'] - 20, ymax * 0.52, 'mean pulled\nby left tail',
                    fontsize=7, color='black', ha='right', va='bottom')

fig.suptitle(
    "PDF slices p(mu1_bias | feat_diff) — density-fitted model, Fritsche E1 cond 1\n"
    "Green = p(bias > 0), Red = p(bias < 0), dashed = E[mu1_bias]  |  ★ = dissociation (E < 0, asym > 0)",
    fontsize=10, y=1.01
)

out = '/workspaces/demixing_model/results/fritsche/pdf_slices_density_fit.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved → {out}")
plt.show()
