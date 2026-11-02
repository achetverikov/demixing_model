# How the Demixing Model works

The Demixing Model explains inter-item biases through the problem of separating overlapping, noisy representations. An observer receives uncertain evidence about two items and estimates the feature of each. Depending on the feature difference and the relative uncertainty of the items, those estimates show attraction or repulsion.

This guide follows the model from internal evidence to behavioral predictions and fitted parameters. For runnable examples, start with the [README](../README.md).

## The observer model

Each item has two coordinates:

- a **feature**, such as orientation or color, which the observer reports;
- an **identifiability coordinate**, such as location or timing, which helps distinguish the items.

The feature coordinate is circular, with a 360° period in model units. The identifiability coordinate is linear. Each item's evidence follows a two-dimensional distribution with independent noise on the two axes: a wrapped normal on the feature axis and a Gaussian on the identifiability axis.

The parameters describe the uncertainty in this evidence:

| Parameter | Meaning |
|---|---|
| `sd_feat1` | Target-item feature noise |
| `sd_feat2` | Competing-item feature noise |
| `sd_idf` | Identifiability noise, shared by the two items |
| `feat_diff` | True feature difference between the items |

True feature means lie at `−feat_diff/2` and `+feat_diff/2`. The identifiability means are separated by 42 model units, giving identifiability sensitivity `d′ = 42 / sd_idf`.

### One simulated trial

1. Draw `n_samples` internal evidence samples from the two-item mixture. Each sample has equal probability of coming from either item, giving a binomial allocation of the total sample count.
2. Fit a two-component mixture to the evidence using expectation–maximization (EM).
3. Match the estimated components to the true items using their identifiability coordinates.
4. Read out each component's estimated feature mean and calculate its angular error.

Errors are signed relative to the other item: **positive means attraction; negative means repulsion**. Repeating the simulation gives a distribution of response errors for each combination of noise parameters and feature difference.

The packaged observer models use 20 or 100 internal evidence samples per simulated trial. This choice sets the amount of evidence available to the observer. The separate `n_simulations` setting determines how many repetitions are used to estimate the response distribution during training-data generation.

### Inference within a simulated trial

EM estimates the component means, diagonal noise parameters, and mixture weights. The standard simulator uses:

| Setting | Value |
|---|---|
| Initializations | Structured (default): a deterministic data-derived bank of min(N, 20) + 34 starts from hard splits and local neighbourhoods; random: 225 starts. Retain the best final likelihood |
| Initial means (random starts) | Randomly selected evidence samples |
| Initial SDs (random starts) | Uniformly sampled from 0.3–2 times the empirical SD |
| Initial weights | 0.5 for each component (random starts); structured starts use each split's assignment share |
| Fitted weights | Constrained to 0.1–0.9 |
| Convergence | Relative likelihood change ≤ `1e-6` and normalized parameter change ≤ `1e-5` |
| Iteration limits | 5,000 updates for every start; capped selected winners continue to 25,000 and are reselected |

Likelihood EM uses soft component and winding probabilities. The feature
likelihood uses three spatial images below SD 60° and six Fourier harmonics
above that threshold; its conditional winding moments update feature means and
SDs. Angular errors are wrapped to the circle. Identifiability calculations use
linear means and differences. The default CUDA backend adds safeguarded
SQUAREM acceleration and an N=500 density shortcut; see the
[simulation guide](../surface_computation/README.md) for execution settings and
numerical differences. Every converged start stops immediately. Convergence
checks reuse the statistics from each update, without another likelihood pass.

## From simulations to a fast predictor

Behavioral fitting requires many evaluations at different noise values. A trained **conditional wrapped-normal mixture (WNM)** approximates the simulated response distribution and makes these evaluations practical.

```text
Noise parameters + stimulus difference
                  │
                  ▼
       Simulate internal evidence
                  │
                  ▼
    Demix evidence and record bias
                  │
                  ▼
      Train conditional WNM predictor
                  │
          ┌───────┴────────┐
          ▼                ▼
   Generate curves    Fit behavioral data
```

### Training data and symmetry

The training tools sample parameter combinations and store per-cell histograms of the simulated feature-bias outcomes for both items (by default) or the raw outcomes. Swapping `sd_feat1` and `sd_feat2` exchanges the target and competing item, so both components contribute training examples through this symmetry.

See [simulation and training-data generation](../surface_computation/README.md) for commands and data formats.

### Predictor architecture

The predictor takes `[sd_feat1, sd_feat2, sd_idf, feat_diff]` and outputs mixture weights, circular means, and scales for the target's response-error distribution:

`p(bias | parameters, feat_diff) = Σₖ weightₖ × WrappedNormal(bias; meanₖ, scaleₖ)`.

Both included models have 12 mixture components, hidden layers of widths 128, 256, and 256, and a minimum component scale of 0.25 model degrees. Training minimizes the negative log-likelihood of simulated biases. The resulting mixture supports direct evaluation of density, circular moments, and attraction–repulsion asymmetry.

The predictor's 12 components describe the shape of the response-error distribution. The observer simulation uses two components to represent the two items.

The declared parameter domain is:

| Input | Range in model degrees |
|---|---|
| `sd_feat1`, `sd_feat2` | 2.5–200 |
| `sd_idf` | 5–200 |
| `feat_diff` | 0.5–180 |

The [pretrained model reference](../pretrained/README.md) records checkpoint identity, training configuration, and domain details. A custom predictor should be validated against independent simulations, comparing response distributions and bias curves across dissimilarity.

## Predictions and response noise

For any supported noise values, the predictor yields a conditional response distribution at each stimulus difference. Useful summaries are:

- **circular mean bias**, describing the direction and magnitude of the average response error;
- **density asymmetry**, describing the balance of probability toward versus away from the competing item;
- **circular SD**, describing response variability.

Optional motor noise represents variability introduced when reporting the response. It is an independent wrapped normal with SD `sd_motor`. Convolving it with the predicted mixture changes each component scale to `sqrt(scale² + sd_motor²)`. This broadens the response distribution while preserving its circular mean.

## Fitting behavioral data

### Parameter sharing

For each participant and experiment, the fitter estimates all conditions jointly:

| Scope | Parameters |
|---|---|
| Each condition | `sd_feat1`, `sd_feat2` |
| Shared across conditions | `sd_idf` |
| Shared when motor noise is enabled | `sd_motor` |

Motor noise is fixed at zero by default. The internal sample count is chosen before fitting by selecting the 20- or 100-sample model.

### Continuous optimization

Fitting minimizes the sum of the condition losses using bounded L-BFGS-B in log-transformed noise parameters. The default search uses 64 deterministic starting points in four sequential batches of 16. It retains the minimum finite rescored loss, with convergence recorded as a diagnostic, and records all starting-point outcomes, the spread of converged losses, and parameters at their bounds. Batch size is part of the run fingerprint; changing it requires a separate run rather than mixing searches in one result file.

Distributional scores integrate reporting cells in chunks of five, sharing adjacent Gaussian CDF boundaries. Differences are taken before summing wrap images to preserve float32 precision for small cell masses.

Likelihood fits pass participant data dynamically to the optimizer and pad each condition's trial arrays to a shared multiple-of-64 capacity. A mask excludes padding from the loss and gradient, preserving every observed stimulus difference and response while allowing groups to reuse a compiled solver.

Feature and identifiability bounds come from the selected predictor. When motor noise is estimated, its upper bound is the smallest condition error SD times 1.1, capped at 50 model degrees, with a lower bound of 0.1°.

The optimizer uses float32 arithmetic with the `highest` matrix-multiplication precision setting. Its settings are recorded with the run so fitting and subsequent scoring use the same numerical configuration.

### Fitting criteria

The criterion determines which aspects of the observations guide parameter estimation. Each selected criterion produces its own fitted parameter set.

| Criterion | Target and loss |
|---|---|
| `density` (default) | Density-asymmetry curve; minimize `1 − CCC`, Lin's concordance loss |
| `smoothed_exp` | Smoothed circular-mean bias curve; minimize weighted squared angular error |
| `likelihood` | Observed trial errors; minimize summed negative log-density |
| `crps` | Trial-level distributions; minimize a circular-distance energy score |
| `balanced_crps` | Conditional distributions; weight supported dissimilarities equally |
| `bias_weighted_crps` | Conditional distributions; weight dissimilarities by empirical bias |

CCC measures agreement in shape, amplitude, and offset. A constant empirical asymmetry curve produces an error because it cannot support this fitting criterion.

Mean-bias criteria are skipped when estimating motor noise: symmetric response noise preserves the circular mean, so these criteria cannot identify its SD.

### Matching empirical and predicted summaries

A smoothed empirical curve reflects both the responses and the stimulus differences sampled by the experiment. The fitting pipeline applies the same averaging weights to the model predictions:

- For `smoothed_exp`, it pools model circular moments at the observed stimulus differences and compares their angles with the pooled empirical moments.
- For `density`, it smooths the model response distribution with the empirical bias-axis KDE bandwidth and pools signed probability mass with the empirical dissimilarity weights.
- For the balanced and bias-weighted scores, it pools model response probabilities with the same dissimilarity weights as the empirical conditional distributions.

The default dissimilarity weights use a Gaussian with nominal SD 20° in model units. Likelihood and plain CRPS evaluate individual trials directly.

This distinction matters when interpreting figures: a theoretical curve shows predictions at chosen stimulus differences, while a fitted summary incorporates the experiment's stimulus distribution and averaging weights.

### Angular units and trial selection

The model's angular period is 360°. For an experiment with period `P`, input angles are multiplied by `360/P`. Orientation data use `P = 180`, so both feature differences and errors are doubled for fitting. Plots and explicitly labelled physical-unit export columns convert back to the study scale.

CSV fitting uses the supplied experiment, participant, condition, stimulus-difference, and error columns. Flagged outliers are excluded by default, and conditions need at least 30 usable trials. A compiled `contextual_biases_database` bundle supplies trial selection, geometry, empirical targets, and row identities shared across model families.

## Reading a fitted result

Assess the parameter estimates together with the predicted distributions and curves across stimulus dissimilarity. Attraction and repulsion can change sign and size along that axis, so a single mean bias across all trials loses the pattern the model is intended to explain.

For a reproducible analysis, report the observer sample count, fitting criterion, angular period, trial-selection rules, parameter-sharing scheme, motor-noise setting, and optimization settings. Inspect boundary estimates and agreement across starting points when assessing the fit.

The saved run fingerprint identifies the data, trained model, and numerical settings. Export and plotting tools read it alongside the fitted results. See the [fitting guide](../model_fit_to_data/Batch_Fit_Analysis_Pipeline_Documentation.md) for commands and output files.

## Notes for developers

The main implementation references are:

- `surface_computation/jax_fit_main.py` and `jax_fit_functions.py`: observer simulation and EM;
- `surface_computation/wnm_simulation.py`: training-data simulation and item-swap augmentation;
- `surrogate_training/wnm/`: training and packaging;
- `shared/wnm.py`, `surrogate.py`, and `prediction.py`: mixture evaluation and model loading;
- `model_fit_to_data/continuous_fit.py`, `continuous_optimizer.py`, and `wnm_scoring.py`: joint parameter fitting and scoring.

Training corpora and independent validation results are generated artifacts under `$DEMIXING_ARTIFACT_ROOT`; they are not shipped and are not expected in a normal checkout. The earlier surface-pipeline explanation is preserved under `$DEMIXING_ARTIFACT_ROOT/exploration_archive/documentation_20260926/`.
