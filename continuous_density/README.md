# continuous_density transition namespace

This directory is being retired.

Maintained WNM components now live in functional locations:

- runtime model and artifact format: `shared/wnm.py`
- prediction/runtime API: `shared/prediction.py`
- fitting/scoring: `model_fit_to_data/`
- simulation and training-data generation: `surface_computation/`
- WNM training and packaging: `surrogate_training/wnm/`

Files that remain here are compatibility shims or research workflows awaiting
relocation. Historical findings and the former transition README are archived
under `docs/history/wnm_transition/`.

No maintained production module should import from `continuous_density`.
