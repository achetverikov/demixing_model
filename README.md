# Demixing Model

This repo implements the modeling pipeline for the Demixing Model (Chetverikov, 2023; Chetverikov & Hansmann-Roth, 2025). The core idea is that memory biases (attraction vs. repulsion) emerge as optimal and inevitable when the brain disentangles overlapping memory signals. The pipeline uses simulations to estimate the optimal solution for the signal disentanglement problem based on the amount of noise and similarity between the items for a two-item case.

## Processing Steps

1) Simulate bias distributions for dual-component memories on parameter grids (`surface_computation/`).
2) Build averaged likelihood surfaces from simulations (`neural_network_optimization/`).
3) Train a mirror-aware neural network surrogate to predict surfaces (`neural_network_optimization/`).
4) Fit model parameters to behavioral data with grid-based optimization (`model_fit_to_data/`).
5) Optional: generate plots/exports or run the Python/R prediction simulator (`model_fit_to_data/`, `surface_simulator_for_predictions/`).

## Requirements

- Python 3 environment with the ability to install packages (virtualenv or conda recommended).
- NVIDIA GPU with a CUDA-enabled JAX build (`jax`/`jaxlib`; see JAX and CUDA docs for instructions); CPU runs are likely possible but _extremely_ slow.
- Core Python packages used across the pipeline: `flax`, `optax`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `pyarrow`.
- Surface browser dependencies: `streamlit` and `plotly` (see browser requirement below).
- Optional R interface: `arrow`, `stringr`, and `data.table` for `surface_simulator_for_predictions/surface_simulator.R`.

## Browser Requirement

- To explore surfaces interactively, install `streamlit` and `plotly`, then launch `streamlit run surface_browser/main_app.py` in a modern desktop browser (Chrome/Edge/Firefox). The app expects averaged surfaces (e.g., `averaged_surfaces_10k_20samples` by default but the folder can be selected in the app) to be present.

## Repo Map

- `surface_computation/`: Simulation and likelihood surface generation.
- `neural_network_optimization/`: Averaging surfaces + NN training.
- `model_fit_to_data/`: Parameter fitting and post-fit plots/exports.
- `surface_simulator_for_predictions/`: Python/R helpers for prediction surfaces and curves.
- `surface_browser/`: Interactive app for exploring surfaces and fitted results.
- `tests/`: Smoke pipeline and test instructions.
- `example_data/`: Sample input data.

## Installation

### Option A: Dev Container (recommended)

The repo ships a VS Code Dev Container that provides a fully configured environment with CUDA-enabled JAX, R, and all required Python packages.

**Prerequisites**

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine on Linux) with the NVIDIA Container Toolkit installed and GPU access enabled.
- VS Code with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.

**Steps**

1. Clone the repo and open it in VS Code.
2. When prompted, click **Reopen in Container** (or run **Dev Containers: Reopen in Container** from the command palette).
3. VS Code pulls `andreychetverikov/demixing-jax-r:latest`, mounts the workspace, and runs `postCreateCommand` which:
   - Creates `/workspaces/.venv` (inheriting global site-packages so CUDA JAX is available immediately).
   - Installs or upgrades any missing/mismatched Python packages listed in `.devcontainer/install_python_packages_if_missing.py`.
   - Installs R packages listed in `.devcontainer/install_r_packages_if_missing.R`.

The Python interpreter is pre-configured to `/workspaces/.venv/bin/python`.

**Rebuilding the Docker image**

The image is built from `.devcontainer/Dockerfile.jax-r`. To rebuild and push after changing it:

```bash
docker build -f .devcontainer/Dockerfile.jax-r -t andreychetverikov/demixing-jax-r:latest .
docker push andreychetverikov/demixing-jax-r:latest
```

---

### Option B: Manual installation

1) **Clone or download**  
   ```bash
   git clone https://github.com/<your-org>/demixing_model.git
   cd demixing_model
   ```
   (Alternatively download the ZIP and unzip to a folder, then `cd` there.)

2) **Create and activate a Python env (3.10+)**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3) **Install dependencies (reads `pyproject.toml` with a list of dependencies automatically when you are in the repo folder)**  
   ```bash
   pip install --upgrade pip
   pip install .
   ```
   If you have a CUDA GPU, install a CUDA-enabled `jaxlib` wheel that matches your CUDA version (see https://github.com/google/jax#installation) so JAX runs on GPU instead of CPU.

4) **Verify with the smoke pipeline** (optional but recommended)  
   ```bash
   bash tests/run_smoke_pipeline.sh
   ```

## Pipeline Docs

- `surface_computation/Likelihood_Surface_Pipeline_Documentation.md`
- `neural_network_optimization/Neural_Network_Optimization_Pipeline_Documentation.md`
- `model_fit_to_data/Batch_Fit_Analysis_Pipeline_Documentation.md`

## Replicating the Paper Results

1) Run the full simulation grid and surface generation in `surface_computation/` using the pipeline doc.
2) Create averaged surfaces and train the mirror-aware NN in `neural_network_optimization/`.
3) Fit the model to the experimental datasets in `model_fit_to_data/`.
4) Generate unified subject plots and CSV exports for figures/tables.
5) Optionally use `surface_simulator_for_predictions/` to reproduce prediction curves for specific parameter sweeps.

Each step is described in the pipeline docs above; start from the repo root and follow the folder-specific instructions in order.

## Quick Start

Run the smoke pipeline from the repo root:

```bash
bash tests/run_smoke_pipeline.sh
```

## Disclaimer

The documentation for the project was generated using AI and may contain errors. This code is provided as-is without warranty of any kind. The author assumes no liability for any damages or consequences arising from its use. 

## References

Chetverikov, A. (2023). Demixing model: A normative explanation for inter-item biases in memory and perception. *bioRxiv*. https://doi.org/10.1101/2023.03.26.534226

Chetverikov, A., & Hansmann-Roth, S. (2025). Noise in Competing Representations Determines the Direction of Memory Biases (p. 2025.12.17.694673). *bioRxiv*. https://doi.org/10.64898/2025.12.17.694673
