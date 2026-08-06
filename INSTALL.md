# Installation

The Demixing Model requires Python 3.10 or newer. An NVIDIA GPU is strongly recommended for fitting and required for practical full-surface generation. The repository provides GPU and CPU development containers as well as a manual installation route.

If you do not usually configure Python or CUDA environments, use the development container. A development container is a prepared workspace that installs the correct software versions for you when the repository opens in VS Code.

## Which setup should I use?

| Your computer | Recommended setup | What to expect |
|---|---|---|
| NVIDIA GPU | [GPU development container](#gpu-development-container-recommended) | Best option for fitting datasets |
| Mac or no NVIDIA GPU | [CPU development container](#cpu-development-container) | Predictions and tests work; fitting is much slower |
| Existing scientific Python/CUDA setup | [Manual installation](#manual-python-installation) | More control, but you must manage compatible software versions |

## GPU development container (recommended)

Prerequisites:

- Docker Desktop, or Docker Engine on Linux;
- an NVIDIA GPU with an up-to-date host driver;
- VS Code with the Dev Containers extension.

On **Windows**, use Docker Desktop with its WSL 2 backend. Update WSL with `wsl --update` and install a current NVIDIA Windows driver that supports GPU access through WSL 2. You do not need to install the CUDA Toolkit on Windows, a Linux NVIDIA driver inside WSL, or the NVIDIA Container Toolkit separately; the CUDA software used by the model is provided by the development container. See the [Docker Desktop GPU requirements](https://docs.docker.com/desktop/features/gpu/) and [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

On **Linux**, install Docker Engine, a compatible NVIDIA driver, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) so Docker can pass the GPU through to the container.

On either system, you can check that Docker can access the GPU before opening the development container:

```bash
docker run --rm --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

Download the repository using **Code → Download ZIP** on GitHub and extract it, or clone it with Git:

```bash
git clone https://github.com/achetverikov/demixing_model.git
cd demixing_model
```

Open the resulting folder in VS Code and run **Dev Containers: Reopen in Container**. The first opening downloads the prepared environment and can take several minutes.

The default configuration, `.devcontainer/devcontainer.json`, uses `andreychetverikov/demixing-jax-r:latest`. It installs the project dependencies, R packages, and selects `/workspaces/.venv/bin/python`. Confirm that JAX sees the GPU:

```bash
/workspaces/.venv/bin/python -c "import jax; print(jax.devices())"
```

At least one reported device should have `platform='gpu'`.

## CPU development container

On macOS or a machine without an NVIDIA GPU, select `.devcontainer/cpu/devcontainer.json` when opening the folder in a container. With the Dev Container CLI, the equivalent command is:

```bash
devcontainer up --workspace-folder . \
  --config .devcontainer/cpu/devcontainer.json
```

This configuration uses `andreychetverikov/demixing-jax-r:cpu` and supports both amd64 and arm64 hosts. It is suitable for editing, tests, R analysis, and small prediction batches. Model fitting is substantially slower: a measured one-model, `density`-only fit of the five demo participant groups took 74m37s on an Intel Core Ultra 9 285 CPU and 6m5s on an RTX 5080 GPU. The complete demo does more work than this benchmark because it runs two checkpoints and five objectives.

## Manual Python installation

Create an environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

This installs the CPU JAX runtime and the dependencies declared in `pyproject.toml`. For a host with a CUDA 13 installation, install the CUDA extra instead:

```bash
python -m pip install ".[cuda]"
```

The NVIDIA driver, CUDA libraries, and JAX wheel must be mutually compatible. Check the selected runtime before starting a fit:

```bash
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

Commands in the documentation assume they are run from the repository root. Where shown, `PYTHONPATH=.` makes the repository modules importable without installing them as a package.

## Optional R interface

Prediction helpers in `surface_simulator_for_predictions/surface_simulator.R` use `arrow`, `stringr`, and `data.table`. Install them in an R session if they are not already available:

```r
install.packages(c("arrow", "stringr", "data.table"))
```

## Verification

Start with the focused tests, which do not regenerate the full model:

```bash
PYTHONPATH=. python -m pytest \
  tests/test_config_import.py \
  tests/test_flag_dispatch.py \
  tests/test_lock_backend.py \
  tests/test_object_store.py \
  tests/test_pooled_bwcrps_export.py
```

The smoke scripts additionally exercise simulation, averaging, network training, and fitting. They are compute-heavy and should normally run on a GPU:

```bash
PYTHON_BIN=python bash tests/run_smoke_pipeline.sh
PYTHON_BIN=python bash tests/run_smoke_standard.sh
PYTHON_BIN=python bash tests/run_smoke_compare_seeds.sh
```

`tests/run_smoke_pipeline.sh` refuses CPU execution unless `ALLOW_CPU=1` is set.

## Troubleshooting

### JAX reports only a CPU

Run the device check above in the same environment used for the model. In the development container, also confirm that `nvidia-smi` works. For a manual setup, reinstall the CUDA extra after checking the installed NVIDIA driver and CUDA 13 libraries.

### CUDA out-of-memory errors

Do not run multiple fitting or surface-generation jobs on one GPU. Stop other GPU processes first. The GPU container already sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`; the same variable can be exported in a manual environment if JAX preallocation conflicts with other processes.

### Fitting appears to restart or skip work

Fitting resumes by default from `<output-dir>/extended_progress.json`. Use a new output directory or pass `--no-resume` for an intentional fresh run. Do not mix results produced with different checkpoints or circular-space settings.

### Generated artifacts are not where expected

Relative fit outputs are placed under `results/`. Surface tools also honor `DEMIXING_ARTIFACT_ROOT`; without it, their default artifact root is the repository-local `results/` directory.

## Maintainer: rebuild container images

```bash
# GPU image (CUDA, amd64)
docker build -f .devcontainer/Dockerfile.jax-r \
  -t andreychetverikov/demixing-jax-r:latest .
docker push andreychetverikov/demixing-jax-r:latest

# CPU image (amd64 and arm64)
docker buildx build --platform linux/amd64,linux/arm64 \
  -f .devcontainer/Dockerfile.jax-r-cpu \
  -t andreychetverikov/demixing-jax-r:cpu --push .
```
