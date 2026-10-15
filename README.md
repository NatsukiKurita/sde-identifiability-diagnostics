# Diagnosing parameter unidentifiability in stochastic dynamics with an application to flocking models

This is the source code for the paper "Diagnosing parameter unidentifiability in stochastic dynamics with an application to flocking models".

## Workflow

Run commands from the repository root. Simulation output is written under `data/simulation`. Estimation, diagnostics, and figures are written under `results`.

### Prerequisites

- Python dependencies (`toml`, `jax`, `h5py`, and the rest of `src/requirements.txt`) are installed into `.venv` at the repository root.
- The simulation stage requires Julia.
- Run setup once before the analysis pipeline.

```bash
./run_setup.sh
```

`./run_setup.sh` runs both setup steps. Pass `python` or `julia` to run only one of them.

| Target | Script | What it does |
|--------|--------|----------------|
| `python` | `scripts/00_setup/00_setup_python_package.py` | Create `.venv` if needed, install `src/requirements.txt`, then `pip install -e .` |
| `julia` | `scripts/00_setup/01_setup_julia_env.py` | Build the Julia sysimage `src/flocking_models/build_tools/sys_simulation.so` used by the simulator |

#### Julia and system toolchain

The simulations were developed and tested with Julia 1.12.6. For reproducibility, install this version with [Juliaup](https://github.com/JuliaLang/juliaup) if Julia is not already available:

```bash
curl -fsSL https://install.julialang.org | sh
```

After the installation completes, restart the terminal so that the updated `PATH` takes effect. Then install and select Julia 1.12.6:

```bash
juliaup add 1.12.6
juliaup default 1.12.6
julia --version
```

On Ubuntu and Debian systems, install the standard compilation toolchain before building the Julia environment:

```bash
sudo apt update
sudo apt install -y build-essential
./run_setup.sh julia
```

The `build-essential` package provides the compiler, linker, and related system tools required by Julia's `PackageCompiler` to create the custom sysimage. Users of other operating systems should install an equivalent C/C++ build toolchain. This step is only necessary when setting up or rebuilding the Julia environment.

The sysimage build runs `src/flocking_models/build_tools/precompile_script.jl`, which simulates a few seconds of every Kinetics × Social pair in the parameter presets in both `Float32` and `Float64`. During this step, simulation logs will appear in the terminal; these are expected precompile runs and do not generate the pipeline's simulation data. The package code those runs use is compiled into the image, so simulations skip most JIT compilation at startup. The build needs roughly 9 GB of memory. If it fails, add swap space and rerun `./run_setup.sh julia`. Rebuild after changing `src/flocking_models/Manifest.toml` or adding a model to the presets.

To use a specific interpreter:

```bash
PYTHON=python3.11 ./run_setup.sh
```

Later stages use `.venv/bin/python` when `PYTHON` is unset. `run_overall_procedure.sh` also accepts `VENV_DIR` if the virtual environment lives somewhere else.

### Full pipeline

Settings are defined by the TOML presets in `src/experiment_configs/parameter_presets/`. Each `setting_*.toml` file becomes one setting named after its file stem (for example `setting_1`). Simulated population sizes are `NUM_AGENTS` in `src/experiment_configs/simulation.py` (currently 2 and 4).

After setup, `./run_overall_procedure.sh` runs every analysis stage in order. `--setting` is forwarded to simulation, estimation, diagnostics, and the paper figures. Without it, those stages use every setting in `PARAMETER_SETTINGS`. Supplemental scripts `00_method.py` and `02_n4_spaghetti.py` default to `setting_1`; run `04_truth_diagnostics.py` separately if you need truth-parameter versions of the fig01/fig02 layouts. `--overwrite` is forwarded to TOML preparation, simulation, estimation, diagnostics, and example trajectories.

```bash
./run_overall_procedure.sh
./run_overall_procedure.sh --setting setting_1
./run_overall_procedure.sh --setting setting_1 --overwrite
```

| Order | Script | Role |
|------:|--------|------|
| 01 | `scripts/01_simulation/00_prepare_toml.py` | Write simulation TOML configs |
| 01 | `scripts/01_simulation/01_run_simulation.py` | Run simulations and restore Delaunay neighbors |
| 02 | `scripts/02_estimation/00_run_estimation.py` | Multistart parameter estimation |
| 03 | `scripts/03_diagnostics/00_execute_diagnostics.py` | Hessian and eigenvalue diagnostics at the saved optimum and at the truth |
| 04 | `scripts/04_figures/fig01_min_eigenvalue_timeseries.py` | Minimum-eigenvalue time series |
| 04 | `scripts/04_figures/fig02_spaghetti.py` | Spaghetti charts of the minimum eigenvalue |
| 04 | `scripts/04_figures/fig03_degeneracy_scatter.py` | 2D scatter of estimates on the degenerate parameter pair (TK2020, KR2021) |
| 04 | `scripts/04_figures/fig04_error_evolution.py` | Parameter-precision time series |
| 04 | `scripts/04_figures/fig05_relative_distance.py` | KR2021 inter-individual distance distribution after burn-in |
| 05 | `scripts/05_supplemental_figures/01_example_trajectories.py` | Example trajectories |
| 05 | `scripts/05_supplemental_figures/00_method.py` | Method comparison of minimum-eigenvalue trajectories |
| 05 | `scripts/05_supplemental_figures/02_n4_spaghetti.py` | Minimum-eigenvalue spaghetti charts for every model with N=4 |

`scripts/02_estimation/01_error_evolution.py` is imported by the estimation script. It re-estimates parameters on cumulative post-burn-in prefixes and is not a separate command. `scripts/04_figures/_common.py` holds the shared figure arguments and plotting helpers.

### Individual stages

```bash
.venv/bin/python scripts/01_simulation/00_prepare_toml.py --overwrite
.venv/bin/python scripts/01_simulation/01_run_simulation.py --setting setting_1 --overwrite
.venv/bin/python scripts/02_estimation/00_run_estimation.py --setting setting_1 --overwrite
.venv/bin/python scripts/03_diagnostics/00_execute_diagnostics.py --setting setting_1 --overwrite
.venv/bin/python scripts/04_figures/fig01_min_eigenvalue_timeseries.py --setting setting_1
.venv/bin/python scripts/04_figures/fig02_spaghetti.py --setting setting_1
.venv/bin/python scripts/04_figures/fig03_degeneracy_scatter.py --setting setting_1
.venv/bin/python scripts/04_figures/fig04_error_evolution.py --setting setting_1
.venv/bin/python scripts/04_figures/fig05_relative_distance.py --setting setting_1
.venv/bin/python scripts/05_supplemental_figures/01_example_trajectories.py --overwrite
.venv/bin/python scripts/05_supplemental_figures/00_method.py
.venv/bin/python scripts/05_supplemental_figures/02_n4_spaghetti.py
.venv/bin/python scripts/05_supplemental_figures/04_truth_diagnostics.py --setting setting_1
```

Help for any stage:

```bash
.venv/bin/python scripts/01_simulation/01_run_simulation.py --help
```

### 01 Simulation

`00_prepare_toml.py` writes the TOML configs consumed by the simulator.

| Option | Description |
|--------|-------------|
| `--overwrite` | Overwrite existing TOML files |

`01_run_simulation.py` runs the simulations. Output goes under `data/simulation` unless `--simulation_root` is set. If the sysimage exists, Julia starts with `--sysimage`. Otherwise the script prints a notice and runs without it. It also warns when `Manifest.toml` is newer than the sysimage.

| Option | Description |
|--------|-------------|
| `--setting SETTING` | One setting, for example `setting_1`. Default: every setting in `PARAMETER_SETTINGS` |
| `--simulation_root PATH` | Directory that contains the `setting_*` folders. Default: `data/simulation` |
| `--overwrite` | Overwrite existing HDF5 output |
| `--skip-restore-delaunay` | Skip Delaunay-neighbor restoration after the simulation |
| `--no-sysimage` | Start Julia without the sysimage, even if it exists |

### 02 Estimation

`00_run_estimation.py` runs multistart estimation. It does not compute the Hessian. Each sample directory receives:

- `summary.json` — optimization summary (`mean_loss_*` and `mean_grad_*` are averaged over time steps and agents) and parameter estimates
- `parameter_evolution.json` — cumulative parameter estimates at fixed time intervals

| Option | Description |
|--------|-------------|
| `--setting SETTING` | One setting. Default: every setting in `PARAMETER_SETTINGS` |
| `--sim_root PATH` | Root of the simulation data. Default: `data/simulation` |
| `--overwrite` | Overwrite existing estimation results |

### 03 Diagnostics

`00_execute_diagnostics.py` evaluates the saved optimum. It updates `summary.json` with the minimum and maximum eigenvalues and writes:

- `result_time_series.h5` — shared `time_seconds` / `time_indices`, plus `hessians` (cumulative Hessian w.r.t. log-parameters, `diag(p) H diag(p) + diag(p * grad)`; may be indefinite), `gradients` (cumulative gradient w.r.t. the raw parameters), and `min_eigenvalues` time series under `results/optimum` and `results/truth`

Options match estimation: `--sim_root`, `--setting`, and `--overwrite`.

### 04 Paper figures

Each figure script reads `results/<setting>/` (minimum eigenvalues from `results/optimum` in `result_time_series.h5`) and writes PDF and PNG files under `results/<setting>/figures/`. Main-text plots use the N=2 samples (`MAIN_TEXT_NUM_AGENT` in `scripts/04_figures/_common.py`).

**Shared time-axis conventions** (fig01, fig02, and other scripts that call `downsample_min_eigenvalue_series` in `_common.py`):

- Data are shown from **t = 1 s** onward (`MIN_EIGENVALUE_PLOT_START_SECONDS`).
- Series are downsampled to **one point per second** (`MIN_EIGENVALUE_PLOT_INTERVAL_SECONDS = 1.0`).
- The horizontal axis runs up to **300 s** unless a script sets a narrower window.

| Script | Figure | Output files / directory |
|--------|--------|---------------------------|
| `fig01_min_eigenvalue_timeseries.py` | Mean (or representative) minimum-eigenvalue time series | `figures/min_eigenvalue_timeseries/identifiable.pdf` (Null, G2012, KV; log–log); `unidentifiable.pdf` (TK2020 plus KR2021 **fluctuation** samples, linear axes, fixed y-limits); `linear_KR2021.pdf` (KR2021 **linear** samples only). KR2021 linear vs fluctuation is split by whether the estimated `Social.m` is below or above the truth. KR2021 Inertial curves use representative samples where noted in `_common.py` (`SPECIAL_MIN_EIGENVALUE_SERIES`). |
| `fig02_spaghetti.py` | Spaghetti charts of the minimum eigenvalue for every social model and kinetics | `figures/spaghetti/`. Identifiable models (Null, G2012, KV): slope=1 guide. TK2020: \(\sqrt{t}\log t\) guide and a **linear–linear inset** over the last 10 s. KR2021 Overdamped: separate `*_linear` and `*_fluctuation` panels. KR2021 Inertial: **one merged panel** (`KR2021_Inertial.pdf`) with no guide and no inset. |
| `fig03_degeneracy_scatter.py` | Estimates on the degenerate parameter pair with the theoretical manifold and ground truth | `figures/degeneracy_scatter/TK2020.pdf` (linear axes), `KR2021.pdf` (log–log axes) |
| `fig04_error_evolution.py` | Worst relative error (max absolute log estimated/true ratio) vs time | `figures/error_plot/Overdamped/` and `figures/error_plot/Inertial/`, each with `error_plot_identifiable.pdf` and `error_plot_unidentifiable.pdf` |
| `fig05_relative_distance.py` | Post-burn-in distribution of \(d/d^{\mathrm{eq}}\) for KR2021 N=2 | `figures/relative_distance/KR2021.pdf` (x-axis limited to 4; narrow histogram bins) |

Shared options:

| Option | Description |
|--------|-------------|
| `--setting SETTING` | One setting. Default: every `setting_*` directory under the results root |
| `--results-root PATH` | Results root. Default: `results` |

### 05 Supplemental figures

| Script | What it plots | Default output |
|--------|----------------|----------------|
| `01_example_trajectories.py` | A short example trajectory for every simulation model | `results/04_supplemental_figures/method/example_trajectories/` |
| `00_method.py` | Minimum-eigenvalue trajectories for an identifiable sample and an unidentifiable sample | `results/04_supplemental_figures/method/` |
| `02_n4_spaghetti.py` | Spaghetti charts for **every** social model and kinetics at **N = 4** (same panel logic as fig02 via `fig02_spaghetti.plot_for_setting`; TK2020 at N=4 uses the slope=1 guide because cohesion and separation decouple) | `results/04_supplemental_figures/n4_spaghetti/<setting>/spaghetti/` |
| `04_truth_diagnostics.py` | Same layouts as fig01 and fig02, but minimum eigenvalues from `results/truth` in `result_time_series.h5`. **KR2021 is not split** by estimated-\(m\) regime (one panel per kinetics). **TK2020 spaghetti has no terminal linear inset.** Fig01 truth outputs: `identifiable.pdf`, `unidentifiable.pdf` (TK2020 only), `KR2021.pdf` (all KR2021 samples). | `results/04_supplemental_figures/truth_diagnostics/<setting>/` |

`00_method.py` reads `results/optimum` from `result_time_series.h5` (or a result directory or `summary.json` that points at it). Its built-in sample paths are under `results/setting_1`. It does not take `--setting`.

Example trajectories:

| Option | Description |
|--------|-------------|
| `--simulation-root PATH` | Simulation data root |
| `--output-root PATH` | Figure output root |
| `--sample-id ID` | Four-digit simulation sample id (default: `0001`) |
| `--duration SECONDS` | Trajectory length after burn-in (default: 15) |
| `--burn-in SECONDS` | Burn-in omitted from the plot |
| `--overwrite` | Overwrite existing PDF and PNG files |

Method comparison:

| Option | Description |
|--------|-------------|
| `--identifiable-path PATH` | Result directory, `summary.json`, or `result_time_series.h5` for the identifiable example |
| `--unidentifiable-path PATH` | Same, for the unidentifiable example |
| `--output-dir PATH` | Directory where the figure is written |
| `--time-limit SECONDS` | Maximum time on the x-axis (default: 60) |
| `--plot-to-end` | Plot the full trajectory instead of stopping at `--time-limit` |

N=4 spaghetti (`02_n4_spaghetti.py`):

| Option | Description |
|--------|-------------|
| `--setting SETTING` | Setting to read (default: `setting_1`) |
| `--results-root PATH` | Results root. Default: `results` |
| `--output-dir PATH` | Output root; figures go under `<output-dir>/<setting>/spaghetti/` |

Truth diagnostics (`04_truth_diagnostics.py`):

| Option | Description |
|--------|-------------|
| `--setting SETTING` | One setting. Default: every `setting_*` under the results root |
| `--results-root PATH` | Results root. Default: `results` |
| `--output-dir PATH` | Output root. Default: `results/04_supplemental_figures/truth_diagnostics` |
