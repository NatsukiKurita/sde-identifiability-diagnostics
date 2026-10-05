#!/usr/bin/env bash
# Run the analysis pipeline except environment setup.
#
# Usage (from the repository root, after ./run_setup.sh):
#   ./run_overall_procedure.sh
#   ./run_overall_procedure.sh --setting setting_1
#   ./run_overall_procedure.sh --setting setting_1 --overwrite
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN=""
SETTING=""
OVERWRITE=0

usage() {
	cat <<'USAGE'
Usage:
  ./run_overall_procedure.sh [--setting SETTING] [--overwrite]

Runs every stage after environment setup, in order:
  01  scripts/01_simulation/00_prepare_toml.py
  01  scripts/01_simulation/01_run_simulation.py
  02  scripts/02_estimation/00_run_estimation.py
  03  scripts/03_diagnostics/00_execute_diagnostics.py
  04  scripts/04_figures/fig01_min_eigenvalue_timeseries.py
      scripts/04_figures/fig02_spaghetti.py
      scripts/04_figures/fig03_min_eigenvector.py
      scripts/04_figures/fig04_error_evolution.py
      scripts/04_figures/fig05_relative_distance.py
  05  scripts/05_supplemental_figures/01_example_trajectories.py
      scripts/05_supplemental_figures/00_method.py
      scripts/05_supplemental_figures/02_tk2020_n4_spaghetti.py
      scripts/05_supplemental_figures/03_kr2021_tandem_example.py

--setting limits stages that accept it (simulation, estimation, diagnostics,
and paper figures). Omitted means every setting in PARAMETER_SETTINGS.
Supplemental figures (example trajectories, method comparison, TK2020 N=4
spaghetti) use those scripts' own defaults. The method and TK2020 N=4
figures read results/setting_1.

--overwrite is forwarded to TOML preparation, simulation, estimation,
diagnostics, and example trajectories.

Environment:
  PYTHON   Python interpreter (default: <project-root>/.venv/bin/python)
  VENV_DIR Virtual environment directory (default: <project-root>/.venv)

Run ./run_setup.sh once before this script.
USAGE
}

resolve_python() {
	if [[ -n "${PYTHON:-}" ]]; then
		PYTHON_BIN="${PYTHON}"
		return
	fi
	PYTHON_BIN="${VENV_DIR}/bin/python"
	if [[ ! -x "${PYTHON_BIN}" ]]; then
		cat >&2 <<EOF

Error: Python interpreter not found at ${PYTHON_BIN}.
Run setup first, or set PYTHON to an interpreter that already has this project installed:

  ./run_setup.sh

EOF
		exit 1
	fi
}

run_py() {
	local label="$1"
	shift
	echo "==> ${label}"
	"${PYTHON_BIN}" "$@"
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--setting)
			if [[ $# -lt 2 || -z "${2}" ]]; then
				echo "Missing value for --setting" >&2
				usage
				exit 2
			fi
			SETTING="$2"
			shift 2
			;;
		--overwrite)
			OVERWRITE=1
			shift
			;;
		-h|--help|help)
			usage
			exit 0
			;;
		*)
			echo "Unknown argument: $1" >&2
			usage
			exit 2
			;;
	esac
done

resolve_python
cd "${PROJECT_ROOT}"

setting_args=()
if [[ -n "${SETTING}" ]]; then
	setting_args=(--setting "${SETTING}")
fi
overwrite_args=()
if [[ "${OVERWRITE}" -eq 1 ]]; then
	overwrite_args=(--overwrite)
fi

run_py "01 simulation: prepare TOML" \
	scripts/01_simulation/00_prepare_toml.py \
	${overwrite_args[@]+"${overwrite_args[@]}"}

run_py "01 simulation: run simulations" \
	scripts/01_simulation/01_run_simulation.py \
	${setting_args[@]+"${setting_args[@]}"} \
	${overwrite_args[@]+"${overwrite_args[@]}"}

run_py "02 estimation" \
	scripts/02_estimation/00_run_estimation.py \
	${setting_args[@]+"${setting_args[@]}"} \
	${overwrite_args[@]+"${overwrite_args[@]}"}

run_py "03 diagnostics" \
	scripts/03_diagnostics/00_execute_diagnostics.py \
	${setting_args[@]+"${setting_args[@]}"} \
	${overwrite_args[@]+"${overwrite_args[@]}"}

run_py "04 figure: minimum eigenvalue time series" \
	scripts/04_figures/fig01_min_eigenvalue_timeseries.py \
	${setting_args[@]+"${setting_args[@]}"}

run_py "04 figure: spaghetti" \
	scripts/04_figures/fig02_spaghetti.py \
	${setting_args[@]+"${setting_args[@]}"}

run_py "04 figure: minimum eigenvector contributions" \
	scripts/04_figures/fig03_min_eigenvector.py \
	${setting_args[@]+"${setting_args[@]}"}

run_py "04 figure: error evolution" \
	scripts/04_figures/fig04_error_evolution.py \
	${setting_args[@]+"${setting_args[@]}"}

run_py "04 figure: KR2021 relative distance" \
	scripts/04_figures/fig05_relative_distance.py \
	${setting_args[@]+"${setting_args[@]}"}

run_py "05 supplemental: example trajectories" \
	scripts/05_supplemental_figures/01_example_trajectories.py \
	${overwrite_args[@]+"${overwrite_args[@]}"}

run_py "05 supplemental: method comparison" \
	scripts/05_supplemental_figures/00_method.py

run_py "05 supplemental: TK2020 N=4 spaghetti" \
	scripts/05_supplemental_figures/02_tk2020_n4_spaghetti.py

run_py "05 supplemental: KR2021 tandem example" \
	scripts/05_supplemental_figures/03_kr2021_tandem_example.py

echo "==> Finished overall procedure."
