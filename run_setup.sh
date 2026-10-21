#!/usr/bin/env bash
# Run both setup scripts in scripts/00_setup/.
#
# Usage (from anywhere):
#   ./run_setup.sh
#   ./run_setup.sh python   # Python only
#   ./run_setup.sh julia    # Julia only (requires pip install -e .)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/00_setup"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN=""
# Tested toolchain versions; keep in sync with src/paths.py and README.md
PYTHON_VERSION="3.10.12"
JULIA_VERSION="1.12.6"
PYTHON_MINOR="${PYTHON_VERSION%.*}"

usage() {
	cat <<'USAGE'
Usage:
  ./run_setup.sh [all]
  ./run_setup.sh python
  ./run_setup.sh julia

Steps:
  python  - create .venv (if needed), pip install -r src/requirements.txt, pip install -e .
  julia   - 01_setup_julia_env.py       (build Julia sysimage; memory-intensive)
  all     - run python, then julia (default)

Versions:
  Python 3.10.x is required (tested: 3.10.12); .venv is created with python3.10.
  Julia 1.12.6 is installed/selected automatically via juliaup.

Environment:
  PYTHON   Python interpreter (if set, venv is not created/used)
  VENV_DIR Virtual environment directory (default: <project-root>/.venv)

Examples:
  ./run_setup.sh
  ./run_setup.sh python
  PYTHON=python3.10 ./run_setup.sh all
USAGE
}

resolve_python() {
	if [[ -n "${PYTHON:-}" ]]; then
		if ! command -v "${PYTHON}" >/dev/null 2>&1; then
			echo "Error: PYTHON=${PYTHON} not found. Unset PYTHON to use ${VENV_DIR}, or set it to an installed interpreter (e.g. python3.10)." >&2
			exit 1
		fi
		PYTHON_BIN="${PYTHON}"
		check_python_version
		return
	fi

	if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
		echo "==> Creating virtual environment at ${VENV_DIR}"
		if ! command -v "python${PYTHON_MINOR}" >/dev/null 2>&1; then
			echo "Error: python${PYTHON_MINOR} not found. Install Python ${PYTHON_MINOR} (tested: ${PYTHON_VERSION}) and rerun setup." >&2
			exit 1
		fi
		if ! "python${PYTHON_MINOR}" -m venv "${VENV_DIR}"; then
			cat >&2 <<'EOF'

Error: failed to create the virtual environment.
On Ubuntu/Debian, install the venv package and rerun setup:

  sudo apt install -y python3.10-venv
  ./run_setup.sh python

EOF
			exit 1
		fi
	fi
	PYTHON_BIN="${VENV_DIR}/bin/python"
	check_python_version
}

check_python_version() {
	local found
	found="$("${PYTHON_BIN}" -c 'import platform; print(platform.python_version())')"
	if [[ "${found%.*}" != "${PYTHON_MINOR}" ]]; then
		echo "Error: Python ${PYTHON_MINOR}.x is required (tested: ${PYTHON_VERSION}), but ${PYTHON_BIN} is ${found}." >&2
		echo "Remove ${VENV_DIR} and rerun setup, or set PYTHON to a Python ${PYTHON_MINOR} interpreter." >&2
		exit 1
	fi
	if [[ "${found}" != "${PYTHON_VERSION}" ]]; then
		echo "Warning: using Python ${found}; results were verified with ${PYTHON_VERSION}." >&2
	fi
}

ensure_julia_version() {
	local juliaup
	juliaup="$(command -v juliaup || echo "${HOME}/.juliaup/bin/juliaup")"
	if [[ ! -x "${juliaup}" ]]; then
		echo "juliaup not found; make sure Julia ${JULIA_VERSION} is on PATH or set JULIA." >&2
		return
	fi
	if ! "${juliaup}" status | grep -qE "(^|[[:space:]])${JULIA_VERSION//./\\.}([[:space:]]|$)"; then
		echo "==> Installing Julia ${JULIA_VERSION} via juliaup"
		"${juliaup}" add "${JULIA_VERSION}"
	fi
}

ensure_venv_tools() {
	if [[ -n "${PYTHON:-}" ]]; then
		return
	fi

	if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
		echo "==> Installing pip into ${VENV_DIR}"
		if ! "${PYTHON_BIN}" -m ensurepip --upgrade; then
			cat >&2 <<'EOF'

Error: failed to install pip into the virtual environment.
On Ubuntu/Debian, install the venv package and rerun setup:

  sudo apt install -y python3.10-venv
  ./run_setup.sh python

EOF
			exit 1
		fi
	fi

	echo "==> Updating Python packaging tools in ${VENV_DIR}"
	"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
}

print_venv_hint() {
	if [[ -n "${PYTHON:-}" ]]; then
		return
	fi
	cat <<EOF

Virtual environment: ${VENV_DIR}
Pipeline scripts use this virtual environment when PYTHON is unset.
To run a stage directly, use one of the following:

  source "${VENV_DIR}/bin/activate"
  # or
  export PYTHON="${VENV_DIR}/bin/python"
EOF
}

run_python_setup() {
	resolve_python
	ensure_venv_tools
	echo "==> Step: Python package setup (using ${PYTHON_BIN})"
	"${PYTHON_BIN}" "${SCRIPT_DIR}/00_setup_python_package.py"
}

run_julia_setup() {
	resolve_python
	ensure_venv_tools
	ensure_julia_version
	echo "==> Step: Julia sysimage setup (using ${PYTHON_BIN})"
	"${PYTHON_BIN}" "${SCRIPT_DIR}/01_setup_julia_env.py"
}

target="${1:-all}"

case "${target}" in
	all)
		cd "${PROJECT_ROOT}"
		run_python_setup
		run_julia_setup
		print_venv_hint
		;;
	python|py)
		cd "${PROJECT_ROOT}"
		run_python_setup
		print_venv_hint
		;;
	julia|jl)
		cd "${PROJECT_ROOT}"
		run_julia_setup
		;;
	-h|--help|help)
		usage
		;;
	*)
		echo "Unknown target: ${target}" >&2
		usage
		exit 2
		;;
esac
