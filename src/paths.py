import functools
import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the data directory
DATA_ROOT = PROJECT_ROOT / "data"

# Julia project directory for the simulator
JULIA_PROJECT_DIR = PROJECT_ROOT / "src/flocking_models"
JULIA_SYSIMAGE_PATH = JULIA_PROJECT_DIR / "build_tools" / "sys_simulation.so"


# Tested toolchain versions; keep in sync with README.md and run_setup.sh
PYTHON_VERSION = "3.10.12"
JULIA_VERSION = "1.12.6"


@functools.lru_cache(maxsize=None)
def julia_executable() -> str:
	"""Return a Julia executable that runs JULIA_VERSION.

	Looks in $JULIA, then PATH, then the default juliaup location. When the juliaup
	launcher is used, JULIAUP_CHANNEL is set so that it (and child processes) select
	JULIA_VERSION.
	"""
	os.environ.setdefault("JULIAUP_CHANNEL", JULIA_VERSION)
	candidates = [os.environ.get("JULIA"), shutil.which("julia"), str(Path.home() / ".juliaup/bin/julia")]
	julia = next(
		(c for c in candidates if c and Path(c).is_file() and os.access(c, os.X_OK)),
		None,
	)
	if julia is None:
		raise FileNotFoundError(
			"Julia executable not found. Install Julia (e.g. via juliaup), add it to PATH, "
			"or set the JULIA environment variable to its path."
		)
	result = subprocess.run([julia, "--version"], capture_output=True, text=True)
	found = result.stdout.strip().removeprefix("julia version ")
	if result.returncode != 0 or found != JULIA_VERSION:
		raise RuntimeError(
			f"Julia {JULIA_VERSION} is required, but {julia} reports "
			f"{found or result.stderr.strip()!r}. Run ./run_setup.sh julia "
			f"(installs it via juliaup) or `juliaup add {JULIA_VERSION}`."
		)
	return julia

# Directory of the estimation models
ESTIM_ROOT = PROJECT_ROOT / "src/QMLE"

# Directory for saving results
RESULTS_ROOT = PROJECT_ROOT / "results"
