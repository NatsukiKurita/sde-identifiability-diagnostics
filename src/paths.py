from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the data directory
DATA_ROOT = PROJECT_ROOT / "data"

# Julia project directory for the simulator
JULIA_PROJECT_DIR = PROJECT_ROOT / "src/flocking_models"
JULIA_SYSIMAGE_PATH = JULIA_PROJECT_DIR / "build_tools" / "sys_simulation.so"

# Directory of the estimation models
ESTIM_ROOT = PROJECT_ROOT / "src/QMLE"

# Directory for saving results
RESULTS_ROOT = PROJECT_ROOT / "results"
