from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# dataディレクトリのパス
DATA_ROOT = PROJECT_ROOT / "data"

# シミュレータのJuliaプロジェクトディレクトリ
JULIA_PROJECT_DIR = PROJECT_ROOT / "src/flocking_models"
JULIA_SYSIMAGE_PATH = JULIA_PROJECT_DIR / "build_tools" / "sys_simulation.so"

# 推定モデルのディレクトリ
ESTIM_ROOT = PROJECT_ROOT / "src/QMLE"

# 結果保存ディレクトリ
RESULTS_ROOT = PROJECT_ROOT / "results"
