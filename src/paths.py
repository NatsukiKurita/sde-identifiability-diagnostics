from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# dataディレクトリのパス
DATA_ROOT = PROJECT_ROOT / "data"

# シミュレータのJuliaプロジェクトディレクトリ
JULIA_PROJECT_DIR = PROJECT_ROOT / "src/flocking_models"

# 推定モデルのディレクトリ
ESTIM_ROOT = PROJECT_ROOT / "parameter_estimation"

# 結果保存ディレクトリ
RESULTS_ROOT = PROJECT_ROOT / "results"
