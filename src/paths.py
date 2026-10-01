from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# dataディレクトリのパス
DATA_ROOT = PROJECT_ROOT / "data"

# シミュレータのJuliaプロジェクトディレクトリ
JULIA_PROJECT_DIR = PROJECT_ROOT / "src/dyn_models"

# 推定モデルのディレクトリ
ESTIM_ROOT = PROJECT_ROOT / "param_estim_jax"

# 結果保存ディレクトリ
RESULTS_ROOT = PROJECT_ROOT / "results"
