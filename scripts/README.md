# シミュレーション・推定・診断・描画ワークフロー

社会相互作用モデルの処理は、次の順で実行します。結果は `data/simulation` と `results` に保存します。

## 前提

- リポジトリルートで作業する
- Python 依存（`toml`, `jax`, `h5py` など）はプロジェクト直下の `.venv` に入れる
- シミュレーション段階には Julia が必要
- 初回はセットアップを済ませておく

```bash
cd /path/to/publish_thesis
scripts/00_setup/run_setup.sh
```

Python インタプリタを変えたい場合のみ:

```bash
export PYTHON=/path/to/your/python
```

以下の例は、未指定なら `.venv/bin/python` を使う前提です。

## 段階

| 段階 | スクリプト | 内容 | 主な出力 |
|------|-------------|------|----------|
| 00 | `scripts/00_setup/run_setup.sh` | Python / Julia 環境 | `.venv` |
| 01 | `scripts/01_simulation/` | TOML 準備とシミュレーション | `data/simulation/` |
| 02 | `scripts/02_estimation/02_estimation.py` | パラメータ推定 | `summary.json`, `parameter_estimates.json` |
| 03 | `scripts/03_diagnostics/03_diagnostics.py` | 最適解のヘシアンと最小固有値 | `results.h5`, `eval_trend_at_opt.json` |
| 04 | `scripts/04_figures/fig*.py` | 論文用図（図ごとに実行） | `results/<setting>/figures/` |
| 05 | `scripts/05_supplemental_figures/` | 補足図 | 各スクリプトの出力先 |

## よく使うコマンド

```bash
# 01: TOML
.venv/bin/python scripts/01_simulation/00_prepare_toml.py --overwrite

# 01: シミュレーション
.venv/bin/python scripts/01_simulation/01_run_simulation.py --setting setting_01_1 --overwrite

# 02: 推定
.venv/bin/python scripts/02_estimation/02_estimation.py --setting setting_01_1 --overwrite

# 03: 最適解のヘシアンと最小固有値
.venv/bin/python scripts/03_diagnostics/03_diagnostics.py --setting setting_01_1 --overwrite

# 04: 論文図（必要な図のスクリプトだけ実行する）
.venv/bin/python scripts/04_figures/fig01_min_eigenvalue_timeseries.py --setting setting_01_1 --num-agents 2 16
.venv/bin/python scripts/04_figures/fig02_spaghetti_all_models.py --setting setting_01_1 --num-agents 2 16
.venv/bin/python scripts/04_figures/fig03_min_eigenvector_lollipop.py --setting setting_01_1 --num-agents 2 16
.venv/bin/python scripts/04_figures/fig04_precision_timeseries.py --setting setting_01_1 --num-agents 2 16
```

各スクリプトの全オプション:

```bash
.venv/bin/python scripts/01_simulation/00_prepare_toml.py --help
.venv/bin/python scripts/01_simulation/01_run_simulation.py --help
.venv/bin/python scripts/02_estimation/02_estimation.py --help
.venv/bin/python scripts/03_diagnostics/03_diagnostics.py --help
.venv/bin/python scripts/04_figures/fig01_min_eigenvalue_timeseries.py --help
```

## 01_simulation

**prepare** (`00_prepare_toml.py`)

| オプション | 説明 |
|-----------|------|
| `--overwrite` | 既存 TOML を上書き |

**simulate** (`01_run_simulation.py`)

| オプション | 説明 |
|-----------|------|
| `--setting SETTING` | 対象 setting（例: `setting_01_1`） |
| `--simulation_root PATH` | シミュレーションルート（省略時: `data/simulation`） |
| `--overwrite` | 既存 HDF5 を上書き |
| `--skip-restore-delaunay` | シミュレーション後の Delaunay 復元をスキップ |

## 02_estimation

| オプション | 説明 |
|-----------|------|
| `--setting SETTING` | 対象 setting |
| `--sim_root PATH` | シミュレーションデータのルート |
| `--overwrite` | 既存推定結果を上書き |

ヘシアンは計算しません。損失とパラメータ推定を `summary.json` と `parameter_estimates.json` に書きます。

## 03_diagnostics

推定済みの最適パラメータで、最適解のヘシアン・固有対・最小固有値の時系列を計算します。真値での時系列もこの段階で書きます。

オプションは推定と同じです（`--sim_root`, `--setting`, `--overwrite`）。

## 04_figures

図ごとにスクリプトを実行します。共通オプションは各スクリプトで同じです。

| スクリプト | 内容 |
|-----------|------|
| `fig01_min_eigenvalue_timeseries.py` | 最小固有値の時系列 |
| `fig02_spaghetti.py` | 全社会モデルのスパゲッティ |
| `fig03_min_eigenvector_lollipop.py` | 最小固有ベクトルのロリポップ |
| `fig04_error_evolution.py` | パラメータ精度の時系列 |

| オプション | 説明 |
|-----------|------|
| `--setting SETTING` | 対象 setting |
| `--results-root PATH` | 結果ルート（省略時: `results`） |
| `--num-agents N ...` | 図に含める個体数 |
| `--main-num-agents N ...` | 本文パネル用の個体数 |

## 05_supplemental_figures

補足図のスクリプトです。ディレクトリ名を変えただけで、各ファイルの中身は以前と同じです。
