from dataclasses import dataclass

from src.constants.master_constants import MasterConstants


@dataclass(frozen=True)
class EigenvalueTimeSeriesConfig:
    FPS = MasterConstants.FPS
    TIME_STEP = 1 / FPS        # シミュレーションの時間刻み（秒）
    BURN_IN = FPS * MasterConstants.BURN_IN_SECONDS    # バーンイン期間（フレーム数）

    # ヘシアンの計算頻度
    HESSIAN_COMPUTATION_INTERVAL = MasterConstants.HESSIAN_COMPUTATION_INTERVAL  # ヘシアンを計算する間隔（秒）                # エージェント数のリスト
    PRECISION = MasterConstants.ESTIMATION_PRECISION              # データ保存の精度
