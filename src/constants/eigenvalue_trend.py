from dataclasses import dataclass

from src.constants.master_constants import MasterConstants


@dataclass(frozen=True)
class EigenvalueTrendConfig:
    FPS = MasterConstants.FPS
    TIME_STEP = 1 / FPS        # シミュレーションの時間刻み（秒）
    BURN_IN = FPS * MasterConstants.BURN_IN_SECONDS    # バーンイン期間（フレーム数）

    # ヘシアンの計算頻度
    HESSIAN_COMPUTATION_INTERVAL = MasterConstants.HESSIAN_COMPUTATION_INTERVAL  # ヘシアンを計算する間隔（秒）                # エージェント数のリスト
    PRECISION = MasterConstants.ESTIMATION_PRECISION              # データ保存の精度

    VOLATILE_SIGNED_KAUFMAN_THRESHOLD = 0.5


def classify_signed_kaufman_region(
    signed_kaufman_ratio: float,
    *,
    threshold: float | None = None,
) -> str:
    """Classify a signed Kaufman ratio using the current volatile threshold."""
    limit = float(
        EigenvalueTrendConfig.VOLATILE_SIGNED_KAUFMAN_THRESHOLD
        if threshold is None
        else threshold
    )
    if float(signed_kaufman_ratio) < limit:
        return "volatile"
    return "well-behaved"


def build_classification_record(
    signed_kaufman_ratio_opt: float,
    signed_kaufman_ratio_truth: float,
    *,
    threshold: float | None = None,
) -> dict[str, float | str]:
    """Build the short classification payload written by trend-calculation scripts."""
    limit = float(
        EigenvalueTrendConfig.VOLATILE_SIGNED_KAUFMAN_THRESHOLD
        if threshold is None
        else threshold
    )
    return {
        "threshold": limit,
        "region_at_opt": classify_signed_kaufman_region(
            signed_kaufman_ratio_opt,
            threshold=limit,
        ),
        "region_at_truth": classify_signed_kaufman_region(
            signed_kaufman_ratio_truth,
            threshold=limit,
        ),
    }
