from dataclasses import dataclass

@dataclass(frozen=True)
class MasterConstants:
    SAFE_EPS_32 = 1e-6
    SAFE_EPS_64 = 1e-14
    ABSOLUTE_VELOCITY = 0.05
    GROUND_TRUTH_PRECISION = "Float64"
    ESTIMATION_PRECISION = "Float64"
    FPS = 24
    BURN_IN_SECONDS = 60 * 2
    TOTAL_TIME_SECONDS = 60 * 7
    HESSIAN_COMPUTATION_INTERVAL = 1/FPS
    PARAMETER_ESTIMATION_INTERVAL = 30
    TRIALS_PER_MODEL = 50
