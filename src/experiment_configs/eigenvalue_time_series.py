from dataclasses import dataclass

from src.experiment_configs.master_constants import MasterConstants


@dataclass(frozen=True)
class EigenvalueTimeSeriesConfig:
    FPS = MasterConstants.FPS
    TIME_STEP = 1 / FPS        # Simulation time step (seconds)
    BURN_IN = FPS * MasterConstants.BURN_IN_SECONDS    # Burn-in period (frames)

    # Hessian computation frequency
    HESSIAN_COMPUTATION_INTERVAL = MasterConstants.HESSIAN_COMPUTATION_INTERVAL  # Interval between Hessian computations (seconds)
    PRECISION = MasterConstants.ESTIMATION_PRECISION              # Precision for saved data
