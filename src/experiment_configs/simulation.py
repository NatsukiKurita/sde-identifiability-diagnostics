from dataclasses import dataclass

from src.experiment_configs.master_constants import MasterConstants
from src.experiment_configs.parameter_settings import load_parameter_presets


@dataclass(frozen=True)
class SocialSimConfig:
    PRECISION = MasterConstants.GROUND_TRUTH_PRECISION
    SAFE_EPS = MasterConstants.SAFE_EPS_32 if PRECISION == "Float32" else MasterConstants.SAFE_EPS_64

    NUM_AGENTS = [2, 4]
    ABSOLUTE_VELOCITY = MasterConstants.ABSOLUTE_VELOCITY
    TRIALS_PER_MODEL = MasterConstants.TRIALS_PER_MODEL
    FPS = MasterConstants.FPS
    TIME_STEP = 1 / FPS
    TOTAL_FRAMES = FPS * MasterConstants.TOTAL_TIME_SECONDS
    PARAMETER_SETTINGS = load_parameter_presets()
