from dataclasses import dataclass

from src.constants.master_constants import MasterConstants
from src.constants.parameter_settings import load_parameter_presets


@dataclass(frozen=True)
class SocialSimConfig:
    PRESET_GROUP = "01_social_interaction"
    SIMULATION_CAMPAIGN = PRESET_GROUP

    PRECISION = MasterConstants.GROUND_TRUTH_PRECISION
    SAFE_EPS = MasterConstants.SAFE_EPS_32 if PRECISION == "Float32" else MasterConstants.SAFE_EPS_64

    NUM_AGENTS = [2, 4, 8] 
    ABSOLUTE_VELOCITY = MasterConstants.ABSOLUTE_VELOCITY
    TRIALS_PER_MODEL = MasterConstants.TRIALS_PER_MODEL
    WALL = "NoWall"
    FPS = MasterConstants.FPS
    TIME_STEP = 1 / FPS
    TOTAL_FRAMES = FPS * MasterConstants.TOTAL_TIME_SECONDS
    PARAMETER_SETTINGS = load_parameter_presets(PRESET_GROUP)
