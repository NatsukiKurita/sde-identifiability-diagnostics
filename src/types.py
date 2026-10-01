import h5py
import numpy as np
from dataclasses import fields, field, dataclass
from pathlib import Path
from typing import TypedDict, FrozenSet, Literal, Optional, Dict

# モデル指定用リテラル型(to be deprecated)
Kinetics = Literal[
    "Overdamped.jl", 
    "Inertial.jl"
]

WallAvoidance = Literal[
    "WA_Elastic.jl", 
    "WA_G2012.jl", 
    "WA_MultiVision_PolyDecay.jl",
    "WA_MultiVision_ExpDecay.jl",
    "WA_TK2020.jl",
    "WA_Null.jl",
]

Social = Literal[
    "Soc_KR2021.jl", 
    "Soc_G2012.jl", 
    "Soc_KV.jl", 
    "Soc_TK2020.jl",
    "Soc_Null.jl"
]

Neighbor = Literal[
    "Delaunay.jl",
    "SoftMetric.jl"
]

# 境界条件リテラル
Walls = Literal["RectWall", "CircleWall", "NoWall"]

# 川・性別リテラル
Rivers = Literal["UA", "LA", "UT", "LT", "UQ", "LQ", "UTun", "LTac"]
Sexes = Literal["M", "F"]

#########################################
# データクラスの定義
#########################################

# --- 共通基底クラス ---
@dataclass
class TrajectoryData:
    """エージェントの軌跡データを表す基底クラス"""
    positions: np.ndarray  # shape: (num_agents, num_timesteps, 2)
    velocities: np.ndarray # shape: (num_agents, num_timesteps, 2)
    orientations: np.ndarray # shape: (num_agents, num_timesteps)
    angular_velocities: np.ndarray = None # shape: (num_agents, num_timesteps)