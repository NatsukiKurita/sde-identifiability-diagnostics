from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc
from typing import Any

from src.constants.master_constants import MasterConstants


@dataclass(frozen=True)
class EstimationConfig:
    # 数値精度の一元設定
    PRECISION = MasterConstants.ESTIMATION_PRECISION
    JNP_FLOAT_DTYPE = jnp.float64 if PRECISION == "Float64" else jnp.float32
    NP_FLOAT_DTYPE = np.float64 if PRECISION == "Float64" else np.float32

    @classmethod
    def configure_jax_precision(cls) -> None:
        jax.config.update("jax_enable_x64", bool(cls.PRECISION == "Float64"))

    @classmethod
    def as_float_array(cls, value):
        return jnp.asarray(value, dtype=cls.JNP_FLOAT_DTYPE)

    @classmethod
    def cast_float_tree(cls, pytree):
        return jax.tree_util.tree_map(
            lambda x: cls.as_float_array(x) if isinstance(x, (int, float, np.floating)) else x,
            pytree,
        )

    # Optimizer settings. The main estimation path uses SciPy trust-constr.
    TOL_PRESEARCH_TRUST_CONSTR = 1e-3
    XTOL_PRESEARCH_TRUST_CONSTR = 1e-5

    TOL_TRUST_CONSTR = 1e-5
    XTOL_TRUST_CONSTR = 1e-9

    MAXITER_PRESEARCH_TRUST_CONSTR = 30
    MAXITER_TRUST_CONSTR = 1000
    PARAMETER_ESTIMATION_INTERVAL = MasterConstants.PARAMETER_ESTIMATION_INTERVAL

    # Legacy L-BFGS-B settings kept for optional scripts.
    TOL_PRESEARCH_LBFGSB = TOL_PRESEARCH_TRUST_CONSTR
    TOL_LBFGSB = TOL_TRUST_CONSTR
    MAXITER_PRESEARCH_LBFGSB = MAXITER_PRESEARCH_TRUST_CONSTR
    MAXITER_LBFGSB = MAXITER_TRUST_CONSTR

    # 関数や定数
    SAFE_EPS = MasterConstants.SAFE_EPS_32 if PRECISION == "Float32" else MasterConstants.SAFE_EPS_64

    # Trust-Region Newton-CGの範囲設定
    BOUNDARIES = {
        "Kinetics": {
            "Kinetics_Overdamped":{
                "sigma_theta": (1e-5, jnp.inf)
            },
            "Kinetics_Inertial":{
                "sigma_omega": (1e-5, jnp.inf),
                "tau"  : (1e-5, jnp.inf)
            }
        },
        "Social": {
            "Soc_KR2021":{
                "k_alg": (0.0, jnp.inf),
                "k_pos": (0.0, jnp.inf),
                "m"    : (0.0, jnp.inf),
                "d_eq" : (0.0, jnp.inf)
            },
            "Soc_G2012":{
                "k_pos": (0.0, jnp.inf),
                "k_alg": (0.0, jnp.inf),
            },
            "Soc_KV":{
                "k_alg": (0.0, jnp.inf)
            },
            "Soc_TK2020":{
                "k_sep": (0.0, jnp.inf),
                "k_coh": (0.0, jnp.inf),
                "k_alg": (0.0, jnp.inf)
            },
            "Soc_Null":{},
        },
        "WallAvoidance": {
            "WA_Elastic":{
                "k_w": (0.0, jnp.inf),
                "d_w": (0.0, jnp.inf)
            },
            "WA_G2012":{
                "k_w": (0.0, jnp.inf)
            },
            "WA_TK2020":{
                "k_wrep": (0.0, jnp.inf),
                "k_wcoh": (0.0, jnp.inf)
            },
            "WA_Null": {},
        },
        "Neighbor": {
            "Neighbor_Delaunay": {}
        }
    }

    @classmethod
    def extract_bounds(cls, config_toml) -> dict:
        lower_bounds = {}
        upper_bounds = {}

        kinetics       = config_toml["Kinetics"]["type"]
        social         = config_toml["Social"]["type"]
        wall_avoidance = config_toml["WallAvoidance"]["type"]
        neighbor       = config_toml["Neighbor"]["type"]

        modules = {
            "Kinetics": kinetics,
            "Social": social,
            "WallAvoidance": wall_avoidance,
            "Neighbor": neighbor
        }

        for key, module in modules.items():
            if not module in cls.BOUNDARIES[key]:
                raise ValueError(f"Unknown model: {module} for {key}. Available: {list(cls.BOUNDARIES[key].keys())}")
            else:
                boundary = cls.BOUNDARIES[key][module]

            lower_bounds[key] = {}
            upper_bounds[key] = {}
        
            for param, (low, high) in boundary.items():
                lower_bounds[key][param] = low
                upper_bounds[key][param] = high
        
        return lower_bounds, upper_bounds

    # マルチスタート
    # 初期点生成はラテン超方格法(LHS)を使用
    NUM_STARTS = 10                 # マルチスタート最適化の開始点数
    MULTISTART_LHS_SEED = 42        # マルチスタートLHSの共通シード

    # 初期点配置
    INITIAL_REGION = {
        "kinetics": {
            "Kinetics_Overdamped":{
                "sigma_theta": (1e-5, 20.0)
            },
            "Kinetics_Inertial":{
                "sigma_omega": (1e-5, 50.0),
                "tau"  : (1e-5, 5.0)
            }
        },
        "social": {
            "Soc_KR2021":{
                "k_alg": (0.0, 20.0),
                "k_pos": (0.0, 20.0),
                "m"    : (0.0, 10.0),
                "d_eq" : (0.0, 1.0)
            },
            "Soc_G2012":{
                "k_pos": (0.0, 20.0),
                "k_alg": (0.0, 20.0),
            },
            "Soc_KV":{
                "k_alg": (0.0, 20.0)
            },
            "Soc_TK2020":{
                "k_sep": (0.0, 400.0),
                "k_coh": (0.0, 400.0),
                "k_alg": (0.0, 400.0)
            },
            "Soc_Null":{},
        },
        "wall_avoidance": {
            "WA_Elastic":{
                "k_w": (0.0, 20.0),
                "d_w": (0.0, 1.0)
            },
            "WA_G2012":{
                "k_w": (0.0, 20.0)
            },
            "WA_TK2020":{
                "k_wrep": (0.0, 400.0),
                "k_wcoh": (0.0, 400.0)
            },
            "WA_Null": {},
        },
        "neighbor": {
            "Neighbor_Delaunay": {}
        }
    }

    @classmethod
    def _expand_param_for_batch(cls, param: Any, num_starts: int) -> np.ndarray:
        arr = np.asarray(param, dtype=cls.NP_FLOAT_DTYPE)
        if arr.ndim == 0:
            return np.full((num_starts,), float(arr), dtype=cls.NP_FLOAT_DTYPE)
        return np.repeat(arr[np.newaxis, ...], repeats=num_starts, axis=0)

    @classmethod
    def sample_multistart_initial_params(cls, config_toml: dict, base_init_params: dict, seed: int | None = None) -> dict:
        """Sample batched initial parameters with Latin Hypercube Sampling.

        Sampling ranges are taken from `INITIAL_REGION` according to model types in
        `config_toml`. Parameters without explicit range keep the base value.
        """
        num_starts = int(cls.NUM_STARTS)
        if num_starts <= 1:
            return base_init_params

        region_key_map = {
            "Kinetics": "kinetics",
            "Social": "social",
            "WallAvoidance": "wall_avoidance",
            "Neighbor": "neighbor",
        }

        param_specs = []
        for module_name, module_params in base_init_params.items():
            if not isinstance(module_params, dict):
                continue
            model_type = config_toml.get(module_name, {}).get("type")
            region_group_key = region_key_map.get(module_name)
            initial_region = {}
            if region_group_key is not None:
                initial_region = cls.INITIAL_REGION.get(region_group_key, {}).get(model_type, {})

            for param_name, base_value in module_params.items():
                region = initial_region.get(param_name)
                param_specs.append((module_name, param_name, base_value, region))

        if not param_specs:
            return base_init_params

        lhs_dim = len(param_specs)
        lhs_seed = cls.MULTISTART_LHS_SEED if seed is None else int(seed)
        lhs_engine = qmc.LatinHypercube(d=lhs_dim, seed=lhs_seed)
        lhs_unit = lhs_engine.random(n=num_starts)

        sampled_values = {}
        for dim_idx, (module_name, param_name, base_value, region) in enumerate(param_specs):
            if region is None:
                sampled = cls._expand_param_for_batch(base_value, num_starts)
            else:
                low, high = region
                low_f = float(low)
                high_f = float(high)
                if not np.isfinite(low_f) or not np.isfinite(high_f) or high_f <= low_f:
                    sampled = cls._expand_param_for_batch(base_value, num_starts)
                else:
                    sampled = (low_f + (high_f - low_f) * lhs_unit[:, dim_idx]).astype(cls.NP_FLOAT_DTYPE)

            sampled_values[(module_name, param_name)] = sampled

        batched_params = {}
        for module_name, module_params in base_init_params.items():
            if not isinstance(module_params, dict):
                batched_params[module_name] = module_params
                continue
            batched_params[module_name] = {}
            for param_name in module_params:
                batched_params[module_name][param_name] = sampled_values[(module_name, param_name)]

        return batched_params




    
