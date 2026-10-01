import jax
import jax.numpy as jnp
import toml
import importlib
import pkgutil

import src.parameter_estimation.registry as registry
from src.parameter_estimation.estimator_modules.Neighbor.Delaunay import neighbor_delaunay_stepwise


def auto_load_modules():
    base_package = "src.parameter_estimation.estimator_modules"
    sub_modules = ["Social", "Kinetics"]

    for sub in sub_modules:
        pkg_name = f"{base_package}.{sub}"

        try:
            pkg = importlib.import_module(pkg_name)
            for _, module_name, _ in pkgutil.iter_modules(pkg.__path__):
                importlib.import_module(f"{pkg_name}.{module_name}")
        except Exception as e:
            print(f"❌ Failed to load {pkg_name}: {e}")


def extract_params_from_toml(config_toml):
    target_modules = ["Kinetics", "Social"]
    params = {}

    for module in target_modules:
        if module in config_toml:
            module_params = {} 
            
            for key, value in config_toml[module].items():
                # bool型（True/False）が数値として抽出されるのを防ぐ
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    module_params[key] = value
            params[module] = module_params
            
    return params


def build_loss_fn_stepwise(config_toml):
    social_fn         = registry.SOCIAL_REGISTRY[config_toml["Social"]["type"]]
    kinetics_fn       = registry.KINETICS_REGISTRY[config_toml["Kinetics"]["type"]]

    def loss_fn_stepwise(params, states_t, state_next, aux_data_t):
        neighbor_weights = neighbor_delaunay_stepwise(params, states_t, aux_data_t)
        social_factor_t  = social_fn(params, states_t, neighbor_weights, aux_data_t)
        loss_t           = kinetics_fn(params, states_t, state_next, social_factor_t, aux_data_t)

        return loss_t

    return loss_fn_stepwise


if __name__ == "__main__":
    config_toml_path = "./data/simulation/model_0001/config_0001.toml"
