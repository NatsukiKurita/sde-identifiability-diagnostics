import jax.numpy as jnp

import src.parameter_estimation.registry as registry

@registry.register_kinetics("Kinetics_Overdamped")
def loss__overdamped_stepwise(
    params: dict,
    state_t,
    state_next,
    social_factor_t: jnp.ndarray,
    wall_factor_t: jnp.ndarray,
    aux_data_t: dict
):
    N = state_t["positions"].shape[0]
    dt = aux_data_t["dt"]
    sigma_theta = params["Kinetics"]["sigma_theta"]

    theta_t = state_t["orientations"]  # shape: (N,)
    theta_next = state_next["orientations"]  # shape: (N,)

    term0 = jnp.log(sigma_theta)  # 定数項 (Nに対して1回)
    term1 = 0.5 * jnp.log(2 * jnp.pi * dt)

    optimal_omega = social_factor_t + wall_factor_t  # shape: (N,)
    sq_diff = (theta_next - theta_t - optimal_omega * dt) ** 2
    term2 = sq_diff / (2 * (sigma_theta ** 2) * dt)

    # 合計
    return (term0 + term1) * N + jnp.sum(term2)