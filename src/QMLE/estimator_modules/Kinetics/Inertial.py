import jax.numpy as jnp

import src.QMLE.registry as registry

@registry.register_kinetics("Dynamics_Inertial")
def loss__inertial_stepwise(
    params: dict,
    state_t,
    state_next,
    social_factor_t: jnp.ndarray,
    aux_data_t: dict
):
    N = state_t["positions"].shape[0]
    dt = aux_data_t["dt"]
    sigma_omega = params["Kinetics"]["sigma_omega"]
    tau = params["Kinetics"]["tau"]

    omega_t = state_t["angular_velocities"]  # shape: (N,)
    omega_next = state_next["angular_velocities"]  # shape: (N,)

    term0 = jnp.log(sigma_omega)  # 定数項 (Nに対して1回)
    term1 = 0.5 * jnp.log(2 * jnp.pi * dt)
    term2 = jnp.log(dt/2)  # 数値積分による補正項

    optimal_omega = social_factor_t # shape: (N,)
    sq_diff = (omega_next - omega_t + (omega_t - optimal_omega) * (dt / tau)) ** 2
    term3 = sq_diff / (2 * (sigma_omega ** 2) * dt)

    # 合計
    return (term0 + term1 + term2) * N + jnp.sum(term3)