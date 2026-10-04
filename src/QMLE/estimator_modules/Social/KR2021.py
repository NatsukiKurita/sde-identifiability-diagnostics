import jax.numpy as jnp

import src.QMLE.registry as registry

@registry.register_social("Soc_KR2021")
def social_kr2021_stepwise(params, states_t, neighbor_weights, aux_data_t):
    k_alg  = params['Social']['k_alg']
    k_pos = params['Social']['k_pos']
    m = params['Social']['m']
    d_eq = params['Social']['d_eq']

    safe_eps = aux_data_t["safe_eps"]  # Small value for numerical stability
    v_0 = aux_data_t["v_0"] 

    pos_t = states_t["positions"]  # shape: (N, Dim)
    vel_t = states_t["velocities"]  # shape: (N, Dim)
    theta_t = states_t["orientations"]  # shape: (N, )
    N = pos_t.shape[0]

    # Alignment
    diff_vel = vel_t[jnp.newaxis, :, :] - vel_t[:, jnp.newaxis, :]  # (N, N, D)
    align_force = k_alg * neighbor_weights[:, :, jnp.newaxis] * diff_vel  # (N, N, D)

    # Repulsion
    diff_pos = pos_t[jnp.newaxis, :, :] - pos_t[:, jnp.newaxis, :]  # (N, N, D)
    diff_pos_norm = jnp.linalg.norm(diff_pos, axis=-1)  # (N, N)
    diff_pos_unit = diff_pos / (diff_pos_norm[:, :, jnp.newaxis] + safe_eps)  # (N, N, D)
    pos_force = k_pos * neighbor_weights[:, :, jnp.newaxis] * jnp.tanh(m * (diff_pos_norm[:, :, jnp.newaxis] - d_eq)) * diff_pos_unit  # (N, N, D)

    # Projection (sign convention matches Julia's SVector(-sin(theta), cos(theta)))
    e_perp = jnp.stack([-jnp.sin(theta_t), jnp.cos(theta_t)], axis=-1)  # (N, D)
    
    total_force = align_force + pos_force  # (N, N, D)
    social_force = jnp.sum(total_force, axis=1)  # (N, D)
    social_factor = jnp.sum(social_force * e_perp, axis=-1) / v_0  # (N,)

    return social_factor
