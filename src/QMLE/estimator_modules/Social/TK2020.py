import jax.numpy as jnp

import src.QMLE.registry as registry

@registry.register_social("Soc_TK2020")
def social_tk2020_stepwise(params, states_t, neighbor_weights, aux_data_t):
    k_coh = params["Social"]["k_coh"]
    k_sep = params["Social"]["k_sep"]
    k_alg = params["Social"]["k_alg"]

    pos_t = states_t["positions"]  # (N, 2)
    vel_t = states_t["velocities"]  # (N, 2)
    theta_t = states_t["orientations"]  # (N,)
    safe_eps = aux_data_t["safe_eps"]  # Small value for numerical stability
    
    N = pos_t.shape[0]

    # Mask to exclude self-interaction (i == j)
    # Redundant if the diagonal is already 0, but kept as a safeguard
    neighbor_weights_no_self = neighbor_weights * (1.0 - jnp.eye(N))

    # ==========================================
    # 1. Repulsion
    # ==========================================
    # Compute relative vectors r_ij for all pairs via broadcasting
    # pos_j (1, N, 2) - pos_i (N, 1, 2) -> r_ij (N, N, 2)
    r_ij = pos_t[jnp.newaxis, :, :] - pos_t[:, jnp.newaxis, :]

    d_ij = jnp.sqrt(jnp.sum(r_ij**2, axis=-1) + safe_eps**2) # (N, N) pairwise distances; safe_eps prevents division by zero
    
    # sep_force_ij: (N, N, 2)
    sep_force_ij = -k_sep * (r_ij / d_ij[..., jnp.newaxis]) * neighbor_weights_no_self[..., jnp.newaxis]
    sep_force = jnp.sum(sep_force_ij, axis=1)            # (N, 2)

    # ==========================================
    # 2. Cohesion & alignment
    # ==========================================
    # Weight normalization term (N, 1)
    regularization = jnp.sum(neighbor_weights_no_self, axis=1, keepdims=True) + safe_eps
    
    # Normalized weights W_ij / sum(W) -> (N, N)
    norm_weights = neighbor_weights_no_self / regularization

    # Centroid of the other agents (N, 2)
    com_others = jnp.sum(pos_t[jnp.newaxis, :, :] * norm_weights[..., jnp.newaxis], axis=1)
    # Mean velocity of the other agents (N, 2)
    avg_vel_others = jnp.sum(vel_t[jnp.newaxis, :, :] * norm_weights[..., jnp.newaxis], axis=1)

    # Cohesion
    dist_to_com = jnp.sqrt(jnp.sum((com_others - pos_t)**2, axis=-1, keepdims=True) + safe_eps**2)
    coh_force = k_coh * (com_others - pos_t) / dist_to_com         # (N, 2)

    # Alignment
    diff_to_avg_vel = avg_vel_others - vel_t        # (N, 2)
    norm_diff_vel = jnp.sqrt(jnp.sum(diff_to_avg_vel**2, axis=-1, keepdims=True) + safe_eps**2)
    alg_force = k_alg * diff_to_avg_vel / norm_diff_vel  # (N, 2)

    # ==========================================
    # 3. Sum and project onto the normal direction
    # ==========================================
    total_force = sep_force + coh_force + alg_force      # (N, 2)

    # Normal vector to the heading, matching SVector{2, T}(-sin(θ), cos(θ)) -> (N, 2)
    normal_vec = jnp.stack([-jnp.sin(theta_t), jnp.cos(theta_t)], axis=-1)

    # Dot product as elementwise product and sum -> (N,)
    social_factor = jnp.sum(total_force * normal_vec, axis=-1)

    return social_factor