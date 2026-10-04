import jax.numpy as jnp

import src.QMLE.registry as registry
import src.QMLE.utils.geometry as geometry

@registry.register_social("Soc_G2012")
def social_g2012_stepwise(params, states_t, neighbor_weights, aux_data_t):
    k_pos = params["Social"]["k_pos"]
    k_alg = params["Social"]["k_alg"]

    safe_eps = aux_data_t["safe_eps"]  # Small value for numerical stability
    pos_t = states_t["positions"]  # shape: (N, 2)
    theta_t = states_t["orientations"]  # shape: (N,)

    # 1. Expand dimensions for broadcasting
    # pos_self: position of agent i, laid out along rows (N, 1, 2)
    # pos_nbr:  position of agent j, laid out along columns (1, N, 2)
    pos_self = pos_t[:, jnp.newaxis, :]  # (N, 1, 2)
    pos_nbr = pos_t[jnp.newaxis, :, :]   # (1, N, 2)
    
    theta_self = theta_t[:, jnp.newaxis] # (N, 1)
    theta_nbr = theta_t[jnp.newaxis, :]  # (1, N)

    # 2. Compute safe relative vectors (N, N, 2)
    r_ij_vec = pos_nbr - pos_self
    
    # Masking to prevent NaN gradients, e.g. on the diagonal (self, distance 0)
    # Use the true vector only where the weight is nonzero; fall back to safe_vec elsewhere
    mask_expanded = jnp.abs(neighbor_weights)[..., jnp.newaxis] # (N, N, 1)
    safe_vec = jnp.array([1.0, 0.0], dtype=r_ij_vec.dtype)
    r_ij_vec_safe = jnp.where(mask_expanded > 0, r_ij_vec, safe_vec)

    # 3. Distances and unit vectors
    d_ij = jnp.linalg.norm(r_ij_vec_safe, axis=-1)            # (N, N)
    d_ij_safe = jnp.maximum(d_ij, safe_eps)                   # (N, N)
    e_ij = r_ij_vec_safe / d_ij_safe[..., jnp.newaxis]        # (N, N, 2)
    
    # 4. Build heading vectors (N, 1, 2) and (1, N, 2)
    dir_self = jnp.stack([jnp.cos(theta_self), jnp.sin(theta_self)], axis=-1)
    dir_nbr = jnp.stack([jnp.cos(theta_nbr), jnp.sin(theta_nbr)], axis=-1)

    # 5. Interaction forces via cross products (broadcast to (N, N))
    sin_phi_ij = geometry.cross_2d(dir_self, e_ij)                      # (N, N)
    omega_pos_k = k_pos * d_ij * sin_phi_ij # (N, N)

    sin_delta_theta_ij = geometry.cross_2d(dir_self, dir_nbr)          # (N, N)
    omega_alg_k = k_alg * sin_delta_theta_ij          # (N, N)

    # 6. Apply the weight matrix (also acts as a mask)
    omega_pos_weighted = omega_pos_k * neighbor_weights        # (N, N)
    omega_alg_weighted = omega_alg_k * neighbor_weights        # (N, N)

    # Sum the contributions to agent i over all agents j
    sum_pos = jnp.sum(omega_pos_weighted, axis=1)             # (N,)
    sum_alg = jnp.sum(omega_alg_weighted, axis=1)             # (N,)

    return (sum_pos + sum_alg) 
