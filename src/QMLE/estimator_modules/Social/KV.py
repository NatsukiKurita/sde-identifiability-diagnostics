import jax.numpy as jnp

import src.QMLE.registry as registry

@registry.register_social("Soc_KV")
def social_kv_stepwise(params, states_t, neighbor_weights, aux_data_t):
    k_alg = params["Social"]["k_alg"]

    # 1. Get orientations from the state (N,)
    # Note: adjust the states_t key names to match the implementation
    theta_t = states_t["orientations"]

    # 2. Expand dimensions for broadcasting
    # theta_self: angle of agent i (N, 1)
    # theta_nbr : angle of agent j (1, N)
    theta_self = theta_t[:, jnp.newaxis] 
    theta_nbr = theta_t[jnp.newaxis, :]  

    # 3. Angle differences and their sines (N, N)
    delta_theta_ij = theta_nbr - theta_self
    sin_delta_theta_ij = jnp.sin(delta_theta_ij)

    # 4. Apply the weight matrix (N, N)
    # Self-interaction (i=j) vanishes here automatically if the diagonal of neighbor_weights is 0
    interaction_weighted = sin_delta_theta_ij * neighbor_weights

    # 5. For each agent i, sum the contributions from all other agents j (N,)
    sum_interaction = jnp.sum(interaction_weighted, axis=1)

    # 6. Multiply by the coupling coefficient k_alg and return the final social_factor
    social_factor = k_alg * sum_interaction

    return social_factor
