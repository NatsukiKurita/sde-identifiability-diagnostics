import jax.numpy as jnp

import src.QMLE.registry as registry

@registry.register_social("Soc_Null")
def social_null_stepwise(params, states_t, neighbor_weights, aux_data_t):
    N = states_t["positions"].shape[0]  # number of agents
    return jnp.zeros(N)  # (N,) zero vector
