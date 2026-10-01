import jax.numpy as jnp

import src.parameter_estimation.registry as registry


@registry.register_wall_avoidance("WA_Null")
def wa_passing_through_stepwise(params, states_t, aux_data_t):
    num_agents = states_t["positions"].shape[0]
    return jnp.zeros(num_agents)
