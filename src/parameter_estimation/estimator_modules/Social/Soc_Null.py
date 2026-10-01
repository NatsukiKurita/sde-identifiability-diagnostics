import jax.numpy as jnp

import src.parameter_estimation.registry as registry

@registry.register_social("Soc_Null")
def social_null_stepwise(params, states_t, neighbor_weights, aux_data_t):
    # 孤立主義モデルは、社会的相互作用がないため、常にゼロを返す
    N = states_t["positions"].shape[0]  # エージェントの数
    return jnp.zeros(N)  # (N,)のゼロベクトルを返す
