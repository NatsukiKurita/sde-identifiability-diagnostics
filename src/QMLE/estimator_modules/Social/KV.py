import jax.numpy as jnp

import src.parameter_estimation.registry as registry

@registry.register_social("Soc_KV")
def social_kv_stepwise(params, states_t, neighbor_weights, aux_data_t):
    k_alg = params["Social"]["k_alg"]

    # 1. 状態から角度(orientations)を取得 (N,)
    # ※ states_t のキー名は実装に合わせて適宜変更してください
    theta_t = states_t["orientations"]

    # 2. ブロードキャストのための次元拡張
    # theta_self: 自分(i)の角度 (N, 1)
    # theta_nbr : 相手(j)の角度 (1, N)
    theta_self = theta_t[:, jnp.newaxis] 
    theta_nbr = theta_t[jnp.newaxis, :]  

    # 3. 角度差とサインの計算 (N, N)
    delta_theta_ij = theta_nbr - theta_self
    sin_delta_theta_ij = jnp.sin(delta_theta_ij)

    # 4. 重み行列による重み付け (N, N)
    # 自分自身への影響(i=j)は、neighbor_weightsの対角成分が0であればここで自動的に消えます
    interaction_weighted = sin_delta_theta_ij * neighbor_weights

    # 5. 各エージェント(i)について、全他個体(j)からの影響を足し合わせる (N,)
    sum_interaction = jnp.sum(interaction_weighted, axis=1)

    # 6. 結合係数 k_alg を掛けて最終的な social_factor を返す
    social_factor = k_alg * sum_interaction

    return social_factor
