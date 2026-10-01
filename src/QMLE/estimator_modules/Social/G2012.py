import jax.numpy as jnp

import src.parameter_estimation.registry as registry
import src.parameter_estimation.utils.geometry as geometry

@registry.register_social("Soc_G2012")
def social_g2012_stepwise(params, states_t, neighbor_weights, aux_data_t):
    k_pos = params["Social"]["k_pos"]
    k_alg = params["Social"]["k_alg"]

    safe_eps = aux_data_t["safe_eps"]  # 数値安定化のための小さな値
    pos_t = states_t["positions"]  # shape: (N, 2)
    theta_t = states_t["orientations"]  # shape: (N,)

    # 1. ブロードキャストのための次元拡張
    # pos_self: 自分(i)の座標。行方向にN個並べるため (N, 1, 2)
    # pos_nbr:  相手(j)の座標。列方向にN個並べるため (1, N, 2)
    pos_self = pos_t[:, jnp.newaxis, :]  # (N, 1, 2)
    pos_nbr = pos_t[jnp.newaxis, :, :]   # (1, N, 2)
    
    theta_self = theta_t[:, jnp.newaxis] # (N, 1)
    theta_nbr = theta_t[jnp.newaxis, :]  # (1, N)

    # 2. 安全な相対ベクトルの計算 (N, N, 2)
    r_ij_vec = pos_nbr - pos_self
    
    # 対角成分（自分自身, 距離0）などで勾配が NaN になるのを防ぐためのマスク処理
    # 重みが 0 でない箇所だけ正しいベクトルを使い、それ以外は safe_vec に逃がす
    mask_expanded = jnp.abs(neighbor_weights)[..., jnp.newaxis] # (N, N, 1)
    safe_vec = jnp.array([1.0, 0.0], dtype=r_ij_vec.dtype)
    r_ij_vec_safe = jnp.where(mask_expanded > 0, r_ij_vec, safe_vec)

    # 3. 距離と単位ベクトルの計算
    d_ij = jnp.linalg.norm(r_ij_vec_safe, axis=-1)            # (N, N)
    d_ij_safe = jnp.maximum(d_ij, safe_eps)                   # (N, N)
    e_ij = r_ij_vec_safe / d_ij_safe[..., jnp.newaxis]        # (N, N, 2)
    
    # 4. 方向ベクトルの構築 (N, 1, 2) と (1, N, 2)
    dir_self = jnp.stack([jnp.cos(theta_self), jnp.sin(theta_self)], axis=-1)
    dir_nbr = jnp.stack([jnp.cos(theta_nbr), jnp.sin(theta_nbr)], axis=-1)

    # 5. 外積を用いた相互作用力の計算（ブロードキャストで (N, N) に展開される）
    sin_phi_ij = geometry.cross_2d(dir_self, e_ij)                      # (N, N)
    omega_pos_k = k_pos * d_ij * sin_phi_ij # (N, N)

    sin_delta_theta_ij = geometry.cross_2d(dir_self, dir_nbr)          # (N, N)
    omega_alg_k = k_alg * sin_delta_theta_ij          # (N, N)

    # 6. 重み行列による重み付け (マスキングを兼ねる)
    omega_pos_weighted = omega_pos_k * neighbor_weights        # (N, N)
    omega_alg_weighted = omega_alg_k * neighbor_weights        # (N, N)

    # i番目のエージェントが受ける影響を全エージェント(j)について足し合わせる
    sum_pos = jnp.sum(omega_pos_weighted, axis=1)             # (N,)
    sum_alg = jnp.sum(omega_alg_weighted, axis=1)             # (N,)

    return (sum_pos + sum_alg) 
