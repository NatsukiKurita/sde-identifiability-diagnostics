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
    safe_eps = aux_data_t["safe_eps"]  # 数値安定化のための小さな値
    
    N = pos_t.shape[0]

    # 自分自身への相互作用(i == j)を排除するためのマスク
    # 既に対角成分が0の場合は不要ですが、念のための安全策です
    neighbor_weights_no_self = neighbor_weights * (1.0 - jnp.eye(N))

    # ==========================================
    # 1. 反発力の計算 (Repulsion)
    # ==========================================
    # ブロードキャストで全ペアの相対ベクトル r_ij を一括計算
    # pos_j (1, N, 2) - pos_i (N, 1, 2) -> r_ij (N, N, 2)
    r_ij = pos_t[jnp.newaxis, :, :] - pos_t[:, jnp.newaxis, :]

    d_ij = jnp.sqrt(jnp.sum(r_ij**2, axis=-1) + safe_eps**2) # (N, N) 各ペアの距離。safe_epsはゼロ割り防止のための微小値
    
    # sep_force_ij: (N, N, 2)
    sep_force_ij = -k_sep * (r_ij / d_ij[..., jnp.newaxis]) * neighbor_weights_no_self[..., jnp.newaxis]
    sep_force = jnp.sum(sep_force_ij, axis=1)            # (N, 2)

    # ==========================================
    # 2. 吸引力・整列力の計算 (Cohesion & Alignment)
    # ==========================================
    # 重みの正規化項 (N, 1)
    regularization = jnp.sum(neighbor_weights_no_self, axis=1, keepdims=True) + safe_eps
    
    # 正規化された重み W_ij / sum(W) -> (N, N)
    norm_weights = neighbor_weights_no_self / regularization

    # 他個体の重心 (N, 2)
    com_others = jnp.sum(pos_t[jnp.newaxis, :, :] * norm_weights[..., jnp.newaxis], axis=1)
    # 他個体の平均速度 (N, 2)
    avg_vel_others = jnp.sum(vel_t[jnp.newaxis, :, :] * norm_weights[..., jnp.newaxis], axis=1)

    # 吸引力 (Cohesion)
    dist_to_com = jnp.sqrt(jnp.sum((com_others - pos_t)**2, axis=-1, keepdims=True) + safe_eps**2)
    coh_force = k_coh * (com_others - pos_t) / dist_to_com         # (N, 2)

    # 整列力 (Alignment)
    diff_to_avg_vel = avg_vel_others - vel_t        # (N, 2)
    norm_diff_vel = jnp.sqrt(jnp.sum(diff_to_avg_vel**2, axis=-1, keepdims=True) + safe_eps**2)
    alg_force = k_alg * diff_to_avg_vel / norm_diff_vel  # (N, 2)

    # ==========================================
    # 3. 合計と法線方向への射影 (Projection)
    # ==========================================
    total_force = sep_force + coh_force + alg_force      # (N, 2)

    # 進行方向に対する法線ベクトル SVector{2, T}(-sin(θ), cos(θ)) に対応 -> (N, 2)
    normal_vec = jnp.stack([-jnp.sin(theta_t), jnp.cos(theta_t)], axis=-1)

    # 内積 (dot) を要素ごとの積と和で計算 -> (N,)
    social_factor = jnp.sum(total_force * normal_vec, axis=-1)

    return social_factor