import jax.numpy as jnp


def neighbor_delaunay_stepwise(params, states_t, aux_data_t):
    # aux_dataからDelaunayの隣接行列を取得
    adjmat_t = aux_data_t["delaunay_neighbors"]  # shape: (N, E_max), -1はダミーエッジを表す
    N = states_t["positions"].shape[0]

    i_idx = jnp.arange(N)[:, None]  # shape: (N, 1)
    j_idx = adjmat_t  # shape: (N, E_max)

    # 1. まずは重み付けされていない 0 or 1 の密な隣接行列を作成する
    # ダミー(-1)は配列の最後尾(N番目)に書き込まれる
    adj_pad = jnp.zeros((N + 1, N + 1))
    
    # 繋がりがある箇所に 1.0 を立てる
    adj_pad = adj_pad.at[i_idx, j_idx].set(1.0) 

    # ダミー領域を切り捨て、N x N の有効な隣接行列を取得
    adj = adj_pad[:N, :N]

    # 2. 各個体の近傍数 (N_i) を行ごとの和として計算
    degrees = jnp.sum(adj, axis=1, keepdims=True)  # shape: (N, 1)

    # 3. 各行ごとに自身の近傍数で割って正規化 (ゼロ割り回避)
    w_ij = jnp.where(degrees > 0.0, adj / degrees, 0.0)

    return w_ij