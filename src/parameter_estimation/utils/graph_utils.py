import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
from dataclasses import dataclass

def convert_neighbors_to_graph(neighbor_data: jnp.ndarray):
    """
    Julia出力 (T, N, K) -> JAX Graph形式 (senders, receivers, mask)
    """
    T, N, K = neighbor_data.shape # ここで形状を取得
    
    # --- 1. Senders (相手) の作成 ---
    flat_senders = neighbor_data.reshape(-1)
    
    # マスク作成 (-1 は無効)
    # juliaの隣接情報は1始まり、Pythonは0始まりなので、-2が無効値
    mask = (flat_senders != -2)
    
    # Julia (1~N, -1) -> Python (0~N-1, -2)
    shifted_senders = flat_senders - 1
    
    # 安全化: 無効な場所を0に置換
    safe_senders = jnp.where(mask, shifted_senders, 0)
    
    # --- 2. Receivers (自分) の作成 ---
    receivers_base = jnp.arange(N)
    receivers_3d = jnp.broadcast_to(receivers_base[None, :, None], (T, N, K))
    flat_receivers = receivers_3d.reshape(-1)

    # --- 3. 整形 ---
    # klamser_social_force は (T, E) の入力を期待するため、Time次元だけ残してリシェイプ
    # ここでの E は N * K (最大エッジ数)
    senders = safe_senders.reshape(T, -1)
    receivers = flat_receivers.reshape(T, -1)
    edge_mask = mask.reshape(T, -1)
    
    return senders, receivers, edge_mask