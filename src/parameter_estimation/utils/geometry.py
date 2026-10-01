import jax
import jax.numpy as jnp
from typing import Tuple

from src.constants.estimation import EstimationConfig


def build_wall_segments(corners):
    corners_arr = jnp.array(corners, dtype=EstimationConfig.JNP_FLOAT_DTYPE).reshape(-1, 2)
    center = jnp.mean(corners_arr, axis=0)

    diff = corners_arr - center
    angles = jnp.arctan2(diff[:,1], diff[:,0])
    sorted_indices = jnp.argsort(angles)
    sorted_corners = corners_arr[sorted_indices]
    
    start_walls = sorted_corners
    end_walls = jnp.roll(sorted_corners, shift=-1, axis=0)
    return start_walls, end_walls


def cross_2d(a, b): # a×b
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def ray_cast(
    pos: jnp.ndarray,        # (N, 2)
    ray_dirs: jnp.ndarray,   # (N, 2)
    start_walls: jnp.ndarray,# (M, 2)
    end_walls: jnp.ndarray,  # (M, 2)
    safe_eps: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    指定されたレイ方向に対する壁との交差計算 (Tensor版)
    Returns:
        min_dist: (N) 最も近い壁までの距離 (hitしない場合は inf)
        normal:   (N, 2) その壁の法線ベクトル
    """

    P = pos[:, jnp.newaxis, :]                                 # (N, 1, 2)
    D = ray_dirs[:, jnp.newaxis, :]                            # (N, 1, 2)
    S = start_walls[jnp.newaxis, :, :]               # (1, M, 2)
    V = (end_walls - start_walls)[jnp.newaxis, :, :] # (1, M, 2)

    # 交差計算
    denom = cross_2d(D, V) # (N, M)

    # 交差角度
    V_unit = V / (jnp.linalg.norm(V, axis=-1, keepdims=True) + safe_eps)
    D_unit = D / (jnp.linalg.norm(D, axis=-1, keepdims=True) + safe_eps)
    angle = cross_2d(D_unit, V_unit)  # (N, M)

    # 平行判定
    is_parallel = jnp.abs(angle) < safe_eps
    denom_safe = jnp.where(is_parallel, 1.0, denom)

    # t: レイ上の距離, u: 壁セグメント上の割合
    val = S - P
    t = cross_2d(val, V) / denom_safe # (N, M)
    u = cross_2d(val, D) / denom_safe # (N, M)

    # ヒット判定
    is_hit = (~is_parallel) & (t > 0.0) & (u >= 0.0) & (u <= 1.0) # (N, M)
    dist = jnp.where(is_hit, t, jnp.inf) # (N, M)

    # 最近傍壁の選択
    min_dist = jnp.min(dist, axis=-1) # (N)
    min_idx = jnp.argmin(dist, axis=-1) # (N)

    # 法線ベクトルの取得
    # 壁法線の基準向きは (-y, x) とし、後で各レイ方向に対して前向きに反転する。
    V_base = end_walls - start_walls # (M, 2)
    norm_vecs = jnp.stack([-V_base[..., 1], V_base[..., 0]], axis=-1) # (1, M, 2)
    norm_vecs = norm_vecs / (jnp.linalg.norm(norm_vecs, axis=-1, keepdims=True) + safe_eps)

    # 最近接壁の法線を取得
    nearest_normal = norm_vecs[min_idx]

    # ヒットしなかった場合は法線をゼロにする (Julia の explicit if r < Inf と同等)
    # これにより、min_idx が arbitrary な値を返す場合でも gradient が汚染されない
    nearest_normal = jnp.where(
        jnp.isinf(min_dist)[:, None],  # (N, 1) for broadcasting with (N, 2)
        jnp.zeros_like(nearest_normal),
        nearest_normal
    )

    # Julia 側と同様に、法線がレイ方向に対して前向きになるよう符号をそろえる。
    # dot(normal, ray_dir) >= 0 を満たす向きに反転する。
    dot_nd = jnp.sum(nearest_normal * ray_dirs, axis=-1, keepdims=True)
    nearest_normal = jnp.where(dot_nd < 0.0, -nearest_normal, nearest_normal)

    return min_dist, nearest_normal


def compute_perpendicular_foot(
    pos: jnp.ndarray,        # (N, 2) : エージェントの座標
    start_walls: jnp.ndarray,# (M, 2) : 壁セグメントの始点
    end_walls: jnp.ndarray,  # (M, 2) : 壁セグメントの終点
    safe_eps: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    各エージェントに対して、最近接の壁までの最短距離と、その壁の法線ベクトルを計算します。
    
    Returns:
        min_distances: (N,) 各エージェントから最も近い壁までの距離
        nearest_normals: (N, 2) その最近接壁の法線ベクトル（内側向き）
    """
    P = jnp.expand_dims(pos, axis=1)          # (N, 1, 2)
    A = jnp.expand_dims(start_walls, axis=0)  # (1, M, 2)
    B = jnp.expand_dims(end_walls, axis=0)    # (1, M, 2)

    # --- 1. 線分上の最短点 C を求める ---
    vec_AB = B - A  # (1, M, 2)
    vec_AP = P - A  # (N, M, 2)

    # 射影パラメータ t = (AP dot AB) / |AB|^2
    AB_squared = jnp.sum(vec_AB ** 2, axis=-1, keepdims=True) + safe_eps  # (1, M, 1)
    t = jnp.sum(vec_AP * vec_AB, axis=-1, keepdims=True) / AB_squared     # (N, M, 1)
    t_clipped = jnp.clip(t, 0.0, 1.0)                                     # (N, M, 1)

    # 線分上の最短点 C
    C = A + t_clipped * vec_AB  # (N, M, 2)

    # --- 2. 距離の計算 ---
    vec_CP = P - C # CからPに向かう向き (N, M, 2)
    
    # 後の処理を扱いやすくするため keepdims=False とし (N, M) の形状にする
    dist_CP = jnp.linalg.norm(vec_CP, axis=-1)  # (N, M)

    # --- 3. 壁固定の法線の計算 ---
    # 壁の法線 (1, M, 2)
    n_wall_temp = jnp.stack([-vec_AB[..., 1], vec_AB[..., 0]], axis=-1)
    
    # cornersの代わりに start_walls の平均を重心として利用し、中心へ向かうベクトルを作成
    vec_to_center = jnp.mean(start_walls, axis=0, keepdims=True) - A  # (1, 2) - (1, M, 2) -> (1, M, 2)
    dot_prod = jnp.sum(n_wall_temp * vec_to_center, axis=-1, keepdims=True)
    
    # 中心を向くように符号を調整し正規化
    n_wall = jnp.sign(dot_prod) * n_wall_temp / (jnp.linalg.norm(n_wall_temp, axis=-1, keepdims=True) + safe_eps)

    # --- 4. 最近接の抽出 ---
    # 各エージェントにとっての最小距離 (N,)
    min_distances = jnp.min(dist_CP, axis=1)

    # 最近接の壁のインデックス (N,)
    min_dist_indices = jnp.argmin(dist_CP, axis=1)
    
    # one-hotマスクを用いて、(1, M, 2) の全壁法線群から該当するものを (N, 2) として抽出
    num_walls = start_walls.shape[0]
    wall_mask = jax.nn.one_hot(min_dist_indices, num_classes=num_walls)  # (N, M)
    wall_mask_expanded = jnp.expand_dims(wall_mask, axis=-1)             # (N, M, 1)

    # ブロードキャスト積 (N, M, 2) -> M方向に和をとることで該当する法線だけが残る (N, 2)
    nearest_normals = jnp.sum(n_wall * wall_mask_expanded, axis=1)

    # 壁がない場合（distance = Inf）は法線をゼロ化
    # これにより gradient が汚染されず、後続の計算で自動的に力=0 になる
    nearest_normals = jnp.where(
        jnp.isinf(min_distances)[:, None],  # (N, 1) for broadcasting with (N, 2)
        jnp.zeros_like(nearest_normals),
        nearest_normals
    )

    return min_distances, nearest_normals


def compute_circle_wall_contact(
    pos: jnp.ndarray,
    center,
    radius: float,
    safe_eps: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Distance to a circular boundary and inward-pointing normals (Julia CircleWall)."""
    center_vec = jnp.asarray(center, dtype=EstimationConfig.JNP_FLOAT_DTYPE).reshape(1, 2)
    radius_value = EstimationConfig.as_float_array(radius)
    vec_from_center = pos - center_vec
    dist_from_center = jnp.linalg.norm(vec_from_center, axis=-1)
    dist_to_wall = radius_value - dist_from_center

    safe_dist = jnp.maximum(dist_from_center, safe_eps)
    unit_from_center = vec_from_center / safe_dist[:, None]
    nearest_normals = -unit_from_center
    nearest_normals = jnp.where(
        (dist_from_center > safe_eps)[:, None],
        nearest_normals,
        jnp.zeros_like(nearest_normals),
    )
    return dist_to_wall, nearest_normals


def ray_cast_circle(
    pos: jnp.ndarray,
    ray_dirs: jnp.ndarray,
    center,
    radius: float,
    safe_eps: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Ray-circle intersection for forward wall sensing (Julia WA_G2012 CircleWall)."""
    center_vec = jnp.asarray(center, dtype=EstimationConfig.JNP_FLOAT_DTYPE).reshape(2)
    radius_value = EstimationConfig.as_float_array(radius)
    oc = pos - center_vec
    b = 2.0 * jnp.sum(oc * ray_dirs, axis=-1)
    c = jnp.sum(oc * oc, axis=-1) - radius_value ** 2
    discriminant = b ** 2 - 4.0 * c
    valid = discriminant >= 0.0
    sqrt_disc = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    t1 = (-b - sqrt_disc) * 0.5
    t2 = (-b + sqrt_disc) * 0.5
    t_candidates = jnp.stack([t1, t2], axis=-1)
    t_candidates = jnp.where(t_candidates >= 0.0, t_candidates, jnp.inf)
    min_dist = jnp.min(t_candidates, axis=-1)
    min_dist = jnp.where(valid, min_dist, jnp.inf)

    collision_point = pos + min_dist[:, None] * ray_dirs
    normal = collision_point - center_vec
    normal_len = jnp.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / jnp.maximum(normal_len, safe_eps)
    normal = jnp.where(jnp.isinf(min_dist)[:, None], jnp.zeros_like(normal), normal)
    return min_dist, normal