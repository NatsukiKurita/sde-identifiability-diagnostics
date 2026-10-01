import jax
import jax.numpy as jnp

import src.parameter_estimation.registry as registry
import src.parameter_estimation.utils.geometry as geometry


def _elastic_force_from_contact(
    params,
    states_t,
    min_distances,
    nearest_normals,
    v_0,
    safe_eps,
):
    d_w = params["WallAvoidance"]["d_w"]
    k_w = params["WallAvoidance"]["k_w"]
    theta_t = states_t["orientations"]

    distance_safe = jnp.maximum(min_distances, safe_eps)
    d_w_safe = jnp.maximum(d_w, safe_eps)
    magnitude = jnp.where(
        min_distances < d_w,
        k_w * jnp.maximum(1.0 / distance_safe - 1.0 / d_w_safe, 0.0),
        0.0,
    )
    F_wall_masked = magnitude[:, jnp.newaxis] * nearest_normals

    s = jnp.sin(theta_t)
    c = jnp.cos(theta_t)
    heading_normals = jnp.stack([-s, c], axis=-1)
    return jnp.sum(F_wall_masked * heading_normals, axis=-1) / v_0


@registry.register_wall_avoidance("WA_Elastic")
def wa_elastic_stepwise(params, states_t, aux_data_t):
    corners = aux_data_t["corners"]
    start_walls, end_walls = geometry.build_wall_segments(corners)
    pos_t = states_t["positions"]
    safe_eps = aux_data_t["safe_eps"]
    v_0 = aux_data_t["v_0"]

    P = jnp.expand_dims(pos_t, axis=1)
    A = jnp.expand_dims(start_walls, axis=0)
    B = jnp.expand_dims(end_walls, axis=0)

    vec_AB = B - A
    vec_AP = P - A
    AB_squared = jnp.sum(vec_AB ** 2, axis=-1, keepdims=True) + safe_eps
    t = jnp.sum(vec_AP * vec_AB, axis=-1, keepdims=True) / AB_squared
    t_clipped = jnp.clip(t, 0.0, 1.0)
    C = A + t_clipped * vec_AB
    vec_CP = P - C
    dist_CP = jnp.linalg.norm(vec_CP, axis=-1)

    n_wall_temp = jnp.stack([-vec_AB[..., 1], vec_AB[..., 0]], axis=-1)
    vec_to_center = jnp.mean(start_walls, axis=0, keepdims=True) - A
    dot_prod = jnp.sum(n_wall_temp * vec_to_center, axis=-1, keepdims=True)
    n_wall = jnp.sign(dot_prod) * n_wall_temp / (
        jnp.linalg.norm(n_wall_temp, axis=-1, keepdims=True) + safe_eps
    )

    min_distances = jnp.min(dist_CP, axis=1)
    min_dist_indices = jnp.argmin(dist_CP, axis=1)
    num_walls = start_walls.shape[0]
    wall_mask = jax.nn.one_hot(min_dist_indices, num_classes=num_walls)
    wall_mask_expanded = jnp.expand_dims(wall_mask, axis=-1)
    nearest_normals = jnp.sum(n_wall * wall_mask_expanded, axis=1)

    return _elastic_force_from_contact(
        params,
        states_t,
        min_distances,
        nearest_normals,
        v_0,
        safe_eps,
    )


@registry.register_wall_avoidance_circle("WA_Elastic")
def wa_elastic_circle_stepwise(params, states_t, aux_data_t):
    pos_t = states_t["positions"]
    safe_eps = aux_data_t["safe_eps"]
    v_0 = aux_data_t["v_0"]
    min_distances, nearest_normals = geometry.compute_circle_wall_contact(
        pos_t,
        aux_data_t["center"],
        aux_data_t["radius"],
        safe_eps,
    )
    return _elastic_force_from_contact(
        params,
        states_t,
        min_distances,
        nearest_normals,
        v_0,
        safe_eps,
    )
