import jax.numpy as jnp

import src.parameter_estimation.registry as registry
import src.parameter_estimation.utils.geometry as geometry


@registry.register_wall_avoidance("WA_G2012")
def wa_g2012_stepwise(params, states_t, aux_data_t):
    k_w = params["WallAvoidance"]["k_w"]
    safe_eps = aux_data_t["safe_eps"]
    corners = aux_data_t["corners"]
    start_walls, end_walls = geometry.build_wall_segments(corners)
    pos_t = states_t["positions"]
    theta_t = states_t["orientations"]
    cos_t = jnp.cos(theta_t)
    sin_t = jnp.sin(theta_t)
    dir_front = jnp.stack([cos_t, sin_t], axis=-1)
    ray_front = jnp.stack([cos_t, sin_t], axis=-1)

    d_i_w, n_i_w = geometry.ray_cast(pos_t, ray_front, start_walls, end_walls, safe_eps)
    d_i_w_safe = jnp.maximum(d_i_w, safe_eps)
    sign_phi_i_w = jnp.sign(geometry.cross_2d(n_i_w, dir_front))
    return k_w * sign_phi_i_w / d_i_w_safe


@registry.register_wall_avoidance_circle("WA_G2012")
def wa_g2012_circle_stepwise(params, states_t, aux_data_t):
    k_w = params["WallAvoidance"]["k_w"]
    safe_eps = aux_data_t["safe_eps"]
    pos_t = states_t["positions"]
    theta_t = states_t["orientations"]
    cos_t = jnp.cos(theta_t)
    sin_t = jnp.sin(theta_t)
    dir_front = jnp.stack([cos_t, sin_t], axis=-1)
    ray_front = jnp.stack([cos_t, sin_t], axis=-1)

    d_i_w, n_i_w = geometry.ray_cast_circle(
        pos_t,
        ray_front,
        aux_data_t["center"],
        aux_data_t["radius"],
        safe_eps,
    )
    d_i_w_safe = jnp.maximum(d_i_w, safe_eps)
    sign_phi_i_w = jnp.sign(geometry.cross_2d(n_i_w, dir_front))
    omega_w = k_w * sign_phi_i_w / d_i_w_safe
    return jnp.where(jnp.isinf(d_i_w), 0.0, omega_w)
