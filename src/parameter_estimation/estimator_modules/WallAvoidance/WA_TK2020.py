import jax.numpy as jnp

import src.parameter_estimation.registry as registry
import src.parameter_estimation.utils.geometry as geometry


def _tk2020_force_from_contact(k_wrep, k_wcoh, headings, nearest_normals):
    dot_prods = jnp.sum(nearest_normals * headings, axis=-1)
    dir_signs = jnp.sign(dot_prods)
    magnitudes = jnp.where(
        dir_signs < 0.0,
        k_wrep,
        jnp.where(dir_signs > 0.0, k_wcoh, 0.0),
    )
    wall_forces = magnitudes[:, jnp.newaxis] * nearest_normals
    normal_headings = jnp.stack([-headings[:, 1], headings[:, 0]], axis=-1)
    return jnp.sum(wall_forces * normal_headings, axis=-1)


@registry.register_wall_avoidance("WA_TK2020")
def wa_tk2020_stepwise(params, states_t, aux_data_t):
    k_wrep = params["WallAvoidance"]["k_wrep"]
    k_wcoh = params["WallAvoidance"]["k_wcoh"]
    safe_eps = aux_data_t["safe_eps"]
    corners = aux_data_t["corners"]
    start_walls, end_walls = geometry.build_wall_segments(corners)
    pos_t = states_t["positions"]
    theta_t = states_t["orientations"]

    s = jnp.sin(theta_t)
    c = jnp.cos(theta_t)
    headings = jnp.stack([c, s], axis=-1)
    _, nearest_normals = geometry.compute_perpendicular_foot(
        pos_t,
        start_walls,
        end_walls,
        safe_eps,
    )
    return _tk2020_force_from_contact(k_wrep, k_wcoh, headings, nearest_normals)


@registry.register_wall_avoidance_circle("WA_TK2020")
def wa_tk2020_circle_stepwise(params, states_t, aux_data_t):
    k_wrep = params["WallAvoidance"]["k_wrep"]
    k_wcoh = params["WallAvoidance"]["k_wcoh"]
    safe_eps = aux_data_t["safe_eps"]
    pos_t = states_t["positions"]
    theta_t = states_t["orientations"]

    s = jnp.sin(theta_t)
    c = jnp.cos(theta_t)
    headings = jnp.stack([c, s], axis=-1)
    _, nearest_normals = geometry.compute_circle_wall_contact(
        pos_t,
        aux_data_t["center"],
        aux_data_t["radius"],
        safe_eps,
    )
    return _tk2020_force_from_contact(k_wrep, k_wcoh, headings, nearest_normals)
