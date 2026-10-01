from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from src.constants.estimation import EstimationConfig


def build_wall_aux_entries(
	config_toml: dict,
	corners_np: np.ndarray,
	boundary_type: str,
) -> dict:
	entries = {
		"corners": jnp.array(corners_np, dtype=EstimationConfig.JNP_FLOAT_DTYPE),
		"boundary_shape": boundary_type,
	}
	if boundary_type == "CircleWall":
		center = config_toml["Boundary"]["center"]
		radius = config_toml["Boundary"]["radius"]
		entries["center"] = jnp.asarray(center, dtype=EstimationConfig.JNP_FLOAT_DTYPE).reshape(2)
		entries["radius"] = EstimationConfig.as_float_array(radius)
	else:
		entries["center"] = jnp.zeros(2, dtype=EstimationConfig.JNP_FLOAT_DTYPE)
		entries["radius"] = EstimationConfig.as_float_array(0.0)
	return entries


WALL_AUX_AXES = {
	"dt": None,
	"corners": None,
	"center": None,
	"radius": None,
	"boundary_shape": None,
	"v_0": None,
	"safe_eps": None,
	"delaunay_neighbors": 0,
}
