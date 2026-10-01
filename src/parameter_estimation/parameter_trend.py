from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.constants.estimation import EstimationConfig
from src.parameter_estimation.cores.trust_constr_optimizer import trust_constr_optimizer


def _build_time_points(num_steps: int, dt: float, interval_seconds: float) -> tuple[np.ndarray, np.ndarray]:
	if num_steps <= 0:
		return np.array([], dtype=int), np.array([], dtype=float)

	interval_steps = max(1, int(round(float(interval_seconds) / float(dt))))
	time_indices = np.arange(interval_steps - 1, num_steps, interval_steps, dtype=int)
	if time_indices.size == 0 or time_indices[-1] != num_steps - 1:
		time_indices = np.append(time_indices, num_steps - 1)
	time_indices = np.unique(time_indices)
	return time_indices, (time_indices + 1) * float(dt)


def _split_active_params(params_tree, activity_tree):
	if isinstance(activity_tree, dict):
		active_out = {}
		for key in activity_tree:
			child = _split_active_params(params_tree[key], activity_tree[key])
			if child is not None:
				active_out[key] = child
		return active_out if active_out else None
	return params_tree if bool(np.asarray(activity_tree)) else None


def _serialize_param_tree(tree):
	if tree is None:
		return {}
	if isinstance(tree, dict):
		return {key: _serialize_param_tree(value) for key, value in tree.items()}
	array = np.asarray(tree)
	return array.item() if array.size == 1 else array.tolist()


def estimate_parameter_trend(
	loss_fn_stepwise,
	fixed_params,
	is_active_tree,
	active_initial_params,
	active_lower,
	active_upper,
	states,
	states_next,
	aux_data,
	aux_axes,
	dt: float,
	interval_seconds: float,
) -> tuple[dict, dict]:
	"""Re-estimate parameters from cumulative trajectory prefixes."""
	time_indices, time_seconds = _build_time_points(
		states["positions"].shape[0],
		dt,
		interval_seconds,
	)
	estimates = []
	final_result = None
	num_agents = int(states["positions"].shape[1])

	for time_index, time_value in zip(time_indices, time_seconds):
		num_steps = int(time_index) + 1
		prefix_states = {key: value[:num_steps] for key, value in states.items()}
		prefix_states_next = {key: value[:num_steps] for key, value in states_next.items()}
		prefix_aux_data = {
			key: value[:num_steps] if aux_axes.get(key) == 0 else value
			for key, value in aux_data.items()
		}

		def prefix_loss_active_only(active_params):
			full_params = jax.tree_util.tree_map(
				lambda fixed, active, mask: active if mask else fixed,
				fixed_params,
				active_params,
				is_active_tree,
			)
			return jnp.sum(
				jax.vmap(loss_fn_stepwise, in_axes=(None, 0, 0, aux_axes))(
					full_params,
					prefix_states,
					prefix_states_next,
					prefix_aux_data,
				)
			)

		print(f"Estimating cumulative parameters at T={float(time_value):g} s...")
		pre_params, _, presearch_iters, pre_meta = trust_constr_optimizer(
			prefix_loss_active_only,
			active_initial_params,
			active_lower,
			active_upper,
			tol=EstimationConfig.TOL_PRESEARCH_TRUST_CONSTR,
			xtol=EstimationConfig.XTOL_PRESEARCH_TRUST_CONSTR,
			maxiter=EstimationConfig.MAXITER_PRESEARCH_TRUST_CONSTR,
			return_metadata=True,
		)
		best_active_params, best_loss, iterations, meta = trust_constr_optimizer(
			prefix_loss_active_only,
			pre_params,
			active_lower,
			active_upper,
			return_metadata=True,
		)
		best_params = jax.tree_util.tree_map(
			lambda fixed, active, mask: active if mask else fixed,
			fixed_params,
			best_active_params,
			is_active_tree,
		)
		estimates.append(
			{
				"time_seconds": float(time_value),
				"time_index": int(time_index),
				"num_steps": num_steps,
				"best_loss": float(best_loss),
				"loss_opt_norm": float(best_loss / float(num_agents * num_steps)),
				"presearch_iterations": int(presearch_iters),
				"iterations": int(iterations),
				"presearch_message": str(pre_meta.get("message", "")),
				"message": str(meta.get("message", "")),
				"best_params": _serialize_param_tree(_split_active_params(best_params, is_active_tree)),
			}
		)
		final_result = {
			"best_active_params": best_active_params,
			"best_loss": float(best_loss),
			"presearch_iterations": int(presearch_iters),
			"presearch_meta": pre_meta,
			"iterations": int(iterations),
			"meta": meta,
		}

	if final_result is None:
		raise ValueError("Cannot estimate a parameter trend from an empty trajectory.")

	return {
		"interval_seconds": float(interval_seconds),
		"time_seconds": [float(value) for value in time_seconds],
		"time_indices": [int(value) for value in time_indices],
		"estimates": estimates,
	}, final_result
