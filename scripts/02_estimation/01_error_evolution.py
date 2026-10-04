from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.experiment_configs.eigenvalue_time_series import EigenvalueTimeSeriesConfig
from src.experiment_configs.estimation import EstimationConfig
from src.QMLE.utils.trust_constr_optimizer import trust_constr_optimizer


def _analysis_window(states, states_next, aux_data, aux_axes):
	"""Keep only the post-burn-in trajectory used for estimation."""
	burn_in_frames = int(EigenvalueTimeSeriesConfig.BURN_IN)
	num_steps = int(states["positions"].shape[0])
	burn_in_frames = min(max(0, burn_in_frames), max(0, num_steps - 1))
	if burn_in_frames == 0:
		return states, states_next, aux_data, 0

	analysis_states = {key: value[burn_in_frames:] for key, value in states.items()}
	analysis_states_next = {key: value[burn_in_frames:] for key, value in states_next.items()}
	analysis_aux_data = {
		key: value[burn_in_frames:] if aux_axes.get(key) == 0 else value
		for key, value in aux_data.items()
	}
	return analysis_states, analysis_states_next, analysis_aux_data, burn_in_frames


def _build_time_points(num_steps: int, dt: float, interval_seconds: float) -> tuple[np.ndarray, np.ndarray]:
	if num_steps <= 0:
		return np.array([], dtype=int), np.array([], dtype=float)

	interval_steps = max(1, int(round(float(interval_seconds) / float(dt))))
	time_indices = np.arange(interval_steps - 1, num_steps, interval_steps, dtype=int)
	if time_indices.size == 0 or time_indices[-1] != num_steps - 1:
		time_indices = np.append(time_indices, num_steps - 1)
	time_indices = np.unique(time_indices)
	return time_indices, (time_indices + 1) * float(dt)


def _serialize_param_tree(tree):
	if tree is None:
		return {}
	if isinstance(tree, dict):
		return {key: _serialize_param_tree(value) for key, value in tree.items()}
	array = np.asarray(tree)
	return array.item() if array.size == 1 else array.tolist()


def estimate_parameter_trend(
	loss_fn_stepwise,
	initial_params,
	lower_bounds,
	upper_bounds,
	states,
	states_next,
	aux_data,
	aux_axes,
	dt: float,
	interval_seconds: float,
) -> tuple[dict, dict]:
	"""Re-estimate parameters from cumulative post-burn-in trajectory prefixes."""
	states, states_next, aux_data, burn_in_frames = _analysis_window(
		states,
		states_next,
		aux_data,
		aux_axes,
	)
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

		def prefix_loss(params):
			return jnp.sum(
				jax.vmap(loss_fn_stepwise, in_axes=(None, 0, 0, aux_axes))(
					params,
					prefix_states,
					prefix_states_next,
					prefix_aux_data,
				)
			)

		print(f"Estimating cumulative parameters at T={float(time_value):g} s...")
		pre_params, _, presearch_iters, pre_meta = trust_constr_optimizer(
			prefix_loss,
			initial_params,
			lower_bounds,
			upper_bounds,
			tol=EstimationConfig.TOL_PRESEARCH_TRUST_CONSTR,
			xtol=EstimationConfig.XTOL_PRESEARCH_TRUST_CONSTR,
			maxiter=EstimationConfig.MAXITER_PRESEARCH_TRUST_CONSTR,
			return_metadata=True,
		)
		best_params, best_loss, iterations, meta = trust_constr_optimizer(
			prefix_loss,
			pre_params,
			lower_bounds,
			upper_bounds,
			return_metadata=True,
		)
		estimates.append(
			{
				"time_seconds": float(time_value),
				"time_index": int(time_index),
				"num_steps": num_steps,
				"mean_loss": float(best_loss / float(num_agents * num_steps)),
				"presearch_iterations": int(presearch_iters),
				"iterations": int(iterations),
				"presearch_message": str(pre_meta.get("message", "")),
				"message": str(meta.get("message", "")),
				"best_params": _serialize_param_tree(best_params),
			}
		)
		final_result = {
			"best_params": best_params,
			"presearch_iterations": int(presearch_iters),
			"presearch_meta": pre_meta,
			"iterations": int(iterations),
			"meta": meta,
		}

	if final_result is None:
		raise ValueError("Cannot estimate a parameter trend from an empty trajectory.")

	return {
		"interval_seconds": float(interval_seconds),
		"burn_in_frames": int(burn_in_frames),
		"burn_in_seconds": float(burn_in_frames) * float(dt),
		"time_seconds": [float(value) for value in time_seconds],
		"time_indices": [int(value) for value in time_indices],
		"estimates": estimates,
	}, final_result
