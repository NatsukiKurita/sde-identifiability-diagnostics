#!/usr/bin/env python3
"""Compute the Hessian at the saved optimum and its minimum eigenvalue.

For each estimated trajectory under ``results/<setting>/``,
this script:

1. loads the optimal parameters written by ``02_estimation``,
2. computes the cumulative Hessian at the optimum and at truth parameters,
3. stores the Hessian, eigenpairs, and minimum-eigenvalue time series.

Outputs (per model / agent count / simulation):
- updates ``summary.json`` with the minimum and maximum eigenvalues
- ``results.h5`` — Hessian, eigenpairs, and the optimum trend
- ``eval_trend_at_opt.json`` — trend series + evaluation metrics at optimum
- ``eval_trend_at_truth.json`` — trend series + evaluation metrics at truth
"""

import argparse
import functools
import json
import math
import sys
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import toml
from jax.flatten_util import ravel_pytree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import src.QMLE.builder as builder
from src.experiment_configs.estimation import EstimationConfig
from src.experiment_configs.eigenvalue_time_series import EigenvalueTimeSeriesConfig
from src.experiment_configs.simulation import SocialSimConfig
from src.paths import DATA_ROOT, RESULTS_ROOT

EstimationConfig.configure_jax_precision()

DEFAULT_SIMDATA_ROOT = DATA_ROOT / "simulation"

BURN_IN_SECONDS = float(EigenvalueTimeSeriesConfig.BURN_IN * EigenvalueTimeSeriesConfig.TIME_STEP)


def _split_params_by_activity(params_tree, activity_tree):
	"""Split params tree into active and nonactive subtrees according to boolean mask tree."""
	if isinstance(activity_tree, dict):
		active_out = {}
		nonactive_out = {}
		for key in activity_tree:
			a_child, n_child = _split_params_by_activity(params_tree[key], activity_tree[key])
			if a_child is not None:
				active_out[key] = a_child
			if n_child is not None:
				nonactive_out[key] = n_child
		return (active_out if active_out else None), (nonactive_out if nonactive_out else None)

	is_active = bool(np.asarray(activity_tree))
	if is_active:
		return params_tree, None
	return None, params_tree


def _serialize_param_tree(tree):
	if tree is None:
		return {}
	if isinstance(tree, dict):
		return {k: _serialize_param_tree(v) for k, v in tree.items()}

	arr = np.asarray(tree)
	if arr.size == 1:
		return arr.item()
	return arr.tolist()


def _build_param_mapping(tree):
	if tree is None:
		return []

	mapping = []
	for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
		path_str = ".".join([str(p.key) for p in path])
		leaf_arr = np.asarray(leaf)
		if leaf_arr.size == 1:
			mapping.append(path_str)
		else:
			for i in range(leaf_arr.size):
				mapping.append(f"{path_str}[{i}]")
	return mapping


def _build_analysis_time_points(num_steps: int, dt: float, interval_seconds: float):
	if num_steps <= 0:
		return np.array([], dtype=int), np.array([], dtype=float)

	interval_steps = max(1, int(round(float(interval_seconds) / float(dt))))
	time_indices = np.arange(interval_steps - 1, num_steps, interval_steps, dtype=int)
	if time_indices.size == 0 or time_indices[-1] != num_steps - 1:
		time_indices = np.append(time_indices, num_steps - 1)
	time_indices = np.unique(time_indices)
	time_seconds = (time_indices + 1) * float(dt)
	return time_indices, time_seconds


def _find_burn_in_start_index(time_seconds: list[float], burn_in_seconds: float) -> int:
	for index, time_seconds_value in enumerate(time_seconds):
		if math.isfinite(time_seconds_value) and time_seconds_value >= burn_in_seconds - 1e-9:
			return index
	return len(time_seconds)


def _compute_sign_reversal_stats(values: list[float], start_index: int) -> tuple[int, int, float]:
	finite_values = [value for value in values[max(0, int(start_index)):] if math.isfinite(value)]
	if len(finite_values) < 2:
		return 0, 0, 0.0
	pairs = list(zip(finite_values[:-1], finite_values[1:]))
	count = sum(1 for a, b in pairs if a * b < 0)
	total = len(pairs)
	rate = float(count) / float(total) if total > 0 else 0.0
	return int(count), int(total), float(rate)


def _build_trend_series(hessian_trend: dict) -> dict:
	return {
		"analysis_start_frame": int(hessian_trend["analysis_start_frame"]),
		"analysis_num_steps": int(hessian_trend["analysis_num_steps"]),
		"interval_seconds": float(hessian_trend["interval_seconds"]),
		"interval_steps": int(hessian_trend["interval_steps"]),
		"time_seconds": [float(x) for x in hessian_trend["time_seconds"]],
		"time_indices": [int(x) for x in hessian_trend["time_indices"]],
		"min_eigenvalues": [float(x) for x in hessian_trend["min_eigenvalues"]],
	}


def _build_eval_trend_payload(hessian_trend: dict) -> dict:
	trend_series = _build_trend_series(hessian_trend)
	time_seconds = trend_series["time_seconds"]
	min_eigenvalues = trend_series["min_eigenvalues"]
	burn_in_start_index = _find_burn_in_start_index(time_seconds, BURN_IN_SECONDS)
	sign_reversal_count, sign_reversal_total, sign_reversal_rate = _compute_sign_reversal_stats(
		min_eigenvalues,
		burn_in_start_index,
	)
	burn_in_start_time_seconds = time_seconds[burn_in_start_index] if burn_in_start_index < len(time_seconds) else None

	payload = dict(trend_series)
	payload["evaluation"] = {
		"num_points": int(len(time_seconds)),
		"sign_reversal_count": int(sign_reversal_count),
		"sign_reversal_total": int(sign_reversal_total),
		"sign_reversal_rate": float(sign_reversal_rate),
		"burn_in_start_index": int(burn_in_start_index),
		"burn_in_start_time_seconds": burn_in_start_time_seconds,
	}
	return payload


def _build_analysis_aux_data(aux_data: dict, analysis_start_frame: int) -> dict:
	analysis_aux_data = {}
	for key, value in aux_data.items():
		if key == "delaunay_neighbors":
			analysis_aux_data[key] = value[analysis_start_frame:]
		else:
			analysis_aux_data[key] = value
	return analysis_aux_data


def _compute_min_eigenvalue_trend(
	loss_fn_stepwise,
	fixed_params,
	is_active_tree,
	active_params,
	states,
	states_next,
	aux_data,
	aux_axes,
	dt: float,
	interval_seconds: float,
	analysis_start_frame: int,
):
	flat_active_params, unravel_active_fn = ravel_pytree(active_params)
	active_dim = int(flat_active_params.size)

	if active_dim == 0:
		return {
			"analysis_start_frame": int(analysis_start_frame),
			"analysis_num_steps": 0,
			"interval_seconds": float(interval_seconds),
			"interval_steps": int(max(1, int(round(float(interval_seconds) / float(dt))))),
			"time_seconds": [],
			"time_indices": [],
			"min_eigenvalues": [],
			"final_hessian": np.zeros((0, 0), dtype=float),
			"final_eigenvalues": np.zeros((0,), dtype=float),
			"final_eigenvectors": np.zeros((0, 0), dtype=float),
		}

	def step_loss_active_only(active_params, states_t, state_next, aux_data_t):
		full_params = jax.tree_util.tree_map(
			lambda fixed, active, mask: active if mask else fixed,
			fixed_params,
			active_params,
			is_active_tree,
		)
		return loss_fn_stepwise(full_params, states_t, state_next, aux_data_t)

	def flat_step_loss_fn(flat_params, states_t, state_next, aux_data_t):
		return step_loss_active_only(unravel_active_fn(flat_params), states_t, state_next, aux_data_t)

	hessian_fn = jax.jit(jax.hessian(flat_step_loss_fn))
	step_hessians = jax.vmap(hessian_fn, in_axes=(None, 0, 0, aux_axes))(
		flat_active_params,
		states,
		states_next,
		aux_data,
	)

	# ヘシアンの無次元化
	weights_active = flat_active_params[jnp.newaxis, :] * flat_active_params[:, jnp.newaxis]
	step_hessians_unit_free = step_hessians * weights_active
	cumulative_hessians = jnp.cumsum(step_hessians_unit_free, axis=0)

	time_indices, time_seconds = _build_analysis_time_points(cumulative_hessians.shape[0], dt, interval_seconds)
	sampled_min_eigenvalues = []
	for idx in time_indices:
		current_hessian = cumulative_hessians[idx]
		current_eigenvalues = jnp.linalg.eigvalsh(current_hessian)
		sampled_min_eigenvalues.append(float(jnp.min(current_eigenvalues)))

	final_hessian = cumulative_hessians[-1]
	final_eigenvalues, final_eigenvectors = jnp.linalg.eigh(final_hessian)
	final_idx = jnp.argsort(final_eigenvalues)[::-1]
	final_eigenvalues = final_eigenvalues[final_idx]
	final_eigenvectors = final_eigenvectors[:, final_idx]

	return {
		"analysis_start_frame": int(analysis_start_frame),
		"analysis_num_steps": int(cumulative_hessians.shape[0]),
		"interval_seconds": float(interval_seconds),
		"interval_steps": int(max(1, int(round(float(interval_seconds) / float(dt))))),
		"time_seconds": [float(x) for x in np.asarray(time_seconds, dtype=float)],
		"time_indices": [int(x) for x in np.asarray(time_indices, dtype=int)],
		"min_eigenvalues": sampled_min_eigenvalues,
		"final_hessian": np.asarray(final_hessian),
		"final_eigenvalues": np.asarray(final_eigenvalues),
		"final_eigenvectors": np.asarray(final_eigenvectors),
	}


def _restore_active_params(saved_active, template_tree, is_active_tree):
	"""Rebuild the optimizer's active tree from the serialized optimum."""
	if isinstance(is_active_tree, dict):
		saved_active = saved_active or {}
		return {
			key: _restore_active_params(saved_active.get(key), template_tree[key], mask)
			for key, mask in is_active_tree.items()
		}

	template = jnp.asarray(template_tree)
	if bool(np.asarray(is_active_tree)):
		value = jnp.asarray(saved_active, dtype=template.dtype)
		return value.reshape(template.shape)
	return jnp.empty(0, dtype=template.dtype)


def run_diagnostics(
	config_toml_path: Path,
	h5_path: Path,
	setting_name: str,
	overwrite: bool = False,
):
	model_code = config_toml_path.stem.split("_")[-1]
	sim_code = h5_path.stem.split("_")[-1]
	agent_count = h5_path.stem.split("_")[-2]
	result_dir = (
		RESULTS_ROOT
		/ setting_name
		/ f"model_{model_code}"
		/ agent_count
		/ f"results_{model_code}_{agent_count}_{sim_code}"
	)
	json_filepath = result_dir / "summary.json"
	h5_filepath = result_dir / "results.h5"
	eval_opt_filepath = result_dir / "eval_trend_at_opt.json"
	eval_truth_filepath = result_dir / "eval_trend_at_truth.json"
	diagnostic_paths = (
		h5_filepath,
		eval_opt_filepath,
		eval_truth_filepath,
	)

	if not json_filepath.exists():
		print(f"Estimation summary not found for Model: {model_code}, Simulation: {sim_code}. Skipping diagnostics.\n")
		return

	existing_result = all(path.exists() for path in diagnostic_paths)
	if existing_result and overwrite:
		print(f"Diagnostics already exist for Model: {model_code}, Simulation: {sim_code}. Overwriting existing results.")
	elif existing_result:
		print(f"Diagnostics already exist for Model: {model_code}, Simulation: {sim_code}. Skipping. \n")
		return

	result_dir.mkdir(parents=True, exist_ok=True)

	with open(config_toml_path, "r") as f:
		config_toml = toml.load(f)

	with h5py.File(h5_path, "r") as f:
		positions = np.asarray(f["positions"][:], dtype=EstimationConfig.NP_FLOAT_DTYPE)
		velocities = np.asarray(f["velocities"][:], dtype=EstimationConfig.NP_FLOAT_DTYPE)
		orientations = np.asarray(f["orientations"][:], dtype=EstimationConfig.NP_FLOAT_DTYPE)
		delaunay_neighbors = f["delaunay_neighbors"][:]
		if "angular_velocities" in f:
			angular_velocities = np.asarray(f["angular_velocities"][:], dtype=EstimationConfig.NP_FLOAT_DTYPE)
		else:
			angular_velocities = np.zeros_like(orientations, dtype=EstimationConfig.NP_FLOAT_DTYPE)

	builder.auto_load_modules()
	base_init_params = EstimationConfig.cast_float_tree(builder.extract_params_from_toml(config_toml))

	states = {
		"positions": positions[:-1],
		"velocities": velocities[:-1],
		"orientations": orientations[:-1],
		"angular_velocities": angular_velocities[:-1],
	}

	states_next = {
		"positions": positions[1:],
		"velocities": velocities[1:],
		"orientations": orientations[1:],
		"angular_velocities": angular_velocities[1:],
	}

	T = states["positions"].shape[0]

	loss_fn_stepwise = builder.build_loss_fn_stepwise(config_toml)

	aux_data = {
		"dt": EstimationConfig.as_float_array(config_toml["Simulation"]["dt"]),
		"v_0": EstimationConfig.as_float_array(config_toml["Simulation"]["v_0"]),
		"safe_eps": EstimationConfig.as_float_array(config_toml["Simulation"]["safe_eps"]),
		"delaunay_neighbors": delaunay_neighbors[:-1],
	}
	aux_axes = {
		"dt": None,
		"v_0": None,
		"safe_eps": None,
		"delaunay_neighbors": 0,
	}

	def total_loss_fn(params, states, states_next, aux_data):
		return jnp.sum(jax.vmap(loss_fn_stepwise, in_axes=(None, 0, 0, aux_axes))(params, states, states_next, aux_data))

	loss_fn_fixed = functools.partial(
		total_loss_fn,
		states=states,
		states_next=states_next,
		aux_data=aux_data,
	)

	print("Detecting structurally inactive parameters via JAX computational graph...")

	key = jax.random.PRNGKey(42)

	def add_dummy_noise(x):
		return x + jax.random.normal(key, x.shape) * 1e-3

	noisy_init_params = jax.tree_util.tree_map(add_dummy_noise, base_init_params)
	grad_tree = jax.jit(jax.grad(loss_fn_fixed))(noisy_init_params)

	is_active_tree = jax.tree_util.tree_map(lambda g: jnp.any(jnp.abs(g) > 1e-12), grad_tree)

	flat_active_flags, _ = ravel_pytree(is_active_tree)
	temp_mapping = []
	for path, leaf in jax.tree_util.tree_leaves_with_path(base_init_params):
		path_str = ".".join([str(p.key) for p in path])
		temp_mapping.extend([path_str] * leaf.size if leaf.size > 1 else [path_str])

	ignored_params = [name for name, active in zip(temp_mapping, flat_active_flags) if not active]
	if ignored_params:
		print(f"-> Automatically IGNORED {len(ignored_params)} inactive parameters: {set(ignored_params)}")
	else:
		print("-> All parameters are ACTIVE.")

	def filter_active(p, is_active):
		return p if is_active else jnp.empty(0)

	fixed_params = base_init_params
	dt_value = float(EstimationConfig.as_float_array(config_toml["Simulation"]["dt"]))

	with open(json_filepath, "r") as f:
		json_data = json.load(f)

	best_active_params = _restore_active_params(
		json_data.get("best_params", {}),
		base_init_params,
		is_active_tree,
	)
	best_params = jax.tree_util.tree_map(
		lambda fixed, active, mask: active if mask else fixed,
		fixed_params,
		best_active_params,
		is_active_tree,
	)
	active_best_params_tree, _ = _split_params_by_activity(best_params, is_active_tree)
	if active_best_params_tree is None:
		active_best_params_tree = {}

	analysis_start_frame = int(EigenvalueTrendConfig.BURN_IN)
	analysis_start_frame = min(max(0, analysis_start_frame), max(0, T - 1))
	analysis_states = {key: value[analysis_start_frame:] for key, value in states.items()}
	analysis_states_next = {key: value[analysis_start_frame:] for key, value in states_next.items()}
	analysis_aux_data = _build_analysis_aux_data(aux_data, analysis_start_frame)
	interval_seconds = float(EigenvalueTrendConfig.HESSIAN_COMPUTATION_INTERVAL)

	print("Calculating Hessian trend (dimensionless) at the active optimal parameters...")
	flat_best_active_output, _ = ravel_pytree(active_best_params_tree)
	hessian_trend_opt = _compute_min_eigenvalue_trend(
		loss_fn_stepwise=loss_fn_stepwise,
		fixed_params=fixed_params,
		is_active_tree=is_active_tree,
		active_params=best_active_params,
		states=analysis_states,
		states_next=analysis_states_next,
		aux_data=analysis_aux_data,
		aux_axes=aux_axes,
		dt=dt_value,
		interval_seconds=interval_seconds,
		analysis_start_frame=analysis_start_frame,
	)

	truth_active_params = jax.tree_util.tree_map(filter_active, base_init_params, is_active_tree)
	print("Calculating Hessian trend (dimensionless) at truth parameters...")
	hessian_trend_truth = _compute_min_eigenvalue_trend(
		loss_fn_stepwise=loss_fn_stepwise,
		fixed_params=base_init_params,
		is_active_tree=is_active_tree,
		active_params=truth_active_params,
		states=analysis_states,
		states_next=analysis_states_next,
		aux_data=analysis_aux_data,
		aux_axes=aux_axes,
		dt=dt_value,
		interval_seconds=interval_seconds,
		analysis_start_frame=analysis_start_frame,
	)

	flat_best_active_params, _ = ravel_pytree(best_active_params)
	active_dim = int(flat_best_active_params.size)
	hessian_unit_free = jnp.asarray(hessian_trend_opt["final_hessian"], dtype=EstimationConfig.JNP_FLOAT_DTYPE)
	eigenvalues = jnp.asarray(hessian_trend_opt["final_eigenvalues"], dtype=EstimationConfig.JNP_FLOAT_DTYPE)
	eigenvectors = jnp.asarray(hessian_trend_opt["final_eigenvectors"], dtype=EstimationConfig.JNP_FLOAT_DTYPE)
	min_eigenvalues = [float(x) for x in hessian_trend_opt["min_eigenvalues"]]
	time_seconds = [float(x) for x in hessian_trend_opt["time_seconds"]]
	time_indices = [int(x) for x in hessian_trend_opt["time_indices"]]
	eval_trend_at_opt = _build_eval_trend_payload(hessian_trend_opt)
	eval_trend_at_truth = _build_eval_trend_payload(hessian_trend_truth)

	if active_dim > 0:
		print("Calculating eigenvalues and eigenvectors in active subspace...")
	else:
		print("No active parameters detected; storing empty Hessian/eigendecomposition.")

	hessian_param_mapping = _build_param_mapping(active_best_params_tree)
	summary = json_data.setdefault("summary", {})
	summary["max_eigenvalue"] = float(eigenvalues[0]) if active_dim > 0 else float("nan")
	summary["min_eigenvalue"] = float(eigenvalues[-1]) if active_dim > 0 else float("nan")
	json_data["hessian_parameter_mapping"] = hessian_param_mapping

	with open(json_filepath, "w") as f:
		json.dump(json_data, f, indent=4)

	with open(eval_opt_filepath, "w") as f:
		json.dump(eval_trend_at_opt, f, indent=4)

	with open(eval_truth_filepath, "w") as f:
		json.dump(eval_trend_at_truth, f, indent=4)

	with h5py.File(h5_filepath, "w") as f:
		f.create_dataset("best_params", data=np.asarray(flat_best_active_output))
		f.create_dataset("hessian", data=np.asarray(hessian_unit_free))
		f.create_dataset("eigenvalues", data=np.asarray(eigenvalues))
		f.create_dataset("eigenvectors", data=np.asarray(eigenvectors))
		f.create_dataset("trend_time_seconds", data=np.asarray(time_seconds, dtype=float))
		f.create_dataset("trend_time_indices", data=np.asarray(time_indices, dtype=int))
		f.create_dataset("trend_min_eigenvalues", data=np.asarray(min_eigenvalues, dtype=float))
		f.attrs["loss_start_norm"] = float(summary.get("loss_start_norm", float("nan")))
		f.attrs["loss_opt_norm"] = float(summary.get("loss_opt_norm", float("nan")))
		f.attrs["grad_norm_start"] = float(summary.get("grad_norm_start", float("nan")))
		f.attrs["grad_norm_opt"] = float(summary.get("grad_norm_opt", float("nan")))

	print(f"Results saved to {result_dir}\n")


def resolve_setting_roots(base: Path, setting: str | None) -> list[Path]:
	if setting is not None:
		return [base / setting]
	return [base / setting_name for setting_name in SocialSimConfig.PARAMETER_SETTINGS]


def run_all_estimations(
	sim_root: Path,
	overwrite: bool = False,
	setting: str | None = None,
):
	base = Path(sim_root)

	for setting_root in resolve_setting_roots(base, setting):
		if not setting_root.is_dir():
			print(f"Warning: {setting_root} does not exist. Skipping.")
			continue

		setting_name = setting_root.name
		print(f"Processing setting: {setting_name}")

		for model_dir in sorted(setting_root.glob("model_*")):
			model_code = model_dir.stem.split("_")[-1]
			config_toml_path = model_dir / f"config_{model_code}.toml"
			if not config_toml_path.is_file():
				print(f"Warning: config not found at {config_toml_path}. Skipping.")
				continue

			with open(config_toml_path, "r") as f:
				config_toml = toml.load(f)
				num_agents_list = config_toml["Simulation"]["num_agents"]
				total_simulation_per_model = config_toml["Simulation"]["total_simulation_per_model"]

			for num_agents in num_agents_list:
				for sim_num in range(1, total_simulation_per_model + 1):
					h5_path = (
						model_dir
						/ "ground_truth"
						/ f"N{num_agents}"
						/ f"simulation_{model_code}_N{num_agents}_{sim_num:04d}.h5"
					)
					if not h5_path.is_file():
						print(f"Warning: simulation file not found at {h5_path}. Skipping.")
						continue

					print(
						f"Running diagnostics for Setting: {setting_name}, "
						f"Model: {model_code}, Agents: {num_agents}, Simulation: {sim_num}"
					)
					run_diagnostics(
						config_toml_path=config_toml_path,
						h5_path=h5_path,
						setting_name=setting_name,
						overwrite=overwrite,
					)


def main():
	parser = argparse.ArgumentParser(description="Compute the Hessian and minimum eigenvalue at the saved optimum.")
	parser.add_argument(
		"--sim_root",
		type=str,
		default=str(DEFAULT_SIMDATA_ROOT),
		help="Base path containing setting_* directories with simulation data.",
	)
	parser.add_argument(
		"--setting",
		type=str,
		default=None,
		help="Run only this setting (e.g. setting_1). Default: all settings in PARAMETER_SETTINGS.",
	)
	parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing results.")
	args = parser.parse_args()

	run_all_estimations(
		sim_root=Path(args.sim_root),
		overwrite=bool(args.overwrite),
		setting=args.setting,
	)


if __name__ == "__main__":
    main()

