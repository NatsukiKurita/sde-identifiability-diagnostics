#!/usr/bin/env python3
"""Run multistart parameter estimation.

For each simulation trajectory under ``data/simulation/<setting>/``,
this script optimizes parameters (multistart + trust-constr) and writes the loss
and parameter estimates. Hessian diagnostics are computed separately.

Outputs (per model / agent count / simulation):
- ``summary.json`` — optimization summary and parameter estimates
- ``parameter_estimates.json`` — cumulative parameter estimates sampled at fixed time intervals
"""

import argparse
import functools
import importlib
import json
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
from src.experiment_configs.simulation import SocialSimConfig
from src.paths import DATA_ROOT, RESULTS_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))
estimate_parameter_trend = importlib.import_module("01_error_evolution").estimate_parameter_trend

EstimationConfig.configure_jax_precision()

DEFAULT_SIMDATA_ROOT = DATA_ROOT / "simulation"


def _compute_loss_and_grad_norm(loss_fn, params):
	loss_value, grad_tree = jax.value_and_grad(loss_fn)(params)
	grad_flat, _ = ravel_pytree(grad_tree)
	grad_norm = float(jnp.linalg.norm(grad_flat))
	return float(loss_value), grad_norm


def _evaluate_multistart_reference(loss_fn, batched_params):
	leaves = jax.tree_util.tree_leaves(batched_params)
	if not leaves:
		raise ValueError("batched_params is empty")

	first_leaf = leaves[0]

	if first_leaf.ndim == 0:
		loss_value, grad_norm = _compute_loss_and_grad_norm(loss_fn, batched_params)
		return {
			"best_start_index": 0,
			"loss_start": float(loss_value),
			"grad_norm_start": float(grad_norm),
			"best_start_params": batched_params,
		}

	batch_size = int(first_leaf.shape[0])
	best_idx = None
	best_loss = np.inf
	best_grad_norm = np.inf
	best_params = None

	for idx in range(batch_size):
		single_params = jax.tree_util.tree_map(lambda x: x[idx], batched_params)
		loss_value, grad_norm = _compute_loss_and_grad_norm(loss_fn, single_params)
		if loss_value < best_loss:
			best_loss = loss_value
			best_grad_norm = grad_norm
			best_idx = idx
			best_params = single_params

	if best_params is None:
		raise RuntimeError("Failed to evaluate multistart reference metrics.")

	return {
		"best_start_index": int(best_idx),
		"loss_start": float(best_loss),
		"grad_norm_start": float(best_grad_norm),
		"best_start_params": best_params,
	}


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


def _build_optimizer_initial_params(config_toml: dict, base_init_params: dict):
	return EstimationConfig.sample_multistart_initial_params(
		config_toml=config_toml,
		base_init_params=base_init_params,
		seed=EstimationConfig.MULTISTART_LHS_SEED,
	)


def run_estimation(
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
	parameter_estimates_filepath = result_dir / "parameter_estimates.json"

	existing_result = json_filepath.exists() and parameter_estimates_filepath.exists()
	if existing_result and overwrite:
		print(f"Results already exist for Model: {model_code}, Simulation: {sim_code}. Overwriting existing results.")
	elif existing_result:
		print(f"Results already exist for Model: {model_code}, Simulation: {sim_code}. Skipping. \n")
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
	optimizer_initial_params = EstimationConfig.cast_float_tree(
		_build_optimizer_initial_params(config_toml, base_init_params)
	)

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

	_, N, _ = states["positions"].shape

	loss_fn_stepwise = builder.build_loss_fn_stepwise(config_toml)
	lower_bounds, upper_bounds = EstimationConfig.extract_bounds(config_toml)

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

	active_opt_init_params = jax.tree_util.tree_map(filter_active, optimizer_initial_params, is_active_tree)
	active_lower = jax.tree_util.tree_map(filter_active, lower_bounds, is_active_tree)
	active_upper = jax.tree_util.tree_map(filter_active, upper_bounds, is_active_tree)

	fixed_params = base_init_params

	dt_value = float(EstimationConfig.as_float_array(config_toml["Simulation"]["dt"]))
	parameter_estimates, final_estimate = estimate_parameter_trend(
		loss_fn_stepwise=loss_fn_stepwise,
		fixed_params=fixed_params,
		is_active_tree=is_active_tree,
		active_initial_params=active_opt_init_params,
		active_lower=active_lower,
		active_upper=active_upper,
		states=states,
		states_next=states_next,
		aux_data=aux_data,
		aux_axes=aux_axes,
		dt=dt_value,
		interval_seconds=float(EstimationConfig.PARAMETER_ESTIMATION_INTERVAL),
	)
	best_active_params = final_estimate["best_active_params"]
	best_loss = final_estimate["best_loss"]
	presearch_iters = final_estimate["presearch_iterations"]
	pre_meta = final_estimate["presearch_meta"]
	iters = final_estimate["iterations"]
	meta = final_estimate["meta"]
	presearch_message = str(pre_meta.get("message", ""))
	print(
		"Final presearch finished "
		f"(iters={presearch_iters}, "
		f"message={presearch_message!r})."
	)
	message = str(meta.get("message", ""))

	best_params = jax.tree_util.tree_map(
		lambda fixed, active, mask: active if mask else fixed,
		fixed_params,
		best_active_params,
		is_active_tree,
	)

	print("Best Loss:", best_loss)
	print("Iterations to Converge:", iters)

	active_best_params_tree, nonactive_best_params_tree = _split_params_by_activity(best_params, is_active_tree)
	if active_best_params_tree is None:
		active_best_params_tree = {}
	if nonactive_best_params_tree is None:
		nonactive_best_params_tree = {}

	burn_in_frames = int(parameter_estimates["burn_in_frames"])
	analysis_states = {key: value[burn_in_frames:] for key, value in states.items()}
	analysis_states_next = {key: value[burn_in_frames:] for key, value in states_next.items()}
	analysis_aux_data = {
		key: value[burn_in_frames:] if aux_axes.get(key) == 0 else value
		for key, value in aux_data.items()
	}
	analysis_loss_fn = functools.partial(
		total_loss_fn,
		states=analysis_states,
		states_next=analysis_states_next,
		aux_data=analysis_aux_data,
	)
	analysis_num_steps = int(analysis_states["positions"].shape[0])

	init_reference = _evaluate_multistart_reference(analysis_loss_fn, optimizer_initial_params)
	loss_start = init_reference["loss_start"]
	grad_norm_start = init_reference["grad_norm_start"]
	loss_opt, grad_norm_opt = _compute_loss_and_grad_norm(analysis_loss_fn, best_params)
	normalizer = float(N * analysis_num_steps)
	loss_start_norm = float(loss_start / normalizer)
	loss_opt_norm = float(loss_opt / normalizer)

	param_mapping = _build_param_mapping(active_best_params_tree)
	nonactive_param_mapping = _build_param_mapping(nonactive_best_params_tree)

	best_params_json = _serialize_param_tree(active_best_params_tree)
	nonactive_params_json = _serialize_param_tree(nonactive_best_params_tree)
	parameter_estimates["parameter_mapping"] = param_mapping

	json_data = {
		"summary": {
			"best_loss": float(best_loss),
			"presearch_iterations": int(presearch_iters),
			"iterations": int(iters),
			"message": message,
			"loss_start_norm": float(loss_start_norm),
			"loss_opt_norm": float(loss_opt_norm),
			"grad_norm_start": float(grad_norm_start),
			"grad_norm_opt": float(grad_norm_opt),
		},
		"parameter_mapping": param_mapping,
		"nonactive_parameter_mapping": nonactive_param_mapping,
		"best_params": best_params_json,
		"nonactive_params": nonactive_params_json,
	}

	with open(json_filepath, "w") as f:
		json.dump(json_data, f, indent=4)

	with open(parameter_estimates_filepath, "w") as f:
		json.dump(parameter_estimates, f, indent=4)

	print(f"Estimation saved to {result_dir}\n")


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
						f"Running estimation for Setting: {setting_name}, "
						f"Model: {model_code}, Agents: {num_agents}, Simulation: {sim_num}"
					)
					run_estimation(
						config_toml_path=config_toml_path,
						h5_path=h5_path,
						setting_name=setting_name,
						overwrite=overwrite,
					)


def main():
	parser = argparse.ArgumentParser(description="Run multistart parameter estimation.")
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

