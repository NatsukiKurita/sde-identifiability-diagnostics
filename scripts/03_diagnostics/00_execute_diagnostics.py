#!/usr/bin/env python3
"""Compute Hessian and minimum-eigenvalue time series at the optimum and at truth.

For each estimated trajectory under ``results/<setting>/``,
this script:

1. loads the optimal parameters written by ``02_estimation``,
2. computes the cumulative log-parameter Hessian at the optimum and at truth parameters,
   i.e. the exact Hessian w.r.t. log(p): diag(p) H diag(p) + diag(p * grad),
3. stores the sampled Hessian, gradient, and minimum-eigenvalue time series.

The log-parameter Hessian may be indefinite away from a stationary point.

Outputs (per model / agent count / simulation):
- updates ``summary.json`` with the minimum and maximum eigenvalues at the optimum
- ``result_time_series.h5`` — log-parameter Hessian, cumulative gradient (w.r.t. p),
  and minimum-eigenvalue time series under ``results/optimum`` and ``results/truth``
"""

import argparse
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
from src.experiment_configs.eigenvalue_time_series import EigenvalueTimeSeriesConfig
from src.experiment_configs.simulation import SocialSimConfig
from src.paths import DATA_ROOT, RESULTS_ROOT

EstimationConfig.configure_jax_precision()

DEFAULT_SIMDATA_ROOT = DATA_ROOT / "simulation"


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


def _build_analysis_aux_data(aux_data: dict, analysis_start_frame: int) -> dict:
	analysis_aux_data = {}
	for key, value in aux_data.items():
		if key == "delaunay_neighbors":
			analysis_aux_data[key] = value[analysis_start_frame:]
		else:
			analysis_aux_data[key] = value
	return analysis_aux_data


def _compute_hessian_time_series(
	loss_fn_stepwise,
	params,
	states,
	states_next,
	aux_data,
	aux_axes,
	time_indices: np.ndarray,
):
	flat_params, unravel_fn = ravel_pytree(params)

	def flat_step_loss_fn(flat_params, states_t, state_next, aux_data_t):
		return loss_fn_stepwise(unravel_fn(flat_params), states_t, state_next, aux_data_t)

	hessian_fn = jax.jit(jax.hessian(flat_step_loss_fn))
	grad_fn = jax.jit(jax.grad(flat_step_loss_fn))
	step_hessians = jax.vmap(hessian_fn, in_axes=(None, 0, 0, aux_axes))(
		flat_params,
		states,
		states_next,
		aux_data,
	)
	step_gradients = jax.vmap(grad_fn, in_axes=(None, 0, 0, aux_axes))(
		flat_params,
		states,
		states_next,
		aux_data,
	)

	# Exact Hessian w.r.t. log-parameters: d^2L/dlog(p_i)dlog(p_j) = p_i p_j H_ij + delta_ij p_i g_i.
	# The gradient term does not vanish away from a stationary point (e.g. at truth).
	weights = flat_params[jnp.newaxis, :] * flat_params[:, jnp.newaxis]
	cumulative_gradients = jnp.cumsum(step_gradients, axis=0)
	cumulative_hessians = jnp.cumsum(step_hessians, axis=0) * weights
	cumulative_hessians = cumulative_hessians + jax.vmap(jnp.diag)(cumulative_gradients * flat_params)

	sampled_hessians = np.asarray(cumulative_hessians[time_indices])
	sampled_gradients = np.asarray(cumulative_gradients[time_indices])
	min_eigenvalues = np.asarray(
		[float(np.min(np.linalg.eigvalsh(hessian))) for hessian in sampled_hessians],
		dtype=float,
	)
	return sampled_hessians, sampled_gradients, min_eigenvalues


def _restore_params(saved_params, template_tree):
	"""Rebuild the parameter tree from the serialized optimum."""
	if isinstance(template_tree, dict):
		saved_params = saved_params or {}
		return {
			key: _restore_params(saved_params.get(key), template)
			for key, template in template_tree.items()
		}

	template = jnp.asarray(template_tree)
	return jnp.asarray(saved_params, dtype=template.dtype).reshape(template.shape)


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
	h5_filepath = result_dir / "result_time_series.h5"

	if not json_filepath.exists():
		print(f"Estimation summary not found for Model: {model_code}, Simulation: {sim_code}. Skipping diagnostics.\n")
		return

	if h5_filepath.exists() and overwrite:
		print(f"Diagnostics already exist for Model: {model_code}, Simulation: {sim_code}. Overwriting existing results.")
	elif h5_filepath.exists():
		print(f"Diagnostics already exist for Model: {model_code}, Simulation: {sim_code}. Skipping. \n")
		return

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
	truth_params = EstimationConfig.cast_float_tree(builder.extract_params_from_toml(config_toml))

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

	dt_value = float(EstimationConfig.as_float_array(config_toml["Simulation"]["dt"]))

	with open(json_filepath, "r") as f:
		json_data = json.load(f)

	best_params = _restore_params(json_data.get("best_params", {}), truth_params)

	analysis_start_frame = int(EigenvalueTimeSeriesConfig.BURN_IN)
	analysis_start_frame = min(max(0, analysis_start_frame), max(0, T - 1))
	analysis_states = {key: value[analysis_start_frame:] for key, value in states.items()}
	analysis_states_next = {key: value[analysis_start_frame:] for key, value in states_next.items()}
	analysis_aux_data = _build_analysis_aux_data(aux_data, analysis_start_frame)
	analysis_num_steps = int(analysis_states["positions"].shape[0])
	interval_seconds = float(EigenvalueTimeSeriesConfig.HESSIAN_COMPUTATION_INTERVAL)
	interval_steps = max(1, int(round(interval_seconds / dt_value)))
	time_indices, time_seconds = _build_analysis_time_points(analysis_num_steps, dt_value, interval_seconds)

	series = {}
	for label, params in (("optimum", best_params), ("truth", truth_params)):
		print(f"Calculating log-parameter Hessian time series at {label} parameters...")
		series[label] = _compute_hessian_time_series(
			loss_fn_stepwise=loss_fn_stepwise,
			params=params,
			states=analysis_states,
			states_next=analysis_states_next,
			aux_data=analysis_aux_data,
			aux_axes=aux_axes,
			time_indices=time_indices,
		)

	final_eigenvalues = np.linalg.eigvalsh(series["optimum"][0][-1])
	summary = json_data.setdefault("summary", {})
	summary["max_eigenvalue"] = float(final_eigenvalues[-1])
	summary["min_eigenvalue"] = float(final_eigenvalues[0])

	with open(json_filepath, "w") as f:
		json.dump(json_data, f, indent=4)

	with h5py.File(h5_filepath, "w") as f:
		f.attrs["analysis_start_frame"] = int(analysis_start_frame)
		f.attrs["analysis_num_steps"] = int(analysis_num_steps)
		f.attrs["interval_seconds"] = float(interval_seconds)
		f.attrs["interval_steps"] = int(interval_steps)
		f.create_dataset("time_seconds", data=np.asarray(time_seconds, dtype=float))
		f.create_dataset("time_indices", data=np.asarray(time_indices, dtype=int))
		for label, (hessians, gradients, min_eigenvalues) in series.items():
			group = f.create_group(f"results/{label}")
			group.create_dataset("hessians", data=hessians)
			group.create_dataset("gradients", data=gradients)
			group.create_dataset("min_eigenvalues", data=min_eigenvalues)

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
	parser = argparse.ArgumentParser(description="Compute log-parameter Hessian and minimum-eigenvalue time series at the optimum and truth.")
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
