import functools
import json
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import toml
from jax.flatten_util import ravel_pytree

import src.QMLE.builder as builder
from src.experiment_configs.estimation import EstimationConfig
from src.QMLE.utils.trust_constr_optimizer import trust_constr_optimizer

EstimationConfig.configure_jax_precision()

RUN_MODE_SINGLE = "single"
RUN_MODE_MULTISTART = "multistart"


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


def _build_optimizer_initial_params(config_toml: dict, base_init_params: dict, run_mode: str):
	if run_mode == RUN_MODE_SINGLE:
		return base_init_params
	if run_mode == RUN_MODE_MULTISTART:
		return EstimationConfig.sample_multistart_initial_params(
			config_toml=config_toml,
			base_init_params=base_init_params,
			seed=EstimationConfig.MULTISTART_LHS_SEED,
		)
	raise ValueError(f"Unknown run_mode: {run_mode}. Supported: {RUN_MODE_SINGLE}, {RUN_MODE_MULTISTART}")


def run_estimation(
	config_toml_path: Path,
	h5_path: Path,
	overwrite: bool = False,
	run_mode: str = RUN_MODE_SINGLE,
	results_subdir: str = "02_eigenvalue_analysis",
):
	model_code = config_toml_path.stem.split("_")[-1]
	sim_code = h5_path.stem.split("_")[-1]
	agent_count = h5_path.stem.split("_")[-2]
	result_dir = Path(f"./results/{results_subdir}/model_{model_code}/{agent_count}/results_{model_code}_{agent_count}_{sim_code}")
	json_filepath = result_dir / "summary.json"
	h5_filepath = result_dir / "results.h5"
	skip_filepath = result_dir / "skip_message.json"

	existing_result = json_filepath.exists() and h5_filepath.exists()
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

	if skip_filepath.exists():
		skip_filepath.unlink()

	builder.auto_load_modules()
	base_init_params = EstimationConfig.cast_float_tree(builder.extract_params_from_toml(config_toml))
	optimizer_initial_params = EstimationConfig.cast_float_tree(
		_build_optimizer_initial_params(config_toml, base_init_params, run_mode=run_mode)
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

	T, N, _ = states["positions"].shape

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

	def loss_fn_active_only(active_params):
		full_params = jax.tree_util.tree_map(
			lambda fixed, active, mask: active if mask else fixed,
			fixed_params,
			active_params,
			is_active_tree,
		)
		return loss_fn_fixed(full_params)

	# Multistart mode: run a loose presearch first, then strict refinement.
	presearch_iters = 0
	if run_mode == RUN_MODE_MULTISTART:
		print("Running loose presearch optimization for multistart...")
		pre_params, _, presearch_iters, pre_meta = trust_constr_optimizer(
			loss_fn_active_only,
			active_opt_init_params,
			active_lower,
			active_upper,
			tol=EstimationConfig.TOL_PRESEARCH_TRUST_CONSTR,
			xtol=EstimationConfig.XTOL_PRESEARCH_TRUST_CONSTR,
			maxiter=EstimationConfig.MAXITER_PRESEARCH_TRUST_CONSTR,
			return_metadata=True,
		)
		active_opt_init_params = pre_params
		presearch_message = str(pre_meta.get("message", ""))
		print(
			"Presearch finished "
			f"(iters={presearch_iters}, "
			f"message={presearch_message!r})."
		)

	best_active_params, best_loss, iters, meta = trust_constr_optimizer(
		loss_fn_active_only,
		active_opt_init_params,
		active_lower,
		active_upper,
		return_metadata=True,
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

	init_reference = _evaluate_multistart_reference(loss_fn_fixed, optimizer_initial_params)
	loss_start = init_reference["loss_start"]
	grad_norm_start = init_reference["grad_norm_start"]
	loss_opt, grad_norm_opt = _compute_loss_and_grad_norm(loss_fn_fixed, best_params)
	normalizer = float(N * T)
	loss_start_norm = float(loss_start / normalizer)
	loss_opt_norm = float(loss_opt / normalizer)

	print("Calculating Hessian (dimensionless) at the active optimal parameters...")
	flat_best_active_output, _ = ravel_pytree(active_best_params_tree)
	flat_best_active_params, unravel_active_fn = ravel_pytree(best_active_params)

	active_dim = int(flat_best_active_params.size)
	if active_dim > 0:
		def flat_loss_fn_active(p_vec):
			return loss_fn_active_only(unravel_active_fn(p_vec))

		hessian_at_optimum_active = jax.jit(jax.hessian(flat_loss_fn_active))(flat_best_active_params)
		weights_active = flat_best_active_params[jnp.newaxis, :] * flat_best_active_params[:, jnp.newaxis]
		hessian_unit_free = hessian_at_optimum_active * weights_active

		print("Calculating eigenvalues and eigenvectors in active subspace...")
		eigenvalues, eigenvectors = jnp.linalg.eigh(hessian_unit_free)
		idx = jnp.argsort(eigenvalues)[::-1]
		eigenvalues = eigenvalues[idx]
		eigenvectors = eigenvectors[:, idx]
	else:
		print("No active parameters detected; storing empty Hessian/eigendecomposition.")
		hessian_unit_free = jnp.zeros((0, 0), dtype=EstimationConfig.JNP_FLOAT_DTYPE)
		eigenvalues = jnp.zeros((0,), dtype=EstimationConfig.JNP_FLOAT_DTYPE)
		eigenvectors = jnp.zeros((0, 0), dtype=EstimationConfig.JNP_FLOAT_DTYPE)

	param_mapping = _build_param_mapping(active_best_params_tree)
	nonactive_param_mapping = _build_param_mapping(nonactive_best_params_tree)
	hessian_param_mapping = _build_param_mapping(active_best_params_tree)

	best_params_json = _serialize_param_tree(active_best_params_tree)
	nonactive_params_json = _serialize_param_tree(nonactive_best_params_tree)

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
			"max_eigenvalue": float(eigenvalues[0]) if active_dim > 0 else float("nan"),
			"min_eigenvalue": float(eigenvalues[-1]) if active_dim > 0 else float("nan"),
		},
		"parameter_mapping": param_mapping,
		"nonactive_parameter_mapping": nonactive_param_mapping,
		"hessian_parameter_mapping": hessian_param_mapping,
		"best_params": best_params_json,
		"nonactive_params": nonactive_params_json,
	}

	with open(json_filepath, "w") as f:
		json.dump(json_data, f, indent=4)

	with h5py.File(h5_filepath, "w") as f:
		f.create_dataset("best_params", data=np.asarray(flat_best_active_output))
		f.create_dataset("hessian", data=np.asarray(hessian_unit_free))
		f.create_dataset("eigenvalues", data=np.asarray(eigenvalues))
		f.create_dataset("eigenvectors", data=np.asarray(eigenvectors))
		f.attrs["loss_start_norm"] = float(loss_start_norm)
		f.attrs["loss_opt_norm"] = float(loss_opt_norm)
		f.attrs["grad_norm_start"] = float(grad_norm_start)
		f.attrs["grad_norm_opt"] = float(grad_norm_opt)

	print(f"Results successfully saved to {json_filepath} and {h5_filepath}\n")


def run_all_estimations(
	sim_root: Path,
	overwrite: bool = False,
	run_mode: str = RUN_MODE_SINGLE,
	results_subdir: str = "02_eigenvalue_analysis",
):
	sim_root = Path(sim_root)
	model_dirs = sim_root.glob("model_*")

	for model_dir in model_dirs:
		model_code = model_dir.stem.split("_")[-1]
		config_toml_path = model_dir / f"config_{model_code}.toml"

		with open(config_toml_path, "r") as f:
			config_toml = toml.load(f)
			num_agents_list = config_toml["Simulation"]["num_agents"]
			total_simulation_per_model = config_toml["Simulation"]["total_simulation_per_model"]

		for num_agents in num_agents_list:
			for sim_num in range(1, total_simulation_per_model + 1):
				h5_path = model_dir / f"ground_truth/N{num_agents}" / f"simulation_{model_code}_N{num_agents}_{sim_num:04d}.h5"
				print(f"Running estimation for Model: {model_code}, Agents: {num_agents}, Simulation: {sim_num}")
				run_estimation(
					config_toml_path=config_toml_path,
					h5_path=h5_path,
					overwrite=overwrite,
					run_mode=run_mode,
					results_subdir=results_subdir,
				)
