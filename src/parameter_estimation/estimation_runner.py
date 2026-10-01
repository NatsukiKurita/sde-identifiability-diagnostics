import functools
import json
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import toml
from jax.flatten_util import ravel_pytree

import src.parameter_estimation.builder as builder
from src.constants.estimation import EstimationConfig
from src.constants.penetration_detection import PenetrationDetectionConfig
from src.parameter_estimation.cores.trust_constr_optimizer import trust_constr_optimizer
from src.parameter_estimation.utils.boundary_aux import WALL_AUX_AXES, build_wall_aux_entries

EstimationConfig.configure_jax_precision()

OUTSIDE_CHECK_TOL = PenetrationDetectionConfig.OUTSIDE_CHECK_TOL
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


def _normalize_corners(corners_raw) -> np.ndarray:
	corners = np.asarray(corners_raw, dtype=float)
	if corners.ndim == 1:
		if corners.size % 2 != 0:
			raise ValueError("Boundary corners must contain an even number of values.")
		corners = corners.reshape(-1, 2)
	elif corners.ndim == 2 and corners.shape[1] == 2:
		pass
	else:
		corners = corners.reshape(-1, 2)

	if corners.shape[0] < 3:
		raise ValueError("Boundary polygon must have at least 3 vertices.")

	center = np.mean(corners, axis=0)
	diff = corners - center
	angles = np.arctan2(diff[:, 1], diff[:, 0])
	order = np.argsort(angles)
	return corners[order]


def _points_inside_polygon(x: np.ndarray, y: np.ndarray, polygon: np.ndarray, tol: float = 1e-9) -> np.ndarray:
	px = np.asarray(x, dtype=float).reshape(-1)
	py = np.asarray(y, dtype=float).reshape(-1)
	poly = np.asarray(polygon, dtype=float)

	valid = np.isfinite(px) & np.isfinite(py)
	inside = np.zeros(px.shape[0], dtype=bool)
	if poly.shape[0] < 3:
		return inside.reshape(x.shape)

	xv = px[valid]
	yv = py[valid]
	inside_valid = np.zeros(xv.shape[0], dtype=bool)
	on_boundary = np.zeros(xv.shape[0], dtype=bool)

	x1 = poly[:, 0]
	y1 = poly[:, 1]
	x2 = np.roll(x1, -1)
	y2 = np.roll(y1, -1)

	for xa, ya, xb, yb in zip(x1, y1, x2, y2):
		intersects = ((ya > yv) != (yb > yv))
		x_intersect = (xb - xa) * (yv - ya) / ((yb - ya) + np.finfo(float).eps) + xa
		inside_valid ^= intersects & (xv < x_intersect)

		cross = (xv - xa) * (yb - ya) - (yv - ya) * (xb - xa)
		dot = (xv - xa) * (xv - xb) + (yv - ya) * (yv - yb)
		on_boundary |= (np.abs(cross) <= tol) & (dot <= tol)

	inside[valid] = inside_valid | on_boundary
	return inside.reshape(x.shape)


def _detect_outside_positions(positions: np.ndarray, corners: np.ndarray, tol: float = OUTSIDE_CHECK_TOL):
	if positions.ndim != 3 or positions.shape[-1] < 2:
		raise ValueError("positions must have shape (T, N, D) with D >= 2.")

	x = positions[..., 0]
	y = positions[..., 1]
	inside_mask = _points_inside_polygon(x, y, corners, tol=tol)
	outside_mask = (~inside_mask) & np.isfinite(x) & np.isfinite(y)
	outside_count = int(np.count_nonzero(outside_mask))

	first_outside_t = None
	first_outside_agent = None
	if outside_count > 0:
		first_idx = np.argwhere(outside_mask)[0]
		first_outside_t = int(first_idx[0])
		first_outside_agent = int(first_idx[1])

	return outside_mask, outside_count, first_outside_t, first_outside_agent


def _write_skip_message(skip_path: Path, payload: dict) -> None:
	skip_path.parent.mkdir(parents=True, exist_ok=True)
	with open(skip_path, "w") as f:
		json.dump(payload, f, indent=4)


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

	boundary_type = str(config_toml.get("Boundary", {}).get("boundary_shape", ""))
	if boundary_type != "NoWall":
		corners_np = _normalize_corners(config_toml["Boundary"]["corners"])
	else:
		corners_np = np.zeros((0, 2), dtype=float)
	
	if boundary_type != "NoWall":
		_, outside_count, first_outside_t, first_outside_agent = _detect_outside_positions(positions, corners_np)
		if outside_count > 0:
			if json_filepath.exists():
				json_filepath.unlink()
			if h5_filepath.exists():
				h5_filepath.unlink()

			skip_payload = {
				"status": "skipped",
				"reason": "trajectory_outside_boundary",
				"model_code": model_code,
				"agent_count": agent_count,
				"simulation_code": sim_code,
				"source_h5": str(h5_path),
				"outside_point_count": int(outside_count),
				"first_outside_time_index": int(first_outside_t) if first_outside_t is not None else None,
				"first_outside_agent_index": int(first_outside_agent) if first_outside_agent is not None else None,
			}
			_write_skip_message(skip_filepath, skip_payload)
			print(
				"Skipping sample due to out-of-bound trajectory "
				f"(outside points={outside_count}, first=(t={first_outside_t}, agent={first_outside_agent})). "
				f"Message saved to {skip_filepath}."
			)
			return
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
		"delaunay_neighbors": jnp.zeros((T, 1, 1), dtype=delaunay_neighbors.dtype),
		**build_wall_aux_entries(config_toml, corners_np, boundary_type),
	}

	if config_toml["Neighbor"]["type"] == "Neighbor_Delaunay":
		aux_data["delaunay_neighbors"] = delaunay_neighbors[:-1]

	aux_axes = dict(WALL_AUX_AXES)

	# if config_toml["Neighbor"] == "Neighbor_Delaunay":
		# from src.src.parameter_estimation.estimator_modules.Neighbor.jl_delaunay_restoration.delaunay_restoration_frondend import restore_delaunay_neighbors

		# aux_data["delaunay_neighbors"] = restore_delaunay_neighbors(states["positions"], aux_data["corners"])

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
