import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from typing import Tuple, Any, Callable
import jaxopt

from src.constants.estimation import EstimationConfig


def _normalize_initial_params(initial_params: Any) -> Any:
    """Convert inputs to JAX arrays and ensure a leading batch dimension exists."""
    params_as_arrays = jax.tree.map(lambda x: jnp.asarray(x), initial_params)
    leaves = jax.tree.leaves(params_as_arrays)

    if not leaves:
        raise ValueError("initial_params is empty")

    first_leaf = leaves[0]
    if first_leaf.ndim == 0:
        # Single-start scalar inputs are treated as batch size 1.
        return jax.tree.map(lambda x: jnp.expand_dims(x, 0), params_as_arrays)

    return params_as_arrays


def _batched_tree_all_finite(tree: Any, batch_size: int) -> jnp.ndarray:
    """Return a per-start mask indicating whether every parameter is finite."""
    masks = []
    for leaf in jax.tree.leaves(tree):
        leaf_arr = jnp.asarray(leaf)
        if leaf_arr.ndim == 0:
            masks.append(jnp.broadcast_to(jnp.isfinite(leaf_arr), (batch_size,)))
        elif leaf_arr.shape[0] == batch_size:
            masks.append(jnp.all(jnp.isfinite(leaf_arr).reshape((batch_size, -1)), axis=1))

    if not masks:
        return jnp.ones((batch_size,), dtype=bool)

    return jnp.all(jnp.stack(masks), axis=0)


def lbfgsb_optimizer(
    loss_fn_fixed: Callable,
    initial_params: Any,  # dict of (NUM_STARTS, ...)
    lower_bounds: dict,
    upper_bounds: dict,
    tol: float = EstimationConfig.TOL_LBFGSB,
    maxiter: int = EstimationConfig.MAXITER_LBFGSB,
    return_metadata: bool = False,
) -> Tuple[Any, float, Any]:
    """Run batched L-BFGS-B optimization and return the best result.

    Args:
        loss_fn_fixed: Objective function to minimize.
        initial_params: Initial parameter pytree. Leading dimension is interpreted as batch.
        lower_bounds: Lower bounds pytree.
        upper_bounds: Upper bounds pytree.
        tol: Convergence tolerance.
        maxiter: Maximum number of optimization iterations.
        return_metadata: If true, return a diagnostics dictionary for the best start.

    Returns:
        best_params: Parameters with the lowest objective value.
        best_loss: Minimum objective value among all starts.
        iter_num: Iteration count corresponding to the best result.
        best_meta (optional): Optimizer diagnostics for the best start.
    """
    initial_params_processed = _normalize_initial_params(initial_params)
    leaves = jax.tree.leaves(initial_params_processed)

    batch_size_candidates = [
        int(leaf.shape[0])
        for leaf in leaves
        if leaf.ndim > 0 and int(leaf.shape[0]) > 0
    ]
    if batch_size_candidates:
        batch_size = batch_size_candidates[0]
        if any(size != batch_size for size in batch_size_candidates):
            raise ValueError("Inconsistent leading batch dimension in initial_params.")
    else:
        batch_size = 1

    single_init = jax.tree.map(
        lambda x: x[0] if x.ndim > 0 and int(x.shape[0]) > 0 else x,
        initial_params_processed,
    )
    flat_single_init, _ = ravel_pytree(single_init)

    if int(flat_single_init.size) == 0:
        best_loss = float(loss_fn_fixed(single_init))
        best_meta = {
            "message": "No active parameters to optimize.",
            "success": True,
            "status": 0,
            "error": 0.0,
            "failed_linesearch": False,
        }
        if return_metadata:
            return single_init, best_loss, 0, best_meta
        return single_init, best_loss, 0

    bounds = (
        jax.tree.map(lambda x: jnp.asarray(x), lower_bounds),
        jax.tree.map(lambda x: jnp.asarray(x), upper_bounds),
    )

    optimizer = jaxopt.LBFGSB(
        fun=loss_fn_fixed,
        tol=tol,
        maxiter=maxiter,
    )

    def single_optimization_step(params):
        return optimizer.run(init_params=params, bounds=bounds)

    batch_fit = jax.jit(jax.vmap(single_optimization_step))

    print(f"Running optimization with {batch_size} initial points...")

    results = batch_fit(initial_params_processed)
    result_values = results.state.value
    result_params_finite = _batched_tree_all_finite(results.params, batch_size)
    result_valid = (
        jnp.isfinite(result_values)
        & jnp.isfinite(results.state.error)
        & result_params_finite
    )

    if bool(jnp.any(result_valid)):
        values_for_selection = jnp.where(result_valid, result_values, jnp.inf)
        best_idx = jnp.argmin(values_for_selection)
        best_params = jax.tree.map(lambda x: x[best_idx], results.params)
        best_loss = result_values[best_idx]
        iter_num = results.state.iter_num[best_idx]
        best_error = results.state.error[best_idx]
        best_failed_linesearch = results.state.failed_linesearch[best_idx]
        fallback_used = False
    else:
        initial_values = jax.jit(jax.vmap(loss_fn_fixed))(initial_params_processed)
        initial_params_finite = _batched_tree_all_finite(initial_params_processed, batch_size)
        initial_valid = jnp.isfinite(initial_values) & initial_params_finite
        if not bool(jnp.any(initial_valid)):
            raise RuntimeError("L-BFGS-B produced no finite result and all initial losses are non-finite.")

        values_for_selection = jnp.where(initial_valid, initial_values, jnp.inf)
        best_idx = jnp.argmin(values_for_selection)
        best_params = jax.tree.map(lambda x: x[best_idx], initial_params_processed)
        best_loss = initial_values[best_idx]
        iter_num = jnp.array(0, dtype=results.state.iter_num.dtype)
        best_error = jnp.inf
        best_failed_linesearch = jnp.array(True)
        fallback_used = True

    best_meta = {
        "message": "L-BFGS-B produced no finite result; using the best finite initial point."
        if fallback_used
        else (
            "L-BFGS-B converged."
            if bool(jnp.asarray(best_error <= tol)) and not bool(jnp.asarray(best_failed_linesearch))
            else "L-BFGS-B stopped before satisfying tolerance."
        ),
        "success": (
            bool(jnp.asarray(best_error <= tol))
            and not bool(jnp.asarray(best_failed_linesearch))
            and not fallback_used
        ),
        "status": 2 if fallback_used else (0 if bool(jnp.asarray(best_error <= tol)) else 1),
        "error": float(best_error),
        "failed_linesearch": bool(best_failed_linesearch),
        "num_fun_eval": 0 if fallback_used else int(results.state.num_fun_eval[best_idx]),
        "num_grad_eval": 0 if fallback_used else int(results.state.num_grad_eval[best_idx]),
        "fallback_used": fallback_used,
    }

    if return_metadata:
        return best_params, float(best_loss), int(iter_num), best_meta

    return best_params, float(best_loss), int(iter_num)