import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy import optimize
from typing import Tuple, Any, Callable

from src.experiment_configs.estimation import EstimationConfig


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


def trust_constr_optimizer(
    loss_fn_fixed: Callable,
    initial_params: Any,  # dict of (NUM_STARTS, ...)
    lower_bounds: dict,
    upper_bounds: dict,
    gtol: float = EstimationConfig.GTOL_TRUST_CONSTR,
    xtol: float = EstimationConfig.XTOL_TRUST_CONSTR,
    maxiter: int = EstimationConfig.MAXITER_TRUST_CONSTR,
    return_metadata: bool = False,
) -> Tuple[Any, float, Any]:
    """Run batched Trust-Region Newton-CG (trust-constr) optimization and return the best result.

    Args:
        loss_fn_fixed: Objective function to minimize.
        initial_params: Initial parameter pytree. Leading dimension is interpreted as batch.
        lower_bounds: Lower bounds pytree.
        upper_bounds: Upper bounds pytree.
        gtol: Gradient norm tolerance (SciPy trust-constr ``gtol``).
        xtol: Step tolerance (SciPy trust-constr ``xtol``).
        maxiter: Maximum number of optimization iterations.

    Returns:
        best_params: Parameters with the lowest objective value.
        best_loss: Minimum objective value among all starts.
        best_iter: Iteration count corresponding to the best result.
        best_meta (optional): Dict containing optimizer diagnostics for the best start.
    """
    initial_params_processed = _normalize_initial_params(initial_params)
    leaves = jax.tree.leaves(initial_params_processed)

    batch_size_candidates = [int(leaf.shape[0]) for leaf in leaves]
    batch_size = batch_size_candidates[0]
    if any(size != batch_size for size in batch_size_candidates):
        raise ValueError("Inconsistent leading batch dimension in initial_params.")

    # Take the first element of the batch to obtain the PyTree structure and flatten/unflatten functions
    single_init = jax.tree.map(lambda x: x[0], initial_params_processed)
    flat_single_init, unflatten_fn = ravel_pytree(single_init)
    opt_dtype = flat_single_init.dtype

    # Flatten the bounds PyTree in the same way and build a SciPy Bounds object
    flat_lower, _ = ravel_pytree(lower_bounds)
    flat_upper, _ = ravel_pytree(upper_bounds)
    flat_lower_np = np.array(flat_lower, dtype=np.float64)
    flat_upper_np = np.array(flat_upper, dtype=np.float64)
    scipy_bounds = optimize.Bounds(flat_lower_np, flat_upper_np)

    # ---------------------------------------------------------
    # Core numerical functions, JIT-compiled with JAX
    # ---------------------------------------------------------
    @jax.jit
    def val_and_grad_flat(x_flat: jnp.ndarray):
        """Compute the objective and its gradient together (for SciPy's jac=True)."""
        params = unflatten_fn(x_flat)
        loss, grads = jax.value_and_grad(loss_fn_fixed)(params)
        grads_flat, _ = ravel_pytree(grads)
        return loss, grads_flat

    @jax.jit
    def hvp_flat(x_flat: jnp.ndarray, p_flat: jnp.ndarray):
        """Compute the Hessian-vector product (HVP)."""
        params = unflatten_fn(x_flat)
        p_tree = jax.tree.map(
            lambda tangent, param: jnp.asarray(tangent, dtype=param.dtype),
            unflatten_fn(p_flat),
            params,
        )
        # Exact HVP via forward-over-reverse automatic differentiation
        _, hvp_tree = jax.jvp(jax.grad(loss_fn_fixed), (params,), (p_tree,))
        hvp_flat_arr, _ = ravel_pytree(hvp_tree)
        return hvp_flat_arr

    # ---------------------------------------------------------
    # Wrapper functions passed to SciPy (bridge between JAX and NumPy arrays)
    # ---------------------------------------------------------
    def scipy_fun(x_np: np.ndarray):
        loss, grad = val_and_grad_flat(jnp.asarray(x_np, dtype=opt_dtype))
        # SciPy expects a Python float and NumPy float64 arrays
        return float(loss), np.array(grad, dtype=np.float64)

    def scipy_hessp(x_np: np.ndarray, p_np: np.ndarray):
        # SciPy may probe with int8 during LinearOperator dtype initialization.
        hvp_res = hvp_flat(jnp.asarray(x_np, dtype=opt_dtype), jnp.asarray(p_np, dtype=opt_dtype))
        return np.array(hvp_res, dtype=np.float64)

    # ---------------------------------------------------------
    # Run the optimization (loop over the batch)
    # ---------------------------------------------------------
    print(f"Running trust-constr optimization with {batch_size} initial points...")
    
    best_loss = np.inf
    best_params_flat = None
    best_iter = 0
    best_meta = {
        "message": "",
        "success": False,
        "status": -1,
    }

    for i in range(batch_size):
        # Extract and flatten the i-th initial parameters
        x0_tree = jax.tree.map(lambda x: x[i], initial_params_processed)
        x0_flat, _ = ravel_pytree(x0_tree)
        x0_flat = np.clip(np.array(x0_flat, dtype=np.float64), flat_lower_np, flat_upper_np)

        res = optimize.minimize(
            fun=scipy_fun,
            x0=x0_flat,
            method='trust-constr',
            jac=True,            # scipy_fun also returns the gradient
            hessp=scipy_hessp,   # HVP function, key to escaping saddle points
            bounds=scipy_bounds,
            options={
                'gtol': gtol,
                'xtol': xtol,
                'maxiter': maxiter,
                'verbose': 0     # Set to 1 or 2 for debugging output
            }
        )

        if res.fun < best_loss:
            best_loss = res.fun
            best_params_flat = np.clip(np.array(res.x, dtype=np.float64), flat_lower_np, flat_upper_np)
            best_iter = res.nit
            best_meta = {
                "message": str(res.message),
                "success": bool(res.success),
                "status": int(res.status),
            }

    # Restore the best flattened parameters to the original PyTree (dict) structure
    if best_params_flat is None:
        raise RuntimeError("Optimization failed across all initial starts.")
        
    best_params = unflatten_fn(jnp.array(best_params_flat))

    if return_metadata:
        return best_params, float(best_loss), int(best_iter), best_meta

    return best_params, float(best_loss), int(best_iter)