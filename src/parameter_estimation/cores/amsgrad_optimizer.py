import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Any, Callable, Tuple

from src.constants.estimation import EstimationConfig


def _normalize_initial_params(initial_params: Any) -> Any:
    """Convert inputs to JAX arrays and ensure a leading batch dimension exists."""
    params_as_arrays = jax.tree.map(lambda x: jnp.asarray(x), initial_params)
    leaves = jax.tree.leaves(params_as_arrays)

    if not leaves:
        raise ValueError("initial_params is empty")

    first_leaf = leaves[0]
    if first_leaf.ndim == 0:
        return jax.tree.map(lambda x: jnp.expand_dims(x, 0), params_as_arrays)

    return params_as_arrays


def amsgrad_optimizer(
    loss_fn_fixed: Callable,
    initial_params: Any,
    lower_bounds: Any,
    upper_bounds: Any,
    maxiter: int = EstimationConfig.MAXITER_AMSGRAD,
    learning_rate: float = EstimationConfig.LR,
) -> Tuple[Any, float, int]:
    """Run batched AMSGrad optimization and return the best result.

    Args:
        loss_fn_fixed: Objective function to minimize.
        initial_params: Initial parameter pytree. Leading dimension is interpreted as batch.
        lower_bounds: Lower bounds pytree.
        upper_bounds: Upper bounds pytree.
        maxiter: Maximum number of AMSGrad iterations.
        learning_rate: AMSGrad learning rate.

    Returns:
        best_params: Parameters with the lowest objective value.
        best_loss: Minimum objective value among all starts.
        best_iter: Iteration count corresponding to the best result.
    """
    EstimationConfig.configure_jax_precision()

    initial_params_processed = _normalize_initial_params(initial_params)
    batch_size = jax.tree.leaves(initial_params_processed)[0].shape[0]

    lower_bounds = jax.tree.map(lambda x: jnp.asarray(x), lower_bounds)
    upper_bounds = jax.tree.map(lambda x: jnp.asarray(x), upper_bounds)

    optimizer = optax.amsgrad(learning_rate=learning_rate)

    def _tree_all_finite(tree: Any) -> bool:
        leaves = jax.tree.leaves(tree)
        return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)

    @jax.jit
    def step(params, opt_state):
        loss, grad = jax.value_and_grad(loss_fn_fixed)(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_params = jax.tree.map(
            lambda x, lo, hi: jnp.clip(x, lo, hi),
            new_params,
            lower_bounds,
            upper_bounds,
        )
        return new_params, opt_state, loss

    print(f"Running AMSGrad optimization with {batch_size} initial points...")

    best_loss = np.inf
    best_params = None
    best_iter = 0

    for i in range(batch_size):
        params = jax.tree.map(lambda x: x[i], initial_params_processed)
        opt_state = optimizer.init(params)

        local_best_loss = np.inf
        local_best_params = params
        local_best_iter = 0

        initial_loss = float(loss_fn_fixed(params))
        if np.isfinite(initial_loss):
            local_best_loss = initial_loss

        for epoch in range(maxiter):
            params_before = params
            params, opt_state, loss_value = step(params, opt_state)
            loss_float = float(loss_value)

            if np.isfinite(loss_float) and loss_float < local_best_loss:
                local_best_loss = loss_float
                # Keep the parameter state at which the above loss was evaluated.
                local_best_params = params_before
                local_best_iter = epoch + 1

            if not _tree_all_finite(params):
                break

        if local_best_loss < best_loss:
            best_loss = local_best_loss
            best_params = local_best_params
            best_iter = local_best_iter

    if best_params is None:
        raise RuntimeError("AMSGRad optimization failed across all initial starts.")

    return best_params, float(best_loss), int(best_iter)
