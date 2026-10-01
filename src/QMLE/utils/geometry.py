import jax.numpy as jnp


def cross_2d(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """2D cross product a_x b_y - a_y b_x. The last axis is the vector component."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
