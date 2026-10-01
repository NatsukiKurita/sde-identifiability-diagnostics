import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

def estimate_vel(vel_data: jnp.ndarray):
    vel_norm = jnp.linalg.norm(vel_data, axis=-1)
    mean_speed = jnp.mean(vel_norm)
    return mean_speed