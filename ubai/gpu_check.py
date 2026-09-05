import jax, jax.numpy as jnp
print("jax", jax.__version__)
print("devices:", jax.devices())
x = jnp.ones((2048, 2048))
print("matmul sum:", float((x @ x).sum()))
import mujoco_playground, brax
print("mujoco_playground OK / brax", brax.__version__)
