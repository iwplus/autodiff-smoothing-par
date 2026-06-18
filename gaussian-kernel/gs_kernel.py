import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np

def generate_data(n_data=100):
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    
    x1 = jax.random.uniform(k1, (n_data, 1), minval=0.0, maxval=1.0)
    x2 = jax.random.uniform(k2, (n_data, 1), minval=-1.0, maxval=1.0)
    x_data = jnp.concatenate([x1, x2], axis=1)
    
    error = jax.random.normal(k3, (n_data, 1)) * 0.05
    y = (0.8/jnp.sqrt(2*jnp.pi)) * jnp.exp(-(x1**2 + x2**2)) + error
    return x_data, y

n_data = 200
x_data, y_data = generate_data(n_data)

@jit
def get_H_matrix(h, x):
    n, dim = x.shape
    diffs = x.T[:, :, None] - x.T[:, None, :]
    u = diffs / h[:, None, None]
    
    kernel_vals = (1/jnp.sqrt(2*jnp.pi)) * jnp.exp(-0.5 * u**2)
    
    K = jnp.prod(kernel_vals, axis=0) / (n * jnp.prod(h))
    
    fk = jnp.sum(K, axis=1, keepdims=True)
    return K / fk

@jit
def gcv(h, x, y):
    n = y.shape[0]
    H_mat = get_H_matrix(h, x)
    
    y_hat = H_mat @ y
    mse = jnp.mean((y - y_hat)**2)
    
    trace_H = jnp.trace(H_mat)
    denom = (1.0 - trace_H / n)**2
    return mse / denom

v_gcv = jit(vmap(lambda h: gcv(h, x_data, y_data)))

n_grid_points = 10
midle = n_data**(-1/5)
step_ = 1e-3
ub = midle + step_
grid_points = jnp.linspace(step_, ub, n_grid_points)

A, B = jnp.meshgrid(grid_points, grid_points)
meshgrid_points = jnp.column_stack([A.ravel(), B.ravel()])

print("Running Grid Search...")
gcv_results = v_gcv(meshgrid_points)

min_idx = jnp.argmin(gcv_results)
h_best = meshgrid_points[min_idx]

print(f"Optimal h: {h_best} with GCV: {gcv_results[min_idx]}")

H_best = get_H_matrix(h_best, x_data)
y_pred = H_best @ y_data
mse = jnp.mean((y_data - y_pred)**2)
print(f"Final MSE: {mse}")

# np.savetxt("band_gs.csv", h_best, delimiter=",", comments='')
np.savetxt("gs_bands.csv", np.asarray(h_best), delimiter=",")

import matplotlib.pyplot as plt
plt.plot(y_data, label= "Data")
plt.plot(y_pred,'--', label= "Prediction")
plt.legend()

plt.show()
