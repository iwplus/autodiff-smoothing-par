import jax
import jax.numpy as jnp
from jax import jit, grad
import numpy as np

def generate_data(n_data=100):
    key = jax.random.PRNGKey(42)
    key_, k1, k2, k3 = jax.random.split(key, 4)
    
    x1 = jax.random.uniform(k1, shape=(n_data, 1), minval=0.0, maxval=1.0)
    x2 = jax.random.uniform(k2, shape=(n_data, 1), minval=-1.0, maxval=1.0)
    error = jax.random.normal(k3, shape=(n_data, 1)) * 0.05
    
    y = (0.8/jnp.sqrt(2 * jnp.pi)) * jnp.exp(-(x1**2 + x2**2)) + error
    x_data = jnp.concatenate([x1, x2], axis=1)
    return x_data, y

n_data = 200
x_data, y_data = generate_data(n_data)

@jit
def kernel_vec(u):
    return (1/jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * (u**2))

@jit
def H_matrix_fast(h, x_data):
    n, m = x_data.shape

    # diffs[l, i, j] = x[i, l] - x[j, l]
    diffs = x_data.T[:, :, None] - x_data.T[:, None, :]
    
    # Calculate u = (x_j - x_i) / h
    u = diffs / h[:, None, None]
    
    K = jnp.prod(kernel_vec(u), axis=0)
    
    h_prod = jnp.prod(h)
    K = K / (n * h_prod)
    
    fk = jnp.sum(K, axis=1, keepdims=True)
    H = K / fk
    return H

@jit
def gcv_loss(h, x_data, y_data):
    h = jnp.exp(h)
    n = y_data.shape[0]
    H_mat = H_matrix_fast(h, x_data)
    
    y_hat = H_mat @ y_data
    error = y_data - y_hat
    
    numerator = jnp.sum(error**2)
    denominator = ((1 - jnp.trace(H_mat) / n))**2
    return (numerator / n) / denominator

gcv_grad_fn = jit(grad(gcv_loss))

key_ = jax.random.PRNGKey(35)
h = jax.random.uniform(key_, shape=(2,))


lr = 1e-3
iter_max = 20000

grid = jnp.linspace(0, 30, 31)
steps = 0.5**grid

print("Starting Optimization...")
gcv_gd = [gcv_loss(h, x_data, y_data)]
for i in range(iter_max):
    g_val = gcv_loss(h, x_data, y_data)
    grads = gcv_grad_fn(h, x_data, y_data)
    scaler = 1/jnp.linalg.norm(grads)
    
    h = h - lr * scaler * grads
    loss_ = gcv_loss(h, x_data, y_data)
    gcv_gd.append(loss_)
    if i % 500 == 0:
        # print(f"Iter {i}: GCV = {g_val:.6f}, h = {jnp.exp(h)}, step size : {step_size}")
        print(f"Iter {i}: GCV = {g_val:.6f}, h = {jnp.exp(h)}")

h_best = jnp.exp(h)
H_final = H_matrix_fast(h_best, x_data)
y_pred = H_final @ y_data
mse = jnp.mean((y_data - y_pred)**2)

print(f"\nFinal GCV: {gcv_gd[-1]:.8f}")
print(f"\nFinal MSE: {mse:.8f}")
print(f"Final h: {h_best}")

# np.savetxt("band_gd.csv", h_best, delimiter=",", comments='')
np.savetxt("gd_bands.csv", np.asarray(h_best), delimiter=",")
np.savetxt("GCV_gd.csv", gcv_gd, delimiter=",", comments='')

import matplotlib.pyplot as plt

plt.plot(gcv_gd)
plt.title("GCV gradient descent")
plt.xlabel("Iteration")
plt.ylabel("GCV")
plt.show()


plt.plot(y_data, label= "Data")
plt.plot(y_pred,'--', label= "Prediction")
plt.legend()
plt.show()