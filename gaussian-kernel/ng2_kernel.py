import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import numpy as np

def generate_data(n_data=100):
    key = jax.random.PRNGKey(42)
    key_, k1, k2, k3 = jax.random.split(key, 4)
    
    x1 = jax.random.uniform(k1, (n_data, 1), minval=0.0, maxval=1.0)
    x2 = jax.random.uniform(k2, (n_data, 1), minval=-1.0, maxval=1.0)
    x_data = jnp.concatenate([x1, x2], axis=1)
    
    error = jax.random.normal(k3, shape=(n_data, 1)) * 0.05
    
    y = (0.8/jnp.sqrt(2 * jnp.pi)) * jnp.exp(-(x1**2 + x2**2)) + error
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
def gcv_loss(log_h, x, y):
    h = jnp.exp(log_h)
    n = y.shape[0]
    H = get_H_matrix(h, x)
    
    y_hat = H @ y
    mse = jnp.mean((y - y_hat)**2)
    
    trace_h = jnp.trace(H)
    denom = (1.0 - trace_h / n)**2
    return mse / denom

gcv_grad_fn = jit(grad(gcv_loss))


@jit
def compute_natural_grad(log_h, x, y):
    gr = gcv_grad_fn(log_h, x, y)
    
    H_approx = jnp.outer(gr, gr)
    
    solution = jnp.linalg.lstsq(H_approx, gr, rcond=None)
    gr_nat = solution[0]
    
    return gr_nat, gr

batch_gcv = jit(vmap(lambda eta, h, g, x, y: gcv_loss(h - eta * g, x, y), in_axes=(0, None, None, None, None)))

midle = n_data**(-1/5)

key_ = jax.random.PRNGKey(35)
log_h = jax.random.uniform(key_, shape=(2,))

steps = 0.1**jnp.linspace(0, 10, 11)

print("Starting Optimized Natural Gradient Descent...")
gcv_hist = [gcv_loss(log_h, x_data, y_data)]

iter_max = 20000
for i in range(iter_max):
    gr_nat, raw_grad = compute_natural_grad(log_h, x_data, y_data)
    scaler = 1.0 / (jnp.linalg.norm(gr_nat) + 1e-8)
    direction = gr_nat * scaler
    
    # line search
    losses = batch_gcv(steps, log_h, direction, x_data, y_data)
    best_step = steps[jnp.argmin(losses)]
    
    log_h = log_h - best_step * direction
    current_gcv = jnp.min(losses)
    gcv_hist.append(current_gcv)
    
    if i % 500 == 0:
        print(f"Iter {i}: GCV={current_gcv:.6f}, h={jnp.exp(log_h)}, step : {best_step}")

h_best = jnp.exp(log_h)
H_final = get_H_matrix(h_best, x_data)
y_pred = H_final @ y_data
mse = jnp.mean((y_data - y_pred)**2)

print(f"\nFinal GCV: {gcv_hist[-1]:.8f}")
print(f"\nFinal MSE: {mse:.10f}")
print(f"Optimal Bandwidth h: {h_best}")

# np.savetxt("band_ng.csv", h_best, delimiter=",", comments='')
np.savetxt("ng_bands.csv", np.asarray(h_best), delimiter=",")
np.savetxt("GCV_ng.csv", gcv_hist, delimiter=",", comments='')

import matplotlib.pyplot as plt

plt.plot(gcv_hist)
plt.ylabel("GCV")
plt.xlabel("Iteration")
plt.show()

plt.plot(y_data, label= "Data")
plt.plot(y_pred,'--', label= "Prediction")
plt.legend()

plt.show()

