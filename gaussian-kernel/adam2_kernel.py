import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
import optax
import numpy as np

def get_data(n_data=100):
    key = jax.random.PRNGKey(42)
    key_, k1, k2, k3 = jax.random.split(key, 4)
    
    x1 = jax.random.uniform(k1, shape=(n_data, 1), minval=0.0, maxval=1.0)
    x2 = jax.random.uniform(k2, shape=(n_data, 1), minval=-1.0, maxval=1.0)
    error = jax.random.normal(k3, shape=(n_data, 1)) * 0.05
    
    y = (0.8/jnp.sqrt(2 * jnp.pi)) * jnp.exp(-(x1**2 + x2**2)) + error
    x_data = jnp.concatenate([x1, x2], axis=1)
    return x_data, y

n_data = 200
x_data, y_data = get_data(n_data)

@jit
def gaussian_kernel(u):
    return (1/jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * u**2)

@jit
def H_matrix_fast(h, x):
    n, dim = x.shape
    diffs = (x.T[:, None, :] - x.T[:, :, None]) 
    
    u = diffs / h[:, None, None]
    
    K_mat = jnp.prod(gaussian_kernel(u), axis=0) / (n * jnp.prod(h))
    
    fk = jnp.sum(K_mat, axis=1, keepdims=True)
    return K_mat / fk

@jit
def gcv_loss(log_h, x, y):
    h = jnp.exp(log_h)  
    n = y.shape[0]
    H = H_matrix_fast(h, x)
    
    y_hat = H @ y
    res = y - y_hat
    
    num = jnp.mean(res**2)
    den = (1.0 - jnp.trace(H) / n)**2
    return num / den


key_ = jax.random.PRNGKey(35)
params = jax.random.uniform(key_, shape=(2,))


# exponential_decay = optax.exponential_decay(
#     init_value=1e-4, 
#     transition_steps=100,
#     transition_begin=150,
#     decay_rate=0.01,
#     end_value=1e-8
# )

optimizer = optax.adam(learning_rate=0.001) # change learning_rate if needed
opt_state = optimizer.init(params)

@jit
def step(params, opt_state, x, y):
    loss, grads = value_and_grad(gcv_loss)(params, x, y)
    scaler = 1/jnp.linalg.norm(grads)
    gr = scaler * grads
    updates, opt_state = optimizer.update(gr, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

gcv_history = [gcv_loss(params, x_data, y_data)]
iter_max = 20000
for i in range(iter_max):
    params, opt_state, loss = step(params, opt_state, x_data, y_data)
    loss_ = gcv_loss(params, x_data, y_data)
    gcv_history.append(loss_)
    if i % 500 == 0:
        print(f"Iter {i}: GCV {loss:.6f}, h {jnp.exp(params)}")

h_best = jnp.exp(params)
H_final = H_matrix_fast(h_best, x_data)
y_pred = H_final @ y_data
mse = jnp.mean((y_data - y_pred)**2)

print(f"\nFinal GCV: {gcv_history[-1]:.8f}")
print(f"\nFinal MSE: {mse}")
print(f"Optimal h: {h_best}")

np.savetxt("adam_bands.csv", np.asarray(h_best), delimiter=",")
np.savetxt("GCV_adam.csv", gcv_history, delimiter=",", comments='')

import matplotlib.pyplot as plt

plt.plot(gcv_history)
plt.title("GCV adam")
plt.xlabel("Iteration")
plt.ylabel("GCV")
plt.show()

plt.plot(y_data, label= "Data")
plt.plot(y_pred,'--', label= "Prediction")
plt.legend()

plt.show()
