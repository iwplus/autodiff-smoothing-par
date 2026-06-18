import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import optax

def generalized_logistic(x, a, b, k=1.0, x0=0.0):
    """
    Maps x to the range [a, b].
    """
    return a + (b - a) / (1 + jnp.exp(-k * (x - x0)))

def inverse_generalized_logistic(y, a, b, k=1.0, x0=0.0):
    """
    Maps a value y in range (a, b) back to the real line (-inf, inf).
    """
    # We add a tiny epsilon to prevent log(0) or division by zero
    eps = 1e-7
    y = jnp.clip(y, a + eps, b - eps)
    
    return x0 + (1/k) * jnp.log((y - a) / (b - y))


def generate_data(n_data=100):
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    
    x1 = jax.random.uniform(k1, (n_data, 1), minval=-1.0, maxval=1.0)
    x2 = jax.random.uniform(k2, (n_data, 1), minval=-1.0, maxval=1.0)
    x_data = jnp.concatenate([x1, x2], axis=1)
    
    error = jax.random.normal(k3, (n_data, 1)) * 0.1
    y_data = 0.8*jnp.sin(x1/2)+0.3*jnp.cos(x2) + error
    return x_data, y_data

n_data = 200
x_data, y_data = generate_data(n_data)

n_features = 2
n_knots = 2
deg = 3


def truncated(knots, x):
    diff = x - knots
    return jnp.where(diff > 0, diff**deg, 0.0)

def B_matrix(knots, x):
    n_samples, n_features = x.shape
    
    intercept = jnp.ones((n_samples, 1))
    
    poly_list = [x**d for d in range(1, deg + 1)]
    poly_terms = jnp.concatenate(poly_list, axis=1)
    
    spline_terms = truncated(knots[None, :, :], x[:, :, None])
    
    spline_terms = spline_terms.reshape(n_samples, -1)
    
    return jnp.concatenate([intercept, poly_terms, spline_terms], axis=1)

def gcv(log_knots, x, y_data):
    knots = generalized_logistic(log_knots, -1, 1) # here a = -1 and b = 1, replace if needed
    B_mat = B_matrix(knots, x)
    n = y_data.shape[0]
    beta = jnp.linalg.lstsq(B_mat, y_data, rcond=None)[0]
    
    y_hat = B_mat @ beta
    residuals = y_data - y_hat
    
    numerator = jnp.sum(residuals**2)
    M = B_mat.T @ B_mat
    M_inv = jnp.linalg.pinv(M)
    H_mat = B_mat @ M_inv @ B_mat.T
    denominator = n * (1 - jnp.trace(H_mat) / n)**2
    
    return numerator / denominator



key_ = jax.random.PRNGKey(35)
initial_knots = inverse_generalized_logistic(jax.random.uniform(key_, shape=(n_features,n_knots), minval=-1.0, maxval=1.0), -1, 1)
    
# exponential_decay = optax.exponential_decay(
#     init_value=1e-4, 
#     transition_steps=100,
#     transition_begin=150,
#     decay_rate=0.01,
#     end_value=1e-8
# )


optimizer = optax.adam(learning_rate=0.0032)
opt_state = optimizer.init(initial_knots)

@jax.jit
def step(knots, opt_state):
    loss_val, grads = jax.value_and_grad(gcv)(knots, x_data, y_data)
    scaler = 1/jnp.linalg.norm(grads)
    updates, opt_state = optimizer.update(scaler * grads, opt_state)
    knots = optax.apply_updates(knots, updates)
    return knots, opt_state, loss_val

gcv_grad = jax.grad(gcv)
knots = initial_knots
gcv_history = [gcv(knots, x_data, y_data)]
for i in range(20000):
    knots, opt_state, loss = step(knots, opt_state)
    if i % 500 == 0:
        print(f"Iteration {i}, GCV Loss: {loss:.6f}")
    loss_ = gcv(knots, x_data, y_data)
    gcv_history.append(loss_)

knots_g = generalized_logistic(knots, -1, 1) # here a = -1 and b = 1, replace if needed

B_mat = B_matrix(knots_g, x_data)
n = y_data.shape[0]
beta = jnp.linalg.lstsq(B_mat, y_data, rcond=None)[0]
y_hat = B_mat @ beta
mse = jnp.mean((y_data - y_hat)**2)
gcv = gcv(knots, x_data, y_data)

print(f"Final knots : {knots}")
print(f"Final MSE: {mse}")
print(f"Final GCV: {gcv}")

np.savetxt("adam_knots.csv", np.asarray(knots), delimiter=",")
np.savetxt("GCV_adam_spline.csv", gcv_history, delimiter=",", comments='')

import matplotlib.pyplot as plt

plt.plot(jnp.log(jnp.array(gcv_history)))
plt.title("GCV adam")
plt.xlabel("Iteration")
plt.ylabel("GCV")
plt.show()

plt.plot(y_data, label= "Data")
plt.plot(y_hat,'--', label= "Prediction")
plt.legend()

plt.show()
