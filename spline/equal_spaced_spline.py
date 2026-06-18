import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np

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

def truncated(knot, deg, x):
    return jnp.maximum(0, x - knot) ** deg

def B_matrix(knots, deg, x):
    """
    x: array of shape (n_samples, n_features)
    knots: array of shape (n_features, n_knots)
    deg: integer degree
    """
    n_samples, n_features = x.shape
    n_knots = knots.shape[1]
    
    intercept = jnp.ones((n_samples, 1))
    
    poly_list = []
    for d in range(1, deg + 1):
        poly_list.append(x ** d)
    poly_terms = jnp.concatenate(poly_list, axis=1)
    
    spline_terms = truncated(knots[None, :, :], deg, x[:, :, None])
    spline_terms = spline_terms.reshape(n_samples, -1)
    
    B = jnp.concatenate([intercept, poly_terms, spline_terms], axis=1)
        
    return B

def gcv(knots, deg, x, y_data):
    B_mat = B_matrix(knots, deg, x)
    n = y_data.shape[0]

    beta = jnp.linalg.lstsq(B_mat, y_data, rcond=None)[0]
    
    y_hat = B_mat @ beta
    residuals = y_data - y_hat
    
    numerator = jnp.sum(residuals**2)
    H_mat = B_mat @ jnp.linalg.pinv(B_mat.T @ B_mat) @ B_mat.T
    denominator = n * (1 - jnp.trace(H_mat) / n)**2
    
    return numerator / denominator


knots_1 = jnp.reshape(jnp.linspace(-1.0, 1.0, n_knots), (1,n_knots))
knots_2 = jnp.reshape(jnp.linspace(-1.0, 1.0, n_knots), (1,n_knots))

knots = jnp.concatenate([knots_1, knots_2])


B_mat = B_matrix(knots, deg, x_data)
n = y_data.shape[0]
beta = jnp.linalg.lstsq(B_mat, y_data, rcond=None)[0]
y_hat = B_mat @ beta
mse = jnp.mean((y_data - y_hat)**2)
gcv = gcv(knots, deg, x_data, y_data)
print(f"Final MSE: {mse}")
print(f"Final GCV: {gcv}")

np.savetxt("es_knots.csv", np.asarray(knots), delimiter=",")

import matplotlib.pyplot as plt
plt.plot(y_data, label= "Data")
plt.plot(y_hat,'--', label= "Prediction")
plt.legend()

plt.show()
