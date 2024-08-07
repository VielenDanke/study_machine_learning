# new_w = w - a * (d / d * w) * J(w (vector), b)
# a - learning rate (between 0 and 1), controls how big is the step we're taking
# (d / d * w) * J(w, b) - derivative term, which direction we're taking
# For derivative w - 1 / m SUM (f(x[i]) - y[i]) * x[i][j], where f(x[i]) = w (vector) * x[i] (vector) + b
#
# new_b = b - a * (d / d * b) * J(w (vector), b)
# a - learning rate (between 0 and 1), controls how big is the step we're taking
# (d / d * b) * J(w, b) - derivative term, which direction we're taking
# For derivative b - 1 / m SUM (f(x[i]) - y[i]), where f(x[i]) = w (vector) * x[i] (vector) + b
#
# Simultaneously update w and b

import copy
import math

import matplotlib.pyplot as plt
import numpy as np

from cost_function_linear_regression import compute_cost
from z_score_normalization import zscore_normalize_features

np.set_printoptions(precision=2)


def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape

    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):
        err = np.dot(X[i], w) + b - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]  # access matrix element X[i][j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m  # we are dividing each value in dj_dw by m, see numpy broadcasting
    dj_db = dj_db / m

    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters
      b_in (scalar)       : initial model parameter
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,)) : Updated values of parameters
      b (scalar)       : Updated value of parameter
      """

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient(X, y, w, b)  ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw  ##None
        b = b - alpha * dj_db  ##None

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(compute_cost(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history  # return final w,b and J history for graphing


# input data
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
X_train = np.array(
    [[2104, 5, 1, 45], [1416, 3, 2, 40],
     [852, 2, 1, 35]]
)
y_train = np.array([460, 232, 178])
iterations = 100
alpha = 1.0e-1
initial_w = np.zeros_like(w_init)
initial_b = 0

# normalize the input using z-score normalization
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)

print(f"X non-normalized: {X_train}")
print(f"X normalized using z-score normalization: {X_norm}")

# some gradient descent settings
# run gradient descent
w_norm, b_norm, J_hist = gradient_descent(X_norm, y_train, initial_w, initial_b, alpha, iterations)

# Input
x_house = np.array([1200, 3, 1, 40])
# Normalize the input
x_house_norm = (x_house - X_mu) / X_sigma
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict:0.0f}")

# draw plot
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration")
ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')
ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')
ax2.set_xlabel('iteration step')
plt.show()
