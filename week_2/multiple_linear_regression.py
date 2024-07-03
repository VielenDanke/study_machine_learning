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

np.set_printoptions(precision=2)


def predict(x, w, b):
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters
      b (scalar):             model parameter

    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b
    return np.int64(p)


def compute_cost(X, y, w, b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = predict(X[i], w, b)  # (n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i]) ** 2  # scalar
    cost = cost / (2 * m)  # scalar
    return cost


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

    dj_dw = np.zeros(n, dtype=np.int64)
    dj_db = 0

    for i in range(m):
        err = predict(X[i], w, b) - y[i]
        for j in range(n):
            dj_dw[j] = np.add(dj_dw[j], np.multiply(err, X[i, j]))   # access matrix element X[i][j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m  # we are dividing each value in dj_dw by m, see numpy broadcasting
    dj_db = dj_db / m

    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
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
        dj_db, dj_dw = gradient_function(X, y, w, b)  ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw  ##None
        b = b - alpha * dj_db  ##None

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history  # return final w,b and J history for graphing


# input data
w_init = np.array([np.float64(0.39133535), np.float64(18.75376741), np.float64(-53.36032453), np.float64(-26.42131618)])
X_train = np.array([[np.int64(2104), np.int64(5), np.int64(1), np.int64(45)], [np.int64(1416), np.int64(3), np.int64(2), np.int64(40)], [np.int64(852), np.int64(2), np.int64(1), np.int64(35)]])
y_train = np.array([np.int64(460), np.int64(232), np.int64(178)])
iterations = 100000
alpha = 0.00000082
initial_w = np.zeros_like(w_init)
initial_b = 0

# some gradient descent settings
# run gradient descent
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                            compute_cost, compute_gradient,
                                            alpha, iterations)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

m, _ = X_train.shape

for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

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
