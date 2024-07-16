import copy
import math

import numpy as np

from cost_function_logistic_regression import compute_cost_logistic_regression_regularization
from sigmoid import sigmoid
from regression_and_classification.week_2.z_score_normalization import zscore_normalize_features


def compute_gradient_logistic_reg(X, y, w, b, lambda_):
    """
    Computes the gradient for linear regression

    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))  # (n,)
    dj_db = 0.0  # scalar

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)  # (n,)(n,)=scalar
        err_i = f_wb_i - y[i]  # scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]  # scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw / m  # (n,)
    dj_db = dj_db / m  # scalar

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_ / m) * w[j]

    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, alpha, lambda_, num_iters):
    """
    Performs batch gradient descent

    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter
    """
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    m = X.shape[0]

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic_reg(X, y, w, b, lambda_)

        # Update Parameters using w, b, alpha and gradient with regularization
        w = w * (1 - alpha * (lambda_ / m)) - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(compute_cost_logistic_regression_regularization(X, y, w, b, lambda_))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b, J_history  # return final w,b and J history for graphing


# input data
X_train = np.array([[1231412, 123], [88888, 10], [1237677, 85], [9999999, 48], [5384213, 99], [123776674, 153]])
y_train = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.zeros_like(X_train[0])
b_tmp = 0.
alph = 5
lambda_ = 0.00001
iters = 10000

# normalize big numbers (feature scaling)
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)

print(f"X Normalized: {X_norm}")

w_out, b_out, _ = gradient_descent(X_norm, y_train, w_tmp, b_tmp, alph, lambda_, iters)
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

# predict new parameters
X_train = np.array([[1231412, 123]])

X_train_norm = (X_train - X_mu) / X_sigma

print(sigmoid(np.dot(X_train_norm, w_out) + b_out))
