# new_w = w - a * (d / d * w) * J(w, b)
# a - learning rate (between 0 and 1), controls how big is the step we're taking
# (d / d * w) * J(w, b) - derivative term, which direction we're taking
# For derivative w - 1 / m SUM (f(x[i]) - y[i]) * x[i], where f(x[i]) = w * x[i] + b
#
# new_b = b - a * (d / d * b) * J(w, b)
# a - learning rate (between 0 and 1), controls how big is the step we're taking
# (d / d * b) * J(w, b) - derivative term, which direction we're taking
# For derivative b - 1 / m SUM (f(x[i]) - y[i]), where f(x[i]) = w * x[i] + b
#
# Simultaneously update w and b

import numpy as np


# Function to calculate the cost
def compute_cost(x_train, y_train, w, b):
    m = x_train.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x_train[i] + b
        cost = cost + (f_wb - y_train[i]) ** 2
    total_cost = 1 / (2 * m) * cost

    return total_cost


def compute_gradient(x_train, y_train, w, b):
    """
    Computes the gradient for linear regression
    Args:
      x_train (ndarray (m,)): Data, m examples
      y_train (ndarray (m,)): target values
      w,b (scalar)    : model parameters
    Returns
      gradient_descent_w (scalar): The gradient of the cost w.r.t. the parameters w
      gradient_descent_b (scalar): The gradient of the cost w.r.t. the parameter b
     """

    # Number of training examples
    m = x_train.shape[0]
    gradient_descent_w = 0
    gradient_descent_b = 0

    for i in range(m):
        f_wb = w * x_train[i] + b
        dj_dw_i = (f_wb - y_train[i]) * x_train[i]
        dj_db_i = f_wb - y_train[i]
        gradient_descent_w += dj_dw_i
        gradient_descent_b += dj_db_i
    gradient_descent_w = gradient_descent_w / m
    gradient_descent_b = gradient_descent_b / m

    return gradient_descent_w, gradient_descent_b


def calculate_gradient_descent(alpha, x_train, y_train, initial_w, initial_b, iterations):
    current_w = initial_w
    current_b = initial_b

    J_history = []
    p_history = []

    for i in range(iterations):
        # write history changes in costs (not necessary)
        J_history.append(compute_cost(x_train, y_train, current_w, current_b))
        p_history.append([current_w, current_b])

        # calculate gradient descent
        dj_dw, dj_db = compute_gradient(x_train, y_train, current_w, current_b)

        # calculate temp_w, temp_b using formulas at the top of the file
        temp_w, temp_b = current_w - alpha * dj_dw, current_b - alpha * dj_db

        # if the values stays the same return current_w current_b (we found the minimum)
        if temp_w == current_w and temp_b == current_b:
            return current_w, current_b, J_history, p_history

        # update simultaneously
        current_w = temp_w
        current_b = temp_b

    return current_w, current_b, J_history, p_history


w_final, b_final, J_hist, p_hist = calculate_gradient_descent(
    # learning rate
    0.01,
    # input x
    np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.4]),
    # output y labeled to x by index i
    np.array([300.0, 500.0, 700, 900, 1000, 1500, 2000]),
    0,
    0,
    100000
)

print(w_final, b_final)
print(f"Price of 1.2 house ${w_final * 1.250 + b_final}")
