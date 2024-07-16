import math

import matplotlib.pyplot as plt
import numpy as np

# plt.style.use('./deeplearning.mplstyle')

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.4])
y_train = np.array([300.0, 500.0, 700, 900, 1000, 1500, 2000])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

i = 0  # Change this to 1 to see (x^1, y^1)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")


# # Plot the data points
# plt.scatter(x_train, y_train, marker='x', c='r')
# # Set the title
# plt.title("Housing Prices")
# # Set the y-axis label
# plt.ylabel('Price (in 1000s of dollars)')
# # Set the x-axis label
# plt.xlabel('Size (1000 sqft)')
# plt.show()


# Formula: (1 / (2 * m)) * SUM i = [0..m-1]((w * x[i] + b) - y[i])^2
def calculate_cost_function(x_train, y_train, w, b):
    total_sum_error = 0
    m = x_train.shape[0]

    for i in range(m):
        f_wb = w * x_train[i] + b
        cost = (f_wb - y_train[i]) ** 2
        total_sum_error += cost

    return (1 / (2 * m)) * total_sum_error


# Calculate best w and best b for f(x) = w * x[i] + b
def calculate_best_w_b():
    min_cost = None
    best_w = 0
    best_b = 0

    for w in range(-300, 300):
        for b in range(-300, 300):
            total_sum_error = calculate_cost_function(x_train, y_train, w, b)

            if min_cost is None or min_cost > total_sum_error:
                best_w = w
                best_b = b
                min_cost = total_sum_error

    print(f"Minimum cost {min_cost} with w = {best_w} and b = {best_b}")

    return best_w, best_b


# compute prediction with best_w and best_b
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples
      w (scalar): model parameter
      b (scalar): model parameters
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb


best_w, best_b = calculate_best_w_b()

tmp_f_wb = compute_model_output(x_train, best_w, best_b)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

# Predictions
x_i = 1.2
cost_1200sqft = best_w * x_i + best_b

print(f"${cost_1200sqft:.0f} thousand dollars")
