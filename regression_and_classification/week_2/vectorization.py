# Example:
import time

import numpy as np

# Data
w = np.array([1.0, 2.5, -3.3])
b = 4
x = np.array([10, 20, 30])
n = w.shape[0]

# Approach 1 (loop)
f = 0
for j in range(0, n):
    f += w[j] * x[j]
f += b

print(f)

# Approach 2 (Vectorization)
# Could use parallel hardware (GPU or CPU), much more efficient
f = np.dot(w, x) + b

print(f)

# Performance tests
n = 1_000_000
w = []
x = []

for i in range(0, n):
    w.append(i)
    x.append(i)

w = np.array(w)
x = np.array(x)

# Approach 1
start = time.time_ns()

f_1 = 0
for j in range(0, n):
    f_1 += w[j] * x[j]
f_1 += b

f_1_time = time.time_ns() - start
print(f"Time to execute approach 1 {f_1_time}. Result {f_1}")

# Approach 2
start = time.time_ns()

f_2 = np.dot(w, x) + b

f_2_time = time.time_ns() - start
print(f"Time to execute approach 2 {f_2_time}. Result {f_2}")

# Result
time_diff = abs(f_1_time / 1_000_000 - f_2_time / 1_000_000)

print(f"Is Approach 2 faster then Approach 1: '{f_2_time < f_1_time}'. Difference: {time_diff} ms")
