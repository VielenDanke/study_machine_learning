# new_w = w - a * (d / d * w) * J(w, b)
# a - learning rate (between 0 and 1), controls how big is the step we're taking
# (d / d * w) * J(w, b) - derivative term, which direction we're taking
# For derivative w - 1 / m SUM (f(x[i]) - y[i]) * x[i], where f(x[i]) = w (vector) * x[i] (vector) + b
#
# new_b = b - a * (d / d * b) * J(w, b)
# a - learning rate (between 0 and 1), controls how big is the step we're taking
# (d / d * b) * J(w, b) - derivative term, which direction we're taking
# For derivative b - 1 / m SUM (f(x[i]) - y[i]), where f(x[i]) = w (vector) * x[i] (vector) + b
#
# Simultaneously update w and b
