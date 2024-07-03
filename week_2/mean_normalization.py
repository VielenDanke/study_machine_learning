import numpy as np


def calculate_mean_max_min(X_train, w_init):
    X_mean_storage = [[] for _ in range(len(X_train[0]))]
    X_max, X_min = [-1 << 30 for _ in range(len(X_train[0]))], [1 << 30 for _ in range(len(X_train[0]))]

    for i in range(len(X_train)):
        for j in range(len(X_train[i])):
            X_mean_storage[j].append(X_train[i][j])
            X_max[j] = max(X_max[j], X_train[i][j])
            X_min[j] = min(X_min[j], X_train[i][j])

    X_mean = np.array([0 for _ in range(w_init.shape[0])])

    for i in range(w_init.shape[0]):
        X_mean[i] = np.mean(X_mean_storage[i])

    return X_mean, X_min, X_max


def calculate_mean_normalization_x_train(X_train, X_mean, X_min, X_max, allowed_min, allowed_max):
    X_train_new = [[0 for _ in range(len(X_train[i]))] for i in range(len(X_train))]

    for i in range(len(X_train)):
        for j in range(len(X_train[i])):
            if X_min[j] > allowed_min and X_max[j] < allowed_max:
                X_train_new[i][j] = X_train[i][j]
            else:
                X_train_max = (X_train[i][j] - X_mean[j]) / (X_max[j] - X_min[j])
                X_train_new[i][j] = X_train_max

    return np.array(X_train_new)
