import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Returns the last weight vector w and its MSE loss.
    """
    w = initial_w
    N = len(y)
    for n_iter in range(max_iters):
        e = y - tx.dot(w)
        grad = -tx.T @ e / N
        w = w - gamma * grad
    final_loss = 0.5 * np.mean((y - tx.dot(w)) ** 2)
    return w, final_loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent.

    Uses mini-batch size 1 (one random data point per step).
    Returns the last weight vector w and its MSE loss on the full dataset.
    """

    w = initial_w
    N = len(y)
    for n_iter in range(max_iters):
        n = np.random.randint(N)
        e_n = y[n] - tx[n].dot(w)
        grad = -tx[n] * e_n
        w = w - gamma * grad

    final_loss = 0.5 * np.mean((y - tx @ w) ** 2)
    return w, final_loss


def least_squares(y, tx):
    """Least squares regression using normal equations.

    Solves X^T X w = X^T y. Returns the optimal w and its MSE loss.
    """
    w_optimal = np.linalg.solve(tx.T @ tx, tx.T @ y)
    mse = 0.5 * np.mean((y - tx.dot(w_optimal)) ** 2)
    return w_optimal, mse


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.

    Solves (X^T X + 2N*lambda_ I) w = X^T y.
    Returns the optimal w and its MSE loss (without the penalty term).
    """

    N, D = tx.shape
    lambda_prime = 2 * N * lambda_
    w = np.linalg.solve(tx.T @ tx + lambda_prime * np.eye(D), tx.T @ y)
    e = y - tx @ w
    loss = e @ e / (2 * N)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.

    Returns the last weight vector w and its logistic loss.
    """
    w = initial_w
    N = len(y)
    for n_iter in range(max_iters):
        sigma = 1 / (1 + np.exp(-tx @ w))
        grad = tx.T @ (sigma - y) / N
        w = w - gamma * grad
    z = tx @ w
    loss = np.mean(np.log(1 + np.exp(z)) - y * z)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized ridge logistic regression using gradient descent

    Returns the last w and its logistic loss (without the penalty term).
    """
    w = initial_w
    N = len(y)
    for n_iter in range(max_iters):
        sigma = 1 / (1 + np.exp(-tx @ w))
        grad = tx.T @ (sigma - y) / N + 2 * lambda_ * w
        w = w - gamma * grad
    z = tx @ w
    loss = np.mean(np.log(1 + np.exp(z)) - y * z)
    return w, loss
