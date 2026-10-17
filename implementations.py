import numpy as np

# They have test whether the functions have docstrings, so i added


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Returns the last weight vector w and its MSE loss.
    """
    w = initial_w
    N = len(y)
    for n_iter in range(max_iters):
        e = y - tx.dot(w)
        grad = -tx.T @ e / N  # I only put here .T
        w = w - gamma * grad
    final_loss = 0.5 * np.mean((y - tx.dot(w)) ** 2)
    return w, final_loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent.

    Uses mini-batch size 1 (one random data point per step).
    Returns the last weight vector w and its MSE loss on the full dataset.
    """
    # I think the signitures of the functions need to match the ones given from them
    # I just removed batch_size becase they dont have it, so i think it means to always
    # use batch_size = 1
    w = initial_w
    N = len(y)
    for n_iter in range(max_iters):
        n = np.random.randint(N)
        e_n = y[n] - tx[n].dot(w)
        grad = -tx[n] * e_n  # I just removed the division because batch_size=1
        w = w - gamma * grad

    final_loss = 0.5 * np.mean((y - tx @ w) ** 2)
    return w, final_loss


# They want the equation to be solved with linear equation system, so i just put that
def least_squares(y, tx):
    """Least squares regression using normal equations.

    Solves X^T X w = X^T y. Returns the optimal w and its MSE loss.
    """
    w_optimal = np.linalg.solve(tx.T @ tx, tx.T @ y)  #  X^T X w = X^T y
    mse = 0.5 * np.mean((y - tx.dot(w_optimal)) ** 2)
    return w_optimal, mse


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.

    Solves (X^T X + 2N*lambda_ I) w = X^T y.
    Returns the optimal w and its MSE loss (without the penalty term).
    """

    N, D = tx.shape
    lambda_prime = 2 * N * lambda_  # lambda' = 2N * lambda
    w = np.linalg.solve(tx.T @ tx + lambda_prime * np.eye(D), tx.T @ y)
    # (X^T X + lambda' I) w = X^T y
    e = y - tx @ w
    loss = e @ e / (2 * N)
    return w, loss
