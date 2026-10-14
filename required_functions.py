from helpers import batch_iter
import numpy as np


def mean_squared_error_gd(y,tx,initial_w, max_iters,gamma):
    w = initial_w
    for n_iter in range(max_iters):
        e = (y - tx.dot(w))
        grad = -tx@e / len(e)
        w = w - gamma * grad
    final_loss = 0.5*np.mean((y-tx.dot(w))**2)
    return w, final_loss


def mean_squared_error_sgd(y, tx, initial_w, batch_size, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            e = y_batch - tx_batch.dot(w)
            grad = -tx_batch @ e / len(e)
            w = w - gamma * grad
    final_loss = 0.5*np.mean((y - tx @ w) ** 2)
    print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=final_loss, w0=w[0], w1=w[1]))
    return w, final_loss

def least_squares(y, tx):
    w_optimal = np.linalg.inv(tx.T@tx)@tx.T@y
    mse = 0.5*np.mean((y-tx.dot(w_optimal))**2)
    return w_optimal, mse