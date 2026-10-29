"""Model training and evaluation used by experiments.py and run.py.

The final model is L2-regularized logistic regression trained with Newton's
method (lab 5). Plain gradient descent (reg_logistic_regression in
implementations.py) converges very slowly on our ~1250 correlated one-hot
features, while Newton's method converges in about 10 steps.
"""
import numpy as np

THRESHOLDS = np.linspace(0.02, 0.8, 157)

# answers of these columns are multiplied by age (_AGE80) in the submitted model
INTERACTIONS = (
    "SEX",
    "GENHLTH",
    "BPHIGH4",
    "TOLDHI2",
    "DIABETE3",
    "CVDSTRK3",
    "CHCKIDNY",
)


def with_leaky_column(decisions, name="HAREHAB1"):
    """Copy of the decisions in which the leaky column is kept as categorical.

    HAREHAB1 (cardiac rehabilitation) is only asked after a heart attack, so its
    presence reveals the label. It is excluded from our analysis, but it is part of
    the test data, so the submitted model uses it (see the report).
    """
    out = dict(decisions)
    out[name] = dict(decisions[name], action="categorical")
    return out


def sigmoid(t):
    """Logistic function, clipped to avoid overflow in exp."""
    return 1.0 / (1.0 + np.exp(-np.clip(t, -30, 30)))


def logistic_loss(y, tx, w):
    """Mean logistic loss (negative log-likelihood), without the penalty term."""
    z = tx @ w
    return float(np.mean(np.logaddexp(0, z) - y * z))


def logistic_newton(
    y,
    tx,
    lambda_,
    initial_w=None,
    max_iters=20,
    tol=1e-6,
    pos_weight=1.0,
    chunk_size=20000,
):
    """L2-regularized logistic regression trained with Newton's method.

    Minimizes mean_n v_n * logloss_n + lambda_ * ||w||^2, where v_n = pos_weight
    for positive samples and 1 otherwise. The bias is the first column of tx and
    is regularized like the other weights (its effect is negligible here).
    The Hessian is accumulated in chunks of rows to keep memory low.

    Args:
        y (np.array): labels in {0, 1}, shape (N,)
        tx (np.array): features with a bias column, shape (N, D), float32
        lambda_ (float): regularization strength
        initial_w (np.array, optional): starting point (warm start), shape (D,)
        max_iters (int): maximum number of Newton steps
        tol (float): stop when the largest weight update is below tol
        pos_weight (float): weight of positive samples (class weighting)

    Returns:
        w (np.array): weights, shape (D,)
        loss (float): mean logistic loss without the penalty term
    """
    n, d = tx.shape
    w = np.zeros(d) if initial_w is None else np.array(initial_w, dtype=np.float64)
    v = np.where(y == 1, pos_weight, 1.0)

    def objective(w):
        z = tx @ w.astype(np.float32)
        return np.mean(v * (np.logaddexp(0, z) - y * z)) + lambda_ * w @ w

    current = objective(w)
    for _ in range(max_iters):
        p = sigmoid(tx @ w.astype(np.float32))
        grad = tx.T @ (v * (p - y)).astype(np.float32) / n + 2 * lambda_ * w
        s = (v * p * (1 - p)).astype(np.float32)
        hess = np.zeros((d, d))
        for start in range(0, n, chunk_size):
            c = tx[start : start + chunk_size]
            hess += c.T @ (c * s[start : start + chunk_size, None])
        hess = hess / n + 2 * lambda_ * np.eye(d)
        step = np.linalg.solve(hess, grad)
        # halve the step while it makes the objective worse (keeps Newton stable)
        for _ in range(20):
            new = objective(w - step)
            if new <= current:
                break
            step = step / 2
        w = w - step
        current = new
        if np.max(np.abs(step)) < tol:
            break
    return w, logistic_loss(y, tx, w.astype(np.float32))


def f1_score(y, y_pred):
    """F1 score of predictions y_pred in {0, 1} against labels y in {0, 1}."""
    tp = np.sum((y_pred == 1) & (y == 1))
    fp = np.sum((y_pred == 1) & (y == 0))
    fn = np.sum((y_pred == 0) & (y == 1))
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0.0


def f1_per_threshold(y, scores, thresholds=THRESHOLDS):
    """F1 score for every threshold in thresholds (score >= threshold -> 1)."""
    return np.array([f1_score(y, (scores >= t).astype(int)) for t in thresholds])


def best_threshold(y, scores, thresholds=THRESHOLDS):
    """Threshold that maximizes F1 on (y, scores)."""
    return thresholds[int(np.argmax(f1_per_threshold(y, scores, thresholds)))]


def stratified_folds(y, k, seed):
    """Split indices into k folds with the same share of positives in each."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in (0, 1):
        idx = rng.permutation(np.where(y == cls)[0])
        for i, part in enumerate(np.array_split(idx, k)):
            folds[i].extend(part)
    return [np.sort(np.array(f)) for f in folds]
