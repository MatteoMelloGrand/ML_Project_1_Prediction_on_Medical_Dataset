"""Train the final model on the full training set and write the AIcrowd submission.

    python run.py

Steps: load the data, preprocess it with the column decisions, train
L2-regularized logistic regression with Newton's method, and classify the test
set with the decision threshold chosen by cross-validation (experiments.py).
Everything is deterministic, so the output file is identical on every run.
"""
import os
import time

import numpy as np

from column_analysis import load_data, load_decisions
from helpers import create_csv_submission
from modeling import f1_score, logistic_newton, sigmoid
from preprocessing import fit_preprocess, transform

DATA_PATH = "data/"
CACHE_PATH = "data/cache.npz"
DECISIONS_PATH = "decisions/column_decisions.csv"
OUTPUT_PATH = "submissions/submission.csv"

# chosen by 5-fold cross-validation in experiments.py (python experiments.py models)
LAMBDA = 0.0001
THRESHOLD = 0.205


def main():
    t0 = time.time()
    x_train, x_test, y_train, _, test_ids, names = load_data(DATA_PATH, CACHE_PATH)
    y = (y_train == 1).astype(np.float32)  # labels {-1, 1} -> {0, 1}
    decisions = load_decisions(DECISIONS_PATH)

    params = fit_preprocess(x_train, names, decisions, {"bias": True})
    z_train = transform(x_train, params)
    print(f"training data: {z_train.shape[0]} rows, {z_train.shape[1] - 1} features")

    w, loss = logistic_newton(y, z_train, LAMBDA)
    train_pred = (sigmoid(z_train @ w.astype(np.float32)) >= THRESHOLD).astype(int)
    print(f"training loss {loss:.4f}, training F1 {f1_score(y, train_pred):.4f}")
    del z_train

    z_test = transform(x_test, params)
    test_pred = sigmoid(z_test @ w.astype(np.float32)) >= THRESHOLD
    labels = np.where(test_pred, 1, -1)  # back to {-1, 1} for AIcrowd

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    create_csv_submission(test_ids, labels, OUTPUT_PATH)
    print(
        f"wrote {OUTPUT_PATH}: {len(labels)} predictions, "
        f"{np.mean(labels == 1):.1%} predicted positive ({time.time() - t0:.0f}s)"
    )


if __name__ == "__main__":
    main()
