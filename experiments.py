"""Cross-validation experiments for the report.

    python experiments.py ablation          # Table 1: preprocessing ablation
    python experiments.py models            # Table 2 + lambda/threshold tuning + Figure 2
    python experiments.py models --gd --knn # also gradient descent and KNN (slower)

All experiments use the same stratified folds (seed 1). The decision threshold of
every model is chosen on the training folds and evaluated on the held-out fold.
Results are written to results/ and figures/.
"""
import argparse
import csv
import os
import time

import numpy as np

from column_analysis import load_data, load_decisions
from preprocessing import fit_preprocess, transform
from implementations import least_squares, ridge_regression, reg_logistic_regression
from extra_implementations import K_nearest_neighbors, Soft_margin_SVM
from preprocessing import PCA
from modeling import (
    THRESHOLDS,
    best_threshold,
    f1_per_threshold,
    f1_score,
    logistic_newton,
    sigmoid,
    stratified_folds,
)

SEED = 1
LAMBDA_ABLATION = 1e-4
LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
POS_WEIGHTS = [3.0, 10.0]
RIDGE_LAMBDAS = [1e-5, 1e-3]
GD_LAMBDA, GD_GAMMA, GD_ITERS = 1e-4, 0.2, 300
KNN_COMPONENTS, KNN_NEIGHBORS, KNN_THRESHOLD = (
    15,
    20,
    0.1,
)  # best values from K_NN.ipynb


def evaluate(y_tr, s_tr, y_va, s_va):
    """Choose the threshold on the training scores, evaluate on the validation fold."""
    t = best_threshold(y_tr, s_tr)
    pred = (s_va >= t).astype(int)
    return {"f1": f1_score(y_va, pred), "acc": np.mean(pred == y_va), "thr": t}


def summarize(rows):
    """Mean and std over folds for a list of per-fold result dicts."""
    out = {}
    for key in rows[0]:
        vals = np.array([r[key] for r in rows], dtype=float)
        out[key] = vals.mean()
        out[key + "_std"] = vals.std()
    return out


def save_table(path, table):
    """Write a list of dicts to a CSV file and print it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(table[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(table)
    print(f"\nsaved {path}")
    for row in table:
        print(
            "  "
            + " | ".join(
                f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values()
            )
        )


def without_drops(decisions, x, names, still_drop=("HAREHAB1",)):
    """Decisions where the dropped columns are kept (except the leaky one)."""
    out = {}
    for name, d in decisions.items():
        if d["action"] == "drop" and name not in still_drop:
            col = x[:, names.index(name)]
            n_unique = len(np.unique(col[~np.isnan(col)]))
            out[name] = dict(
                d,
                action="categorical" if n_unique <= 10 else "continuous",
                missing_codes="",
                zero_codes="",
                reason="kept for the ablation",
            )
        else:
            out[name] = d
    return out


def run_ablation(x, y, names, decisions, folds):
    """Table 1: add one preprocessing step at a time (same model for every row)."""
    raw = {"apply_codes": False, "onehot": False, "missing_indicators": False}
    all_cols = without_drops(decisions, x, names)
    leaky = dict(decisions)
    leaky["HAREHAB1"] = dict(decisions["HAREHAB1"], action="categorical")
    variants = [
        ("complete columns only, raw values", all_cols, dict(raw, max_missing=0.0)),
        ("all columns, raw values, median fill", all_cols, raw),
        (
            "+ survey codes (88 -> 0, 77/99 -> missing)",
            all_cols,
            dict(raw, apply_codes=True),
        ),
        (
            "+ one-hot encoding (codes, missing as categories)",
            all_cols,
            {"missing_indicators": False},
        ),
        ("+ missing-value indicators", all_cols, {}),
        ("+ evidence-based column drops (final)", decisions, {}),
        ("final + leaky HAREHAB1 (not used)", leaky, {}),
    ]
    table = []
    for label, dec, options in variants:
        t0, rows = time.time(), []
        for i, va in enumerate(folds):
            tr = np.concatenate([f for j, f in enumerate(folds) if j != i])
            prm = fit_preprocess(x[tr], names, dec, dict(options, bias=True))
            z_tr, z_va = transform(x[tr], prm), transform(x[va], prm)
            w, _ = logistic_newton(y[tr], z_tr, LAMBDA_ABLATION)
            s_tr = sigmoid(z_tr @ w.astype(np.float32))
            s_va = sigmoid(z_va @ w.astype(np.float32))
            rows.append(
                dict(evaluate(y[tr], s_tr, y[va], s_va), n_features=z_tr.shape[1] - 1)
            )
            del z_tr, z_va
        res = summarize(rows)
        table.append(
            {
                "preprocessing": label,
                "f1": res["f1"],
                "f1_std": res["f1_std"],
                "accuracy": res["acc"],
                "n_features": int(res["n_features"]),
            }
        )
        print(
            f"{label:52s} F1 {res['f1']:.4f} ± {res['f1_std']:.4f}  ({time.time() - t0:.0f}s)",
            flush=True,
        )
    save_table("results/ablation.csv", table)


def run_models(x, y, names, decisions, folds, with_gd, with_knn):
    """Table 2 (model comparison), lambda and class-weight tuning, Figure 2."""
    per_model = {}
    lam_curves = {lam: [] for lam in LAMBDAS}
    for i, va in enumerate(folds):
        t0 = time.time()
        tr = np.concatenate([f for j, f in enumerate(folds) if j != i])
        prm = fit_preprocess(x[tr], names, decisions, {"bias": True})
        z_tr, z_va = transform(x[tr], prm), transform(x[va], prm)
        y_tr, y_va = y[tr], y[va]

        def add(name, s_tr, s_va):
            per_model.setdefault(name, []).append(evaluate(y_tr, s_tr, y_va, s_va))

        # least squares and ridge regression (implementations.py), thresholded scores
        try:
            w, _ = least_squares(y_tr, z_tr)
            add("least squares", z_tr @ w, z_va @ w)
        except np.linalg.LinAlgError:
            # indicator columns of the same survey module are identical
            per_model.setdefault("least squares (X^T X singular)", []).append(
                {"f1": np.nan, "acc": np.nan, "thr": np.nan}
            )
        for lam in RIDGE_LAMBDAS:
            w, _ = ridge_regression(y_tr, z_tr, lam)
            add(f"ridge regression (lambda={lam:g})", z_tr @ w, z_va @ w)

        # logistic regression with Newton's method; lambda grid with warm starts
        w = None
        w_ref = None
        for lam in LAMBDAS:
            w, _ = logistic_newton(y_tr, z_tr, lam, initial_w=w)
            s_tr = sigmoid(z_tr @ w.astype(np.float32))
            s_va = sigmoid(z_va @ w.astype(np.float32))
            add(f"logistic, Newton (lambda={lam:g})", s_tr, s_va)
            lam_curves[lam].append(f1_per_threshold(y_va, s_va))
            if lam == LAMBDA_ABLATION:
                w_ref = w.copy()
        for pw in POS_WEIGHTS:
            w, _ = logistic_newton(
                y_tr, z_tr, LAMBDA_ABLATION, initial_w=w_ref, pos_weight=pw
            )
            s_tr = sigmoid(z_tr @ w.astype(np.float32))
            s_va = sigmoid(z_va @ w.astype(np.float32))
            add(
                f"logistic, Newton (lambda={LAMBDA_ABLATION:g}, positive weight {pw:g})",
                s_tr,
                s_va,
            )

        # linear SVM (extra_implementations.py); the bias is already in z
        svm = Soft_margin_SVM(fit_intercept=False, random_state=SEED).fit(z_tr, y_tr)
        add(
            "linear SVM (hinge loss, SGD)",
            svm.decision_function(z_tr),
            svm.decision_function(z_va),
        )

        if with_gd:
            w0 = np.zeros(z_tr.shape[1], dtype=np.float32)
            w, _ = reg_logistic_regression(
                y_tr, z_tr, GD_LAMBDA, w0, GD_ITERS, GD_GAMMA
            )
            add(
                f"logistic, gradient descent ({GD_ITERS} it., gamma={GD_GAMMA:g})",
                sigmoid(z_tr @ w),
                sigmoid(z_va @ w),
            )
        if with_knn:
            pca = PCA(KNN_COMPONENTS).fit(z_tr[:, 1:])
            p_tr, p_va = pca.transform(z_tr[:, 1:]), pca.transform(z_va[:, 1:])
            knn = K_nearest_neighbors(n_neighbors=KNN_NEIGHBORS, batch_size=256).fit(
                p_tr, y_tr
            )
            pred = (knn.predict_proba(p_va) > KNN_THRESHOLD).astype(int)
            per_model.setdefault("KNN + PCA (fixed threshold)", []).append(
                {
                    "f1": f1_score(y_va, pred),
                    "acc": np.mean(pred == y_va),
                    "thr": KNN_THRESHOLD,
                }
            )
        del z_tr, z_va
        print(f"fold {i + 1}/{len(folds)} done ({time.time() - t0:.0f}s)", flush=True)

    table = []
    for name, rows in per_model.items():
        res = summarize(rows)
        table.append(
            {
                "model": name,
                "f1": res["f1"],
                "f1_std": res["f1_std"],
                "accuracy": res["acc"],
                "threshold": res["thr"],
            }
        )
    save_table("results/models.csv", table)

    # best lambda, and the values run.py should use
    lam_rows = [
        r
        for r in table
        if r["model"].startswith("logistic, Newton (lambda=")
        and "weight" not in r["model"]
    ]
    best = max(lam_rows, key=lambda r: r["f1"])
    best_lam = LAMBDAS[
        [f"logistic, Newton (lambda={l:g})" for l in LAMBDAS].index(best["model"])
    ]
    print(
        f"\nFor run.py: LAMBDA = {best_lam:g}, THRESHOLD = {best['threshold']:.3f}  (CV F1 {best['f1']:.4f})"
    )
    plot_figure2(lam_curves, lam_rows, best_lam)


def plot_figure2(lam_curves, lam_rows, best_lam):
    """Figure 2: F1 against the decision threshold, and against lambda."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.8))
    curves = np.array(lam_curves[best_lam])
    mean = curves.mean(axis=0)
    axes[0].plot(THRESHOLDS, mean)
    axes[0].fill_between(
        THRESHOLDS, mean - curves.std(axis=0), mean + curves.std(axis=0), alpha=0.3
    )
    i_half = int(np.argmin(np.abs(THRESHOLDS - 0.5)))
    axes[0].plot([0.5], [mean[i_half]], "o", color="tab:red", markersize=3)
    axes[0].set_xlabel("decision threshold")
    axes[0].set_ylabel("validation F1")
    f1s = [r["f1"] for r in lam_rows]
    stds = [r["f1_std"] for r in lam_rows]
    axes[1].errorbar(LAMBDAS, f1s, yerr=stds, marker="o", markersize=3)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("lambda")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig2_threshold_lambda.pdf")
    print("saved figures/fig2_threshold_lambda.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=["ablation", "models"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--gd", action="store_true", help="also gradient descent")
    parser.add_argument("--knn", action="store_true", help="also KNN + PCA")
    parser.add_argument(
        "--rows", type=int, default=0, help="quick test on the first rows only"
    )
    args = parser.parse_args()

    x, _, y_pm, _, _, names = load_data("data/", "data/cache.npz")
    if args.rows:
        x, y_pm = x[: args.rows], y_pm[: args.rows]
    y = (y_pm == 1).astype(np.float32)
    decisions = load_decisions("decisions/column_decisions.csv")
    folds = stratified_folds(y, args.folds, SEED)
    if args.group == "ablation":
        run_ablation(x, y, names, decisions, folds)
    else:
        run_models(x, y, names, decisions, folds, args.gd, args.knn)


if __name__ == "__main__":
    main()
