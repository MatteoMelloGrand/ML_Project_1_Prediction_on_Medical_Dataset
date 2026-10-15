"""Preprocessing of the BRFSS data.

How each column is handled is read from decisions/column_decisions.csv, which is
created in notebooks/01_column_analysis.ipynb. fit_preprocess learns everything it
needs (categories, medians, means, stds) from the training data only, and transform
applies it to any data. Each step can be turned off with the options argument,
which we use for the ablation study.
"""
import numpy as np

DEFAULT_OPTIONS = {
    "apply_drops": True,  # drop the columns marked "drop"
    "max_missing": 1.0,  # also drop columns with a larger missing fraction
    "apply_codes": True,  # replace missing_codes by NaN and zero_codes by 0
    "onehot": True,  # one-hot encode categorical columns, missing included
    "min_count": 50,  # minimum count for a category to get a column
    "drop_first": False,  # use k-1 dummies (for least squares without regularization)
    "missing_indicators": True,  # add a 0/1 column marking missing values
    "standardize": True,  # standardize with the training mean and std
}


def _codes(text):
    return [float(t) for t in str(text).split()] if text else []


def _apply_codes(col, decision, as_continuous):
    """Return a copy of the column with its special codes replaced."""
    col = np.array(col, dtype=np.float64, copy=True)
    for c in _codes(decision.get("missing_codes")):
        col[col == c] = np.nan
    if as_continuous:
        for c in _codes(decision.get("zero_codes")):
            col[col == c] = 0.0
    return col


def fit_preprocess(x, names, decisions, options=None):
    """Learn the preprocessing from the training data.

    Args:
        x (np.array): raw training features, shape (N, D)
        names (list): the D column names
        decisions (dict): {column name: decision}, from load_decisions
        options (dict, optional): values that replace DEFAULT_OPTIONS

    Returns:
        params (dict): what transform needs, plus a report of what was done
    """
    opt = dict(DEFAULT_OPTIONS)
    opt.update(options or {})
    names = list(names)
    missing = np.isnan(x).mean(axis=0)
    specs, report = [], []

    for j, name in enumerate(names):
        d = decisions.get(
            name, {"action": "continuous", "reason": "no decision recorded"}
        )
        action, reason = d["action"], d.get("reason", "")
        if action == "drop" and opt["apply_drops"]:
            report.append((name, "dropped", "", reason))
            continue
        if missing[j] > opt["max_missing"]:
            report.append(
                (name, "dropped", "", f"missing {missing[j]:.0%} > max_missing")
            )
            continue
        if action == "drop":  # only reached when apply_drops is False
            action = (
                "categorical"
                if len(np.unique(x[~np.isnan(x[:, j]), j])) <= 10
                else "continuous"
            )

        if action == "categorical" and opt["onehot"]:
            col = (
                _apply_codes(x[:, j], d, as_continuous=False)
                if opt["apply_codes"]
                else x[:, j].astype(np.float64)
            )
            vals, counts = np.unique(col[~np.isnan(col)], return_counts=True)
            levels = [v for v, c in zip(vals, counts) if c >= opt["min_count"]]
            if np.isnan(col).sum() >= opt["min_count"]:
                levels.append(np.nan)
            if opt["drop_first"]:
                levels = levels[1:]
            if not levels:
                report.append((name, "dropped", "", "no answer frequent enough"))
                continue
            specs.append(
                {"name": name, "j": j, "kind": "cat", "levels": levels, "decision": d}
            )
            shown = ", ".join("missing" if np.isnan(v) else f"{v:g}" for v in levels)
            report.append((name, "one-hot", f"{len(levels)} columns: {shown}", reason))
        else:
            col = (
                _apply_codes(x[:, j], d, as_continuous=True)
                if opt["apply_codes"]
                else x[:, j].astype(np.float64)
            )
            if np.all(np.isnan(col)):
                report.append((name, "dropped", "", "empty after code handling"))
                continue
            ind = bool(opt["missing_indicators"] and np.isnan(col).mean() > 0.001)
            spec = {
                "name": name,
                "j": j,
                "kind": "num",
                "median": float(np.nanmedian(col)),
                "indicator": ind,
                "decision": d,
            }
            specs.append(spec)
            detail = f"median fill {spec['median']:g}" + (
                " + missing indicator" if ind else ""
            )
            report.append((name, "continuous", detail, reason))

    params = {
        "specs": specs,
        "options": opt,
        "mean": None,
        "std": None,
        "report": report,
    }
    if opt["standardize"]:
        raw = _build(x, params)
        params["mean"] = raw.mean(axis=0, dtype=np.float64).astype(np.float32)
        params["std"] = (raw.std(axis=0, dtype=np.float64) + 1e-8).astype(np.float32)
        del raw
    params["feature_names"] = _feature_names(params)
    return params


def _n_out(params):
    """Return the number of columns of the output matrix."""
    return sum(
        len(s["levels"]) if s["kind"] == "cat" else 1 + int(s["indicator"])
        for s in params["specs"]
    )


def _build(x, params):
    """Build the feature matrix column by column (float32, to save memory)."""
    opt = params["options"]
    out = np.empty((x.shape[0], _n_out(params)), dtype=np.float32)
    k = 0
    for s in params["specs"]:
        if opt["apply_codes"]:
            col = _apply_codes(
                x[:, s["j"]], s["decision"], as_continuous=(s["kind"] == "num")
            )
        else:
            col = x[:, s["j"]].astype(np.float64)
        if s["kind"] == "cat":
            for v in s["levels"]:
                out[:, k] = np.isnan(col) if np.isnan(v) else (col == v)
                k += 1
        else:
            miss = np.isnan(col)
            if s["indicator"]:
                out[:, k] = miss
                k += 1
            col[miss] = s["median"]
            out[:, k] = col
            k += 1
    return out


def _feature_names(params):
    out = []
    for s in params["specs"]:
        if s["kind"] == "cat":
            out += [
                f"{s['name']}=missing" if np.isnan(v) else f"{s['name']}={v:g}"
                for v in s["levels"]
            ]
        else:
            if s["indicator"]:
                out.append(f"{s['name']}_missing")
            out.append(s["name"])
    return out


def transform(x, params):
    """Apply the fitted preprocessing to x (train, validation or test data)."""
    z = _build(x, params)
    if params["mean"] is not None:
        z -= params["mean"]  # in place to avoid copying the matrix
        z /= params["std"]
    return z


def print_report(params, show=("dropped", "one-hot", "continuous"), max_rows=None):
    """Print what was done to each column and why."""
    rep = params["report"]
    counts = {}
    for _, what, _, _ in rep:
        counts[what] = counts.get(what, 0) + 1
    print(
        "SUMMARY:",
        ", ".join(f"{k}: {v}" for k, v in counts.items()),
        f"| final number of features: {len(params['feature_names'])}",
    )
    for what in show:
        rows = [r for r in rep if r[1] == what]
        print(f"\n--- {what} ({len(rows)}) ---")
        for name, _, detail, reason in rows[:max_rows]:
            print(f"  {name:10s} {detail[:60]:60s} | {reason}")