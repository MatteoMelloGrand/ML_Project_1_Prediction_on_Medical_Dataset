"""Functions to describe the columns of the BRFSS dataset.

They are used in notebooks/01_column_analysis.ipynb, where we decide how each column
is handled. The decisions are saved to decisions/column_decisions.csv and applied
in preprocessing.py.
"""
import csv
import os

import numpy as np

# Values that BRFSS often uses for "don't know", "refused", "none" or "never".
# Their meaning depends on the column, so they are checked column by column in the notebook.
# See the BRFSS 2015 codebook (e.g. GENHLTH p.13, PHYSHLTH p.13, DIABAGE2 p.21):
# https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf
SPECIAL_CODES = [7, 9, 77, 88, 99, 555, 777, 888, 999]


def load_data(data_path, cache_path=None):
    """Load the data and the column names.

    Reads the CSV files line by line, which uses less memory than load_csv_data.
    If cache_path is given, the arrays are saved there and loaded from it next time.

    Args:
        data_path (str): folder with x_train.csv, y_train.csv and x_test.csv
        cache_path (str, optional): path of the .npz cache file

    Returns:
        x_train, x_test, y_train (labels in {-1, 1}), train_ids, test_ids, names
    """
    if cache_path and os.path.exists(cache_path):
        c = np.load(cache_path)
        return (
            c["x_train"],
            c["x_test"],
            c["y_train"],
            c["train_ids"],
            c["test_ids"],
            [str(s) for s in c["names"]],
        )

    def read(path):
        with open(path) as f:
            header = f.readline().strip().split(",")
            n_rows = sum(1 for _ in f)
        arr = np.empty((n_rows, len(header)), dtype=np.float32)
        with open(path) as f:
            f.readline()
            for i, line in enumerate(f):
                arr[i] = [
                    float(t) if t else np.nan for t in line.rstrip("\n").split(",")
                ]
        return header, arr

    header, tr = read(os.path.join(data_path, "x_train.csv"))
    _, te = read(os.path.join(data_path, "x_test.csv"))
    _, y = read(os.path.join(data_path, "y_train.csv"))
    assert (y[:, 0] == tr[:, 0]).all(), "x_train and y_train ids do not match"
    out = (
        tr[:, 1:],
        te[:, 1:],
        y[:, 1].astype(int),
        tr[:, 0].astype(int),
        te[:, 0].astype(int),
        header[1:],
    )
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(
            cache_path,
            x_train=out[0],
            x_test=out[1],
            y_train=out[2],
            train_ids=out[3],
            test_ids=out[4],
            names=np.array(out[5]),
        )
    return out


def column_profile(x_train, x_test, y01, names, min_group=500):
    """Compute summary statistics for every column.

    label_gap is the difference between the highest and the lowest positive rate
    over the values that at least min_group people gave (missing counts as a value).
    A large gap means the column is related to the label.

    Args:
        x_train, x_test (np.array): raw features
        y01 (np.array): training labels in {0, 1}
        names (list): column names
        min_group (int): minimum group size used for label_gap

    Returns:
        list of dicts, one per column
    """
    pos = y01 == 1
    profile = []
    for j, name in enumerate(names):
        col = x_train[:, j].astype(np.float64)
        nan = np.isnan(col)
        vals, counts = np.unique(col[~nan], return_counts=True)
        order = np.argsort(-counts)[:6]
        codes = {c: float(np.mean(col == c)) for c in SPECIAL_CODES if np.any(col == c)}
        rates = []
        if len(vals) <= 60:
            rates = [
                pos[col == v].mean() for v, k in zip(vals, counts) if k >= min_group
            ]
        if nan.sum() >= min_group:
            rates.append(pos[nan].mean())
        profile.append(
            {
                "name": name,
                "missing_train": float(nan.mean()),
                "missing_test": float(np.isnan(x_test[:, j]).mean()),
                "n_unique": int(len(vals)),
                "min": float(vals.min()) if len(vals) else np.nan,
                "max": float(vals.max()) if len(vals) else np.nan,
                "top_values": [(float(vals[i]), int(counts[i])) for i in order],
                "codes": codes,
                "label_gap": float(max(rates) - min(rates))
                if len(rates) > 1
                else np.nan,
                "pos_rate_missing": float(pos[nan].mean()) if nan.any() else np.nan,
                "pos_rate_present": float(pos[~nan].mean()) if (~nan).any() else np.nan,
            }
        )
    return profile


def label_rate_table(x, y01, names, name, min_count=1):
    """Return (value, count, positive rate) for every value of one column.

    Missing values are reported as a separate row with value nan.
    """
    col = x[:, names.index(name)].astype(np.float64)
    pos = y01 == 1
    rows = []
    for v in np.unique(col[~np.isnan(col)]):
        m = col == v
        if m.sum() >= min_count:
            rows.append((float(v), int(m.sum()), float(pos[m].mean())))
    m = np.isnan(col)
    if m.sum() >= min_count:
        rows.append((np.nan, int(m.sum()), float(pos[m].mean())))
    return rows


def print_label_rates(x, y01, names, name, min_count=1):
    """Print the output of label_rate_table."""
    print(f"{name}:  value   count   positive rate")
    for v, n, r in label_rate_table(x, y01, names, name, min_count):
        print(f"   {'NaN' if np.isnan(v) else f'{v:g}':>8} {n:8d}   {r:.3f}")


def find_exact_duplicates(x, names):
    """Return pairs (column, earlier identical column), comparing missing values too."""
    seen, pairs = {}, []
    for j, name in enumerate(names):
        key = np.nan_to_num(x[:, j], nan=-1e9).tobytes()
        if key in seen:
            pairs.append((name, seen[key]))
        else:
            seen[key] = name
    return pairs


def find_correlated_pairs(x, names, threshold=0.95, max_rows=50000, seed=0):
    """Return pairs of columns whose absolute correlation is at least threshold.

    The correlation is computed on a random sample of rows, using only the rows where
    both columns have a value. Survey codes are not removed first.

    Returns:
        list of (name_a, name_b, correlation, number of rows used)
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(max_rows, len(x)), replace=False)
    sub = x[idx].astype(np.float64)
    present = ~np.isnan(sub)
    out = []
    for a in range(sub.shape[1]):
        for b in range(a + 1, sub.shape[1]):
            m = present[:, a] & present[:, b]
            if m.sum() < 1000:
                continue
            u, v = sub[m, a], sub[m, b]
            if u.std() == 0 or v.std() == 0:
                continue
            r = np.corrcoef(u, v)[0, 1]
            if abs(r) >= threshold:
                out.append((names[a], names[b], float(r), int(m.sum())))
    return out


def find_determined_columns(x, names, max_values=11, exclude=()):
    """Return pairs (b, a) where column b is fully determined by column a.

    b is determined by a if every value of a (missing counted as a value) always comes
    with the same value of b. Only columns a with at most max_values values are used,
    because then a is one-hot encoded and b adds no information to the model.
    """
    codes = {}
    for j, name in enumerate(names):
        if name in exclude:
            continue
        v = np.nan_to_num(x[:, j].astype(np.float64), nan=-1e12)
        u, inv = np.unique(v, return_inverse=True)
        codes[name] = (inv.astype(np.int64), len(u))
    pairs = []
    for a, (ia, ka) in codes.items():
        if ka > max_values:
            continue
        for b, (ib, kb) in codes.items():
            if a == b or kb > ka or kb < 2:
                continue
            if len(np.unique(ia * 1000003 + ib)) == ka:
                pairs.append((b, a))
    return pairs


def choose_determined_drops(pairs, names):
    """Choose which determined columns to drop so that the information is kept.

    Columns are visited from last to first; a column is dropped if a column that is
    still kept determines it. Returns a dict {dropped column: column that determines it}.
    """
    by_b = {}
    for b, a in pairs:
        by_b.setdefault(b, []).append(a)
    order = {n: i for i, n in enumerate(names)}
    dropped = {}
    for b in sorted(by_b, key=lambda n: -order[n]):
        kept = [a for a in by_b[b] if a not in dropped]
        if kept:
            dropped[b] = min(kept, key=lambda n: order[n])
    # follow chains (c determined by b, b determined by a) to the column that is kept
    for b in dropped:
        while dropped[b] in dropped:
            dropped[b] = dropped[dropped[b]]
    return dropped


def leakage_scan(x, y01, names, min_answered=100, min_pos_rate=0.6):
    """Return columns that were answered mostly by positive people.

    Such a column is probably a follow-up question asked only after a heart problem,
    so the fact that it has a value already gives away the label.

    Returns:
        list of (name, number of answers, positive rate among them)
    """
    pos = y01 == 1
    out = []
    for j, name in enumerate(names):
        m = ~np.isnan(x[:, j])
        if m.sum() >= min_answered and pos[m].mean() >= min_pos_rate:
            out.append((name, int(m.sum()), float(pos[m].mean())))
    return out


DECISION_FIELDS = ["name", "action", "missing_codes", "zero_codes", "reason"]
# action is one of drop, categorical, continuous.
# missing_codes are set to NaN, zero_codes are set to 0 (space separated values).


def save_decisions(decisions, path):
    """Save the list of column decisions to a CSV file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DECISION_FIELDS)
        w.writeheader()
        for d in decisions:
            w.writerow({k: d.get(k, "") for k in DECISION_FIELDS})


def load_decisions(path):
    """Load the decisions CSV into a dict {column name: decision}."""
    with open(path, newline="") as f:
        return {row["name"]: row for row in csv.DictReader(f)}


def summarize_decisions(decisions, max_reasons=None):
    """Print the number of columns per action and their reasons."""
    by_action = {}
    for d in decisions:
        by_action.setdefault(d["action"], []).append(d)
    for action, ds in sorted(by_action.items()):
        print(f"{action:12s}: {len(ds)} columns")
        reasons = {}
        for d in ds:
            reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
        for r, n in sorted(reasons.items(), key=lambda t: -t[1])[:max_reasons]:
            print(f"      {n:4d}  {r}")
