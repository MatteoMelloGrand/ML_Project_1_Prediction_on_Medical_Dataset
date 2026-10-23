import time
from collections import Counter
import numpy as np
from preprocessing import PCA, fit_preprocess, transform
def f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def stratified_kfold_indices(y, n_splits, rng):
    """Yield (train_idx, val_idx), keeping class proportions in each fold."""
    folds = [[] for _ in range(n_splits)]
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        # deal samples round-robin, starting at the currently smallest fold
        start = int(np.argmin([len(f) for f in folds]))
        for j, sample in enumerate(idx):
            folds[(start + j) % n_splits].append(sample)
    for i in range(n_splits):
        val = np.array(sorted(folds[i]))
        train = np.setdiff1d(np.arange(len(y)), val)
        yield train, val



class K_nearest_neighbors:

    def __init__(self, n_neighbors=5, batch_size=32, threshold=0.5):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.threshold = threshold

    def fit(self, x, y):
        self.x_train = np.asarray(x, dtype=np.float32)
        self.y_train = np.asarray(y)
        self.train_sq = (self.x_train ** 2).sum(axis=1)
        return self

    def predict_proba(self, x):
        x = np.asarray(x, dtype=np.float32)
        k = self.n_neighbors
        out = np.empty(len(x))
        for start in range(0, len(x), self.batch_size):
            chunk = x[start:start + self.batch_size]                  # (B, D)
            chunk_sq = (chunk ** 2).sum(axis=1)[:, None]              # (B, 1)
            dist = chunk_sq - 2 * chunk @ self.x_train.T + self.train_sq  # (B, N)
            idx = np.argpartition(dist, k - 1, axis=1)[:, :k]         # (B, k)
            out[start:start + self.batch_size] = self.y_train[idx].mean(axis=1)
        return out
   
    def predict(self, x):
        return (self.predict_proba(x) > self.threshold).astype(int)


class grid_search_KNN:
    """K-fold grid search over PCA components, KNN neighbors and threshold.

    PCA is fit on each training fold only. It is fit once with max(n_components)
    and sliced for smaller values (exact for PCA). predict_proba does not depend
    on the threshold, so it is computed once per (n_components, k) and every
    threshold is applied to it.
    """

    def __init__(self, n_components=(10, 20, 30, 50, 70, 100),
                 thresholds=(0.08, 0.1, 0.2, 0.3, 0.5),
                 n_neighbors=(5, 10, 15, 20),
                 n_splits=10, batch_size=256, scoring=f1_score,
                 random_state=42, verbose=True):
        self.verbose = verbose
        self.n_components = n_components
        self.thresholds = thresholds
        self.n_neighbors = n_neighbors
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.scoring = scoring
        self.random_state = random_state

    def fit(self, x, y, names, decisions, preprocess_options=None):
        x = np.asarray(x)
        y = np.asarray(y)
        self.names_ = list(names)
        self.decisions_ = decisions
        self.preprocess_options_ = preprocess_options

        rng = np.random.default_rng(self.random_state)
        scores = {(c, k, t): [] for c in self.n_components
                  for k in self.n_neighbors for t in self.thresholds}

        n_combos = len(scores)
        total = n_combos * self.n_splits
        done = 0
        start_time = time.time()

        for fold, (train_idx, val_idx) in enumerate(
            stratified_kfold_indices(y, self.n_splits, rng), 1
        ):
            params = fit_preprocess(
                x[train_idx], self.names_, self.decisions_,
                options=self.preprocess_options_,
            )
            x_train_fold = transform(x[train_idx], params)
            x_val_fold = transform(x[val_idx], params)
            pca = PCA(max(self.n_components)).fit(x_train_fold)
            z_train = pca.transform(x_train_fold)
            z_val = pca.transform(x_val_fold)
            y_val = y[val_idx]

            for c in self.n_components:
                for k in self.n_neighbors:
                    knn = K_nearest_neighbors(n_neighbors=k,
                                              batch_size=self.batch_size)
                    knn.fit(z_train[:, :c], y[train_idx])
                    proba = knn.predict_proba(z_val[:, :c])
                    for t in self.thresholds:
                        scores[(c, k, t)].append(
                            self.scoring(y_val, (proba > t).astype(int)))
                    done += len(self.thresholds)
                    if self.verbose:
                        elapsed = time.time() - start_time
                        eta = elapsed / done * (total - done)
                        print(
                            f"\rfold {fold}/{self.n_splits} | "
                            f"evaluations {done}/{total} ({done / total:.0%}) | "
                            f"elapsed {elapsed / 60:.1f} min, "
                            f"ETA {eta / 60:.1f} min",
                            end="", flush=True,
                        )

        if self.verbose:
            print()
        self.results_ = [
            {"n_components": c, "n_neighbors": k, "threshold": t,
             "mean_score": np.mean(s), "std_score": np.std(s)}
            for (c, k, t), s in scores.items()
        ]
        self.results_.sort(key=lambda r: r["mean_score"], reverse=True)
        self.best_params_ = {key: self.results_[0][key] for key in
                             ("n_components", "n_neighbors", "threshold")}
        self.best_score_ = self.results_[0]["mean_score"]
        return self

    def fit_best(self, x, y):
        """Fit preprocessing, PCA and KNN on the full training data."""
        self.preprocess_params_ = fit_preprocess(
            x, self.names_, self.decisions_, options=self.preprocess_options_
        )
        x_processed = transform(x, self.preprocess_params_)
        p = self.best_params_
        self.model_ = K_nearest_neighbors(n_neighbors=p["n_neighbors"],
                                          batch_size=self.batch_size,
                                          threshold=p["threshold"])
        self.pca_ = PCA(p["n_components"]).fit(x_processed)
        self.model_.fit(self.pca_.transform(x_processed), y)
        return self

    def predict(self, x):
        x_processed = transform(x, self.preprocess_params_)
        return self.model_.predict(self.pca_.transform(x_processed))