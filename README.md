# ML Project 1: Predicting coronary heart disease (MICHD) from BRFSS 2015

EPFL CS-433 Machine Learning, Project 1 (Fall 2026).

## Team

1. Simona Ristovska (simona.ristovska@epfl.ch)
2. Matej Mitev (matej.mitev@epfl.ch)
3. Matteo Mello Grand (matteo.mellogrand@epfl.ch)

## Repository structure

| Path | Content |
| --- | --- |
| `implementations.py` | The 6 required methods (`mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression`, `logistic_regression`, `reg_logistic_regression`) |
| `run.py` | Trains the final model and writes the submission (reproduces our best AIcrowd submission) |
| `experiments.py` | Cross-validation experiments for the report: preprocessing ablation and model comparison |
| `modeling.py` | Logistic regression with Newton's method, F1 score, threshold selection, stratified folds |
| `extra_implementations.py` | Additional models: K-nearest neighbors, grid search, linear SVMs |
| `preprocessing.py` | Preprocessing driven by `decisions/column_decisions.csv`, and PCA |
| `column_analysis.py` | Functions that measure the columns of the dataset |
| `helpers.py` | Course helper functions (`load_csv_data`, `create_csv_submission`), unchanged |
| `notebooks/` | `01_column_analysis.ipynb` (column decisions, Figure 1), `K_NN.ipynb` (KNN tuning) |
| `decisions/` | `column_decisions.csv`: how each column is handled, with the reason |
| `results/` | Tables produced by `experiments.py` |
| `figures/` | Figures used in the report |
| `submissions/` | `submission.csv`: the predictions submitted to AIcrowd (submission 338161), produced by `run.py` |
| `data/` | Dataset (not committed, see Setup) |

## Setup

Python 3.9, same package versions as the course grading environment:

```bash
conda create -n ml_project1 python=3.9 numpy=1.23.1 matplotlib=3.5.2 ipykernel -y
conda activate ml_project1
python -m pip install black==22.6.0
```

Download the data from the [AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
(Resources tab) and put `x_train.csv`, `y_train.csv` and `x_test.csv` directly in `data/` without renaming them.

## Reproducing the submission

```bash
python run.py
```

This writes `submissions/submission.csv` (about 2 minutes, 4 GB of memory). The first run also
creates `data/cache.npz`, a faster copy of the CSV files. The column decisions used by the code are
committed in `decisions/column_decisions.csv`; running `notebooks/01_column_analysis.ipynb`
regenerates them.

To reproduce the tables and figures of the report:

```bash
python experiments.py ablation          # Table 1, results/ablation.csv
python experiments.py models --gd --knn # Table 2 and Figure 2, results/models.csv
python experiments.py final             # threshold used by run.py, results/final_candidates.csv
```

## Results

| | F1 | Accuracy |
| --- | --- | --- |
| 5-fold CV, without `HAREHAB1` (model analysed in the report) | 0.431 ± 0.006 | 0.874 |
| 5-fold CV, with `HAREHAB1` (submitted model) | 0.439 ± 0.006 | |
| AIcrowd, without `HAREHAB1` (submission 338157) | 0.443 | 0.878 |
| AIcrowd, with `HAREHAB1` (submission 338161, final) | 0.451 | 0.877 |

Final model: L2-regularized logistic regression trained with Newton's method,
lambda = 1e-4, decision threshold 0.195 (chosen on out-of-fold predictions by
`python experiments.py final`). Full tables: `results/models.csv` (model comparison),
`results/ablation.csv` (preprocessing ablation) and `results/final_candidates.csv`.

`HAREHAB1` (cardiac rehabilitation) is only asked after a heart attack, so it leaks
the label and stays dropped in `decisions/column_decisions.csv`. For the competition
submission only, `run.py` adds it back with `modeling.with_leaky_column`; the report
explains why.

## Data analysis and preprocessing

How the raw data (321 columns) becomes the model input:

| Step                    | Where                                | What it does                                                                                                                                                                         |
| ----------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1. Measure the columns  | `column_analysis.py`                 | Missing rates, special answer codes, positive rate per answer, duplicate and determined columns, label leakage                                                                       |
| 2. Decide per column    | `notebooks/01_column_analysis.ipynb` | Shows the evidence (cells 2-8) and decides for every column whether it is dropped, categorical or continuous, and which codes mean "missing" or "none" (cell 9)                      |
| 3. Record the decisions | `decisions/column_decisions.csv`     | One row per column: action, codes and the reason, with the notebook cell that supports it                                                                                            |
| 4. Apply the decisions  | `preprocessing.py`                   | `fit_preprocess` learns categories, medians and scaling from the training data only, `transform` applies them to any data, `print_report` lists what was done to each column and why |

Each preprocessing step can be switched off with the `options` argument of `fit_preprocess`, which is how the ablation study is run.

Current status:

- 69 columns are dropped, 188 are one-hot encoded and 64 are treated as continuous, giving 1249 features.
- Columns coded per day/week/month or as dates are dropped; CDC's cleaned per-day versions are kept instead.
- Codes were checked against the [BRFSS 2015 codebook](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf) and the [calculated variables document](https://www.cdc.gov/brfss/annual_data/2015/pdf/2015_Calculated_Variables_Version4_08_10_17-508c.pdf).

## Rules

**Project rules (from the course)**

- Allowed libraries: Python standard library, NumPy, and matplotlib/seaborn for plots only.
  No pandas, scikit-learn, PyTorch or other external libraries; no external code or data.
- `implementations.py`: all functions return `(w, loss)` with the last `w`; vectors are 1D arrays;
  MSE uses the 1/2 factor; SGD uses batch size 1; the loss of the regularized methods excludes the penalty term;
  `numpy.linalg.lstsq` is not used.
- `run.py` must produce exactly the CSV of our best AIcrowd submission.
- Report: at most 2 pages in LaTeX (references may go on a third page), no appendix.
- At most 5 AIcrowd submissions per day for the whole team.
- Deadline: Thursday Oct 29, 2026, 16:00.

**Team workflow**

- One branch per component; merge into `main` directly as soon as it works (no pull requests).
- Branch names: `feat/`, `fix/`, `chore/`, `docs/`, `refactor/` + short description (e.g. `feat/column-analysis`).
- Commit messages start with a verb: Add, Fix, Update, Remove, Refactor.
- Format Python files with `black` (version 22.6.0) before committing.
- `main` must always run: run the code once before merging.
- Never commit the data (`data/` is ignored); check `git status` before every commit.
- All randomness uses a fixed seed passed explicitly, so results are reproducible.
- Every AIcrowd submission is saved in `submissions/` as `YYYY-MM-DD_<commit>.csv`, and its score is noted.
- Run the course grading tests before merging changes to `implementations.py`.
