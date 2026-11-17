# STEP 9 (optional): Let the GA also choose the KERNEL.
# The search space is now mixed (some numbers, some categories):
#   kernel in {linear, poly, rbf}     poly degree in {2, 3, 4}     log10(C)     log10(gamma)
# Trick: every gene is a plain number. The two category genes are in [0, 3) and we round DOWN:
#   gene 0.0-0.99 -> linear, 1.0-1.99 -> poly, 2.0-2.99 -> rbf   (same idea for the degree)
# Uses scikit-learn's SVC (our own SVM only has the RBF kernel).
# QUICK = True uses a smaller budget and 2 seeds so it finishes fast.
# Run it with:  python step9_kernel_search.py

import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import load_data
from fitness import BOUNDS
from ga import ga
from helpers import test_accuracy_model, make_output_dirs, RESULTS_DIR
import os

# ------------------------- SETTINGS -------------------------
QUICK = False
N_SEEDS = 5
MAX_EVALS = 150          # the search space is bigger, so the GA gets a bigger budget than 100
# ------------------------------------------------------------
if QUICK:
    N_SEEDS, MAX_EVALS = 2, 40

warnings.filterwarnings("ignore", category=ConvergenceWarning)    # poly/linear with huge C may hit max_iter

KERNELS = ["linear", "poly", "rbf"]
DEGREES = [2, 3, 4]
# genes: [kernel, degree, log_C, log_gamma]
SPACE = np.array([[0.0, 3.0], [0.0, 3.0], BOUNDS[0], BOUNDS[1]])

X_train, X_test, y_train, y_test = load_data()
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X_train, y_train))
cache = {}               # decoded model -> CV accuracy


def decode(kernel_gene, degree_gene, log_C, log_gamma):
    kernel = KERNELS[min(int(kernel_gene), 2)]
    degree = DEGREES[min(int(degree_gene), 2)]
    return kernel, degree, 10 ** log_C, 10 ** log_gamma


def make_model(kernel, degree, C, gamma):
    return SVC(kernel=kernel, degree=degree, C=C, gamma=gamma, max_iter=50000)


def score(kernel_gene, degree_gene, log_C, log_gamma):
    kernel, degree, C, gamma = decode(kernel_gene, degree_gene, log_C, log_gamma)
    # Settings that a kernel ignores must not count as "different models".
    key = (kernel, degree if kernel == "poly" else 0, round(np.log10(C), 3),
           round(np.log10(gamma), 3) if kernel != "linear" else 0)
    if key not in cache:
        pipe = make_pipeline(StandardScaler(), make_model(kernel, degree, C, gamma))    # scaling inside each fold
        cache[key] = float(np.mean(cross_val_score(pipe, X_train, y_train, cv=folds, scoring="accuracy")))
    return cache[key]


make_output_dirs()
rows = []
for seed in range(N_SEEDS):
    best_x, best_cv = ga(score, SPACE, pop_size=15, max_evals=MAX_EVALS, mutation_rate=0.3, seed=[seed, 3])
    kernel, degree, C, gamma = decode(*best_x)
    test_acc = test_accuracy_model(X_train, y_train, X_test, y_test, make_model(kernel, degree, C, gamma))
    rows.append({"seed": seed, "kernel": kernel, "degree": degree if kernel == "poly" else "-",
                 "C": C, "gamma": gamma if kernel != "linear" else float("nan"),
                 "cv_acc": best_cv, "test_acc": test_acc})
    print(f"Seed {seed}: best = {kernel}" + (f" (degree {degree})" if kernel == "poly" else "") +
          f", C = {C:.3f}" + (f", gamma = {gamma:.5f}" if kernel != "linear" else "") +
          f" | CV acc {best_cv:.4f} | test acc {test_acc:.4f}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS_DIR, "kernel_search.csv"), index=False)

# Which kernel was best, looking at EVERY model scored (all seeds together)?
print(f"\n{len(cache)} different models were scored in total. Best CV accuracy found per kernel:")
for kernel in KERNELS:
    best = max((v for k, v in cache.items() if k[0] == kernel), default=float("nan"))
    count = sum(1 for k in cache if k[0] == kernel)
    print(f"  {kernel:<7} best CV acc = {best:.4f}   ({count} models tried)")
print(f"\nThe GA picked: {df['kernel'].value_counts().to_dict()} (one winner per seed)")
print("Saved: results/kernel_search.csv")
