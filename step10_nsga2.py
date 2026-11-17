# STEP 10 (optional): Proper multi-objective search with NSGA-II (from the pymoo library).
# Two objectives at once:  MAXIMISE CV accuracy  and  MINIMISE the number of support vectors.
# NSGA-II returns a whole Pareto front in one run. We plot it next to the front from step 7
# (which was built by simply collecting every point PSO and GA happened to visit).
# Run step 7 first if you want the comparison:  python step7_pareto.py
# Run it with:  python step10_nsga2.py

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
except ImportError:
    sys.exit("This step needs the pymoo library. Install it with:\n    pip install pymoo\nand run again.")

from data import load_data
from fitness import Fitness, BOUNDS
from helpers import make_output_dirs, RESULTS_DIR, FIGURES_DIR

# ------------------------- SETTINGS -------------------------
POP_SIZE = 20
N_GENERATIONS = 15       # about POP_SIZE * N_GENERATIONS = 300 SVM evaluations
SEED = 1
# ------------------------------------------------------------

make_output_dirs()
X_train, X_test, y_train, y_test = load_data()
fitness = Fitness(X_train, y_train, use_my_svm=False)


class SvmProblem(Problem):
    """pymoo MINIMISES every objective, so accuracy becomes -accuracy."""

    def __init__(self):
        super().__init__(n_var=2, n_obj=2, xl=BOUNDS[:, 0], xu=BOUNDS[:, 1])

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for log_C, log_gamma in X:
            acc, n_sv = fitness.lookup(log_C, log_gamma)      # plain CV score and support-vector count
            F.append([-acc * 100, n_sv])
        out["F"] = np.array(F)


result = minimize(SvmProblem(), NSGA2(pop_size=POP_SIZE), ("n_gen", N_GENERATIONS), seed=SEED, verbose=False)

order = np.argsort(result.F[:, 1])
front = np.column_stack([result.F[order, 1], -result.F[order, 0], result.X[order]])    # n_sv, acc %, log_C, log_gamma
print(f"NSGA-II scored {len(fitness.cache)} different (C, gamma) pairs and found {len(front)} Pareto-optimal ones:\n")
print(f"{'support vectors':>15} {'CV acc %':>9} {'C':>9} {'gamma':>9}")
for s, a, lc, lg in front:
    print(f"{s:>15.1f} {a:>9.2f} {10**lc:>9.3f} {10**lg:>9.5f}")
pd.DataFrame(front, columns=["avg_support_vectors", "cv_acc_percent", "log_C", "log_gamma"]).to_csv(
    os.path.join(RESULTS_DIR, "pareto_nsga2.csv"), index=False)

# ---------- compare with step 7 ----------
step7_file = os.path.join(RESULTS_DIR, "pareto_step7.csv")
step7 = pd.read_csv(step7_file) if os.path.exists(step7_file) else None
if step7 is None:
    print("\n(results/pareto_step7.csv not found: run step7_pareto.py to add the comparison.)")
else:
    print(f"\nStep 7 front: {len(step7)} points | NSGA-II front: {len(front)} points")
    # For each step-7 point, is it beaten by some NSGA-II point?
    s_a, s_n = step7["cv_acc_percent"].to_numpy(), step7["avg_support_vectors"].to_numpy()
    beaten = sum(bool(np.any((front[:, 1] >= a) & (front[:, 0] <= n) & ((front[:, 1] > a) | (front[:, 0] < n))))
                 for a, n in zip(s_a, s_n))
    print(f"Step-7 points dominated by an NSGA-II point: {beaten} of {len(step7)}")
    print(f"Best accuracy:  step 7 = {s_a.max():.2f}%   NSGA-II = {front[:, 1].max():.2f}%")
    print(f"Fewest support vectors on the front:  step 7 = {s_n.min():.1f}   NSGA-II = {front[:, 0].min():.1f}")

all_pts = np.array([(n, a * 100) for a, n in fitness.cache.values()])
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(all_pts[:, 0], all_pts[:, 1], s=14, color="#c3c2b7", label="Every setting NSGA-II tried")
if step7 is not None:
    ax.plot(step7["avg_support_vectors"], step7["cv_acc_percent"], color="#2a78d6", linewidth=2, marker="o",
            markersize=5, label="Step 7 front (PSO + GA points)")
ax.plot(front[:, 0], front[:, 1], color="#eb6834", linewidth=2, marker="^", markersize=6, label="NSGA-II front")
ax.set_xlabel("Average number of support vectors (smaller = simpler model)")
ax.set_ylabel("CV accuracy (%)")
top = max(all_pts[:, 1].max(), 0)
ax.set_ylim(max(0, top - 5), top + 0.5)
ax.set_title("Pareto fronts: NSGA-II vs. step 7", loc="left")
ax.grid(color="#e5e4e0")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "pareto_nsga2.png"), dpi=150)
print("\nSaved: figures/pareto_nsga2.png, results/pareto_nsga2.csv")
