# STEP 7 (optional): Multi-objective view - accuracy vs. model size.
# A model with fewer support vectors is smaller and faster to use.
# We look at every (C, gamma) tried and keep the "Pareto-optimal" ones:
# points where you cannot get higher accuracy without using more support vectors.
# The front is saved to results/pareto_step7.csv so step 10 (NSGA-II) can compare against it.
# Run it with:  python step7_pareto.py

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_data
from fitness import Fitness, BOUNDS
from pso import pso
from ga import ga
from helpers import make_output_dirs, RESULTS_DIR, FIGURES_DIR

make_output_dirs()

X_train, X_test, y_train, y_test = load_data()
fitness = Fitness(X_train, y_train, use_my_svm=False)

# Collect lots of evaluated points by running PSO and GA a few times.
for seed in range(5):
    pso(fitness.evaluate, BOUNDS, seed=[seed, 2])
    ga(fitness.evaluate, BOUNDS, seed=[seed, 3])

# fitness.cache holds the plain CV score (no tie-break penalty) and the support-vector count.
points = np.array([(acc * 100, n_sv, lc, lg) for (lc, lg), (acc, n_sv) in fitness.cache.items()])
acc, n_sv = points[:, 0], points[:, 1]

# A point is Pareto-optimal if no other point has BOTH higher-or-equal accuracy
# AND fewer-or-equal support vectors (and is strictly better in at least one).
pareto = []
for i in range(len(points)):
    dominated = np.any((acc >= acc[i]) & (n_sv <= n_sv[i]) & ((acc > acc[i]) | (n_sv < n_sv[i])))
    pareto.append(not dominated)
pareto = np.array(pareto)
front = points[pareto][np.argsort(points[pareto][:, 1])]

print(f"{len(points)} different (C, gamma) pairs were evaluated; {pareto.sum()} are Pareto-optimal:\n")
print(f"{'support vectors':>15} {'CV acc %':>9} {'C':>9} {'gamma':>9}")
for a, s, lc, lg in front:
    print(f"{s:>15.1f} {a:>9.2f} {10**lc:>9.3f} {10**lg:>9.5f}")

pd.DataFrame({"avg_support_vectors": front[:, 1], "cv_acc_percent": front[:, 0],
              "log_C": front[:, 2], "log_gamma": front[:, 3]}).to_csv(
    os.path.join(RESULTS_DIR, "pareto_step7.csv"), index=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(n_sv[~pareto], acc[~pareto], s=20, color="#c3c2b7", label="Other settings tried")
ax.plot(front[:, 1], front[:, 0], color="#2a78d6", linewidth=2, marker="o", markersize=6,
        label="Pareto front (best trade-offs)")
ax.set_xlabel("Average number of support vectors (smaller = simpler model)")
ax.set_ylabel("CV accuracy (%)")
ax.set_ylim(max(0, acc.max() - 5), acc.max() + 0.5)           # zoom on the top 5 percentage points
ax.set_xlim(max(0, n_sv.min() - 10), n_sv.max() + 10)
ax.set_title("Accuracy vs. model size: you can't improve one without the other", loc="left")
ax.grid(color="#e5e4e0")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "pareto_front.png"), dpi=150)
print("\nSaved: figures/pareto_front.png, results/pareto_step7.csv")
