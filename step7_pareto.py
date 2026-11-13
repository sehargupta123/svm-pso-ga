# STEP 7 (optional): Multi-objective view - accuracy vs. model size.
# A model with fewer support vectors is smaller and faster to use.
# We look at every (C, gamma) tried and keep the "Pareto-optimal" ones:
# points where you cannot get higher accuracy without using more support vectors.
# Run it with:  python step7_pareto.py

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_data
from fitness import Fitness, BOUNDS
from pso import pso
from ga import ga

os.makedirs("results", exist_ok=True)   # create the output folders if they are missing
os.makedirs("figures", exist_ok=True)

X_train, X_test, y_train, y_test = load_data()
fitness = Fitness(X_train, y_train, use_my_svm=False)

# Collect lots of evaluated points by running PSO and GA a few times.
for seed in range(5):
    pso(fitness.evaluate, BOUNDS, seed=seed)
    ga(fitness.evaluate, BOUNDS, seed=seed)

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

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(n_sv[~pareto], acc[~pareto], s=20, color="#c3c2b7", label="Other settings tried")
ax.plot(front[:, 1], front[:, 0], color="#2a78d6", linewidth=2, marker="o", markersize=6,
        label="Pareto front (best trade-offs)")
ax.set_xlabel("Average number of support vectors (smaller = simpler model)")
ax.set_ylabel("CV accuracy (%)")
ax.set_ylim(94, 99)
ax.set_xlim(20, 260)
ax.set_title("Accuracy vs. model size: you can't improve one without the other", loc="left")
ax.grid(color="#e5e4e0")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig("figures/pareto_front.png", dpi=150)
print("\nSaved: figures/pareto_front.png")