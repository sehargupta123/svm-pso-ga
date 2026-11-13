# STEP 6: The full experiment.
# Runs PSO, GA and random search 10 times each (different random seeds) and grid search once.
# Saves the results to results/ and the plots to figures/.
#
# USE_MY_SVM = False uses scikit-learn's SVC inside the fitness function. Step 3 showed it
# gives the same accuracy as our SVM, and it makes this script take about 1 minute.
# Set it to True to use our own SVM everywhere (takes about 30-40 minutes).
# Run it with:  python step6_experiments.py

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # save plots to files instead of opening windows
import matplotlib.pyplot as plt

from data import load_data
from fitness import Fitness, BOUNDS
from pso import pso
from ga import ga
from baselines import grid_search, random_search
from helpers import test_accuracy

USE_MY_SVM = False
SEEDS = range(10)
TARGET_GAP = 0.001   # "reached the target" = within 0.1 percentage points of the best score found
COLORS = {"Grid search": "#2a78d6", "Random search": "#eb6834", "PSO": "#1baf7a", "GA": "#eda100"}

os.makedirs("results", exist_ok=True)   # create the output folders if they are missing
os.makedirs("figures", exist_ok=True)

X_train, X_test, y_train, y_test = load_data()
fitness = Fitness(X_train, y_train, use_my_svm=USE_MY_SVM)

methods = {
    "Grid search":   lambda f, s: grid_search(f, BOUNDS),
    "Random search": lambda f, s: random_search(f, BOUNDS, seed=s),
    "PSO":           lambda f, s: pso(f, BOUNDS, n_particles=10, n_iters=10, seed=s),
    "GA":            lambda f, s: ga(f, BOUNDS, pop_size=10, max_evals=100, seed=s),
}

# ---------- 1. Run everything ----------
rows, curves, paths = [], {m: [] for m in methods}, {}
for name, run in methods.items():
    seeds = [0] if name == "Grid search" else SEEDS     # grid search has no randomness
    for seed in seeds:
        fitness.reset_history()
        best_x, best_cv = run(fitness.evaluate, seed)
        best_so_far = np.maximum.accumulate(fitness.history)  # best score seen up to each trial
        curves[name].append(best_so_far)
        if seed == 0:
            paths[name] = np.array(fitness.points)
        rows.append({
            "method": name, "seed": seed,
            "best_C": 10 ** best_x[0], "best_gamma": 10 ** best_x[1],
            "cv_acc": best_cv,
            "test_acc": test_accuracy(X_train, y_train, X_test, y_test, *best_x),
            "curve": best_so_far,
        })
    print(f"Finished {name}")

# "Target" = within TARGET_GAP of the best CV accuracy any method found.
target = max(r["cv_acc"] for r in rows) - TARGET_GAP
for r in rows:
    reached = np.where(r["curve"] >= target)[0]
    r["trials_to_target"] = int(reached[0]) + 1 if len(reached) else np.nan
    del r["curve"]

df = pd.DataFrame(rows)
df.to_csv("results/all_runs.csv", index=False)

summary = df.groupby("method", sort=False).agg(
    runs=("seed", "count"),
    cv_acc_mean=("cv_acc", "mean"), cv_acc_std=("cv_acc", "std"),
    test_acc_mean=("test_acc", "mean"),
    trials_to_target_mean=("trials_to_target", "mean"),
    trials_to_target_median=("trials_to_target", "median"),
    runs_reaching_target=("trials_to_target", "count"),
).round(4)
summary.to_csv("results/summary.csv")
print(f"\nTarget CV accuracy: {target:.4f}\n")
print(summary.to_string())

# ---------- 2. Convergence plot ----------
fig, ax = plt.subplots(figsize=(8, 4.5))
for name, cs in curves.items():
    mean_curve = np.mean(cs, axis=0) * 100
    trials = np.arange(1, len(mean_curve) + 1)
    ax.plot(trials, mean_curve, color=COLORS[name], linewidth=2, label=name)
ax.set_xlabel("Trial (number of SVMs evaluated)")
ax.set_ylabel("Best CV accuracy so far (%)")
ax.set_title("Best accuracy found vs. number of trials (average over runs)", loc="left")
ax.set_ylim(90, 99)
ax.set_xlim(1, 100)
ax.grid(axis="y", color="#e5e4e0")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig("figures/convergence.png", dpi=150)
plt.close(fig)

# ---------- 3. Box plot: trials needed to reach the target ----------
stoch = ["Random search", "PSO", "GA"]
fig, ax = plt.subplots(figsize=(7, 4.5))
data = [df.loc[df.method == m, "trials_to_target"].dropna() for m in stoch]
bp = ax.boxplot(data, patch_artist=True, widths=0.5)
labels = [f"{m}\n(reached in {len(d)}/10 runs)" for m, d in zip(stoch, data)]
ax.set_xticks(range(1, len(stoch) + 1), labels)
for patch, m in zip(bp["boxes"], stoch):
    patch.set_facecolor(COLORS[m]); patch.set_alpha(0.35); patch.set_edgecolor(COLORS[m])
for med in bp["medians"]:
    med.set_color("#3d3d3a"); med.set_linewidth(2)
grid_trials = df.loc[df.method == "Grid search", "trials_to_target"].iloc[0]
ax.axhline(grid_trials, color=COLORS["Grid search"], linestyle="--", linewidth=1.5)
ax.annotate(f"Grid search: {grid_trials:.0f} trials", (0.55, grid_trials), xytext=(0, 5),
            textcoords="offset points", fontsize=9, color="#3d3d3a")
ax.set_ylabel("Trials needed to reach target (lower is better)")
ax.set_title(f"Trials to reach {target*100:.2f}% CV accuracy (10 runs each)", loc="left")
ax.grid(axis="y", color="#e5e4e0")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig("figures/trials_to_target.png", dpi=150)
plt.close(fig)

# ---------- 4. Search-space map with where PSO and GA went ----------
n = 30
log_C_axis = np.linspace(*BOUNDS[0], n)
log_g_axis = np.linspace(*BOUNDS[1], n)
landscape = Fitness(X_train, y_train, use_my_svm=False)
acc = np.array([[landscape.evaluate(c, g) for c in log_C_axis] for g in log_g_axis]) * 100

fig, ax = plt.subplots(figsize=(7.5, 5.5))
im = ax.imshow(acc, origin="lower", aspect="auto", cmap="Blues", vmin=60, vmax=100,
               extent=[*BOUNDS[0], *BOUNDS[1]])
fig.colorbar(im, ax=ax, label="CV accuracy (%)")
for name, marker in (("PSO", "o"), ("GA", "^")):
    p = paths[name]
    ax.scatter(p[:, 0], p[:, 1], s=36, marker=marker, color=COLORS[name],
               edgecolor="white", linewidth=0.8, label=f"{name} trials (run 1)")
ax.set_xlabel("log10(C)")
ax.set_ylabel("log10(gamma)")
ax.set_title("Where PSO and GA searched, over the accuracy landscape", loc="left")
ax.legend(loc="lower left", frameon=True)
fig.tight_layout()
fig.savefig("figures/search_space.png", dpi=150)
plt.close(fig)

print("\nSaved: results/all_runs.csv, results/summary.csv")
print("Saved: figures/convergence.png, figures/trials_to_target.png, figures/search_space.png")