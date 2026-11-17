# STEP 8 (optional): Sensitivity study - how much do the PSO and GA settings matter?
#   PSO inertia w          in {0.4, 0.7, 0.9}   (c1 = c2 = 1.5)
#   PSO c1 = c2            in {1.0, 1.5, 2.0}   (w = 0.7)
#   GA mutation_rate       in {0.05, 0.15, 0.3}
# Each setting is run with N_SEEDS different seeds. Uses scikit-learn's SVC for speed.
# Saves results/sensitivity.csv and figures/sensitivity.png.
# QUICK = True uses 2 seeds so it finishes fast.
# Run it with:  python step8_sensitivity.py

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

# ------------------------- SETTINGS -------------------------
QUICK = False
N_SEEDS = 10
TARGET_GAP = 0.001
# ------------------------------------------------------------
if QUICK:
    N_SEEDS = 2

make_output_dirs()
X_train, X_test, y_train, y_test = load_data()
fitness = Fitness(X_train, y_train, use_my_svm=False)       # SVC: fast, same answers as our SVM

# (group, setting name, function(score_fn, seed))
configs = []
for w in (0.4, 0.7, 0.9):
    configs.append(("PSO inertia w", f"w={w}",
                    lambda f, s, w=w: pso(f, BOUNDS, n_particles=10, n_iters=10, w=w, c1=1.5, c2=1.5, seed=[s, 2])))
for c in (1.0, 1.5, 2.0):
    configs.append(("PSO c1 = c2", f"c1=c2={c}",
                    lambda f, s, c=c: pso(f, BOUNDS, n_particles=10, n_iters=10, w=0.7, c1=c, c2=c, seed=[s, 2])))
for m in (0.05, 0.15, 0.3):
    configs.append(("GA mutation rate", f"mutation={m}",
                    lambda f, s, m=m: ga(f, BOUNDS, pop_size=10, max_evals=100, mutation_rate=m, seed=[s, 3])))

# ---------- Run everything ----------
runs = []       # one entry per (config, seed)
for group, setting, run in configs:
    for seed in range(N_SEEDS):
        fitness.reset_history()
        best_x, _ = run(fitness.evaluate, seed)
        curve = np.maximum.accumulate(fitness.history)
        runs.append({"group": group, "setting": setting, "seed": seed,
                     "cv_acc": fitness.lookup(*best_x)[0], "curve": curve})
    print(f"Finished {group}: {setting}")

# Target = within TARGET_GAP of the best score seen in the whole study.
target = max(r["curve"][-1] for r in runs) - TARGET_GAP
for r in runs:
    reached = np.where(r["curve"] >= target)[0]
    r["trials_to_target"] = int(reached[0]) + 1 if len(reached) else np.nan

df = pd.DataFrame([{k: v for k, v in r.items() if k != "curve"} for r in runs])
table = df.groupby(["group", "setting"], sort=False).agg(
    cv_acc_mean=("cv_acc", "mean"), cv_acc_std=("cv_acc", "std"),
    trials_to_target_mean=("trials_to_target", "mean"),
    runs_reaching_target=("trials_to_target", "count"),
).round(4)
table["seeds"] = N_SEEDS
table.to_csv(os.path.join(RESULTS_DIR, "sensitivity.csv"))
print(f"\nTarget CV accuracy: {target:.4f}  (seeds per setting: {N_SEEDS})\n")
print(table.to_string())

# ---------- One figure, three panels: average convergence for each setting ----------
groups = list(dict.fromkeys(g for g, _, _ in configs))
shades = ["#9ecae1", "#4292c6", "#08519c"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, group in zip(axes, groups):
    for shade, (g, setting, _) in zip(shades, [c for c in configs if c[0] == group]):
        curves = np.array([r["curve"] for r in runs if r["setting"] == setting and r["group"] == group])
        ax.plot(np.arange(1, curves.shape[1] + 1), curves.mean(axis=0) * 100, color=shade, linewidth=2, label=setting)
    ax.set_title(group, loc="left")
    ax.set_xlabel("Trial")
    ax.grid(axis="y", color="#e5e4e0")
    ax.legend(frameon=False, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("Best CV accuracy so far (%)")
low = min(np.mean([r["curve"] for r in runs], axis=0)[0] * 100, 100) - 0.5
axes[0].set_ylim(max(0, low), 100)
fig.suptitle(f"Sensitivity to PSO and GA settings (average of {N_SEEDS} seeds)", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "sensitivity.png"), dpi=150)
print("\nSaved: results/sensitivity.csv, figures/sensitivity.png")
