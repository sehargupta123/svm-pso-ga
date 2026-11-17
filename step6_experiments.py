# STEP 6: The full experiment.
# Runs the stochastic methods (Random, PSO, GA, SA, and Bayesian/TPE if optuna is installed)
# N_SEEDS times each (different random seeds) and grid search once.
# Saves the results to results/ and the plots to figures/.
#
# USE_MY_SVM = False uses scikit-learn's SVC inside the fitness function. Step 3 showed it
# gives the same accuracy as our SVM, and it makes this script take a few minutes.
# Set it to True to use our own SVM everywhere (much slower: hours with 30 seeds).
# QUICK = True runs 3 seeds and a 10x10 landscape so you can check the script in about a minute.
# Run it with:  python step6_experiments.py

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # save plots to files instead of opening windows
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

import os
import fitness as fitness_module
from data import load_data
from fitness import Fitness, BOUNDS
from tuners import make_methods, BUDGET
from helpers import test_report, make_output_dirs, RESULTS_DIR, FIGURES_DIR

# ------------------------- SETTINGS -------------------------
USE_MY_SVM = False
N_SEEDS = 30           # how many different random seeds each stochastic method is run with
QUICK = False          # True = 3 seeds + 10x10 landscape (fast check)
LANDSCAPE_N = 30       # the accuracy map is LANDSCAPE_N x LANDSCAPE_N points
TARGET_GAP = 0.001     # "reached the target" = within 0.1 percentage points of the best score found
# ------------------------------------------------------------

if QUICK:
    N_SEEDS, LANDSCAPE_N = 3, 10
SEEDS = list(range(N_SEEDS))
NEVER = BUDGET + 1     # "never reached the target" counts as 101 trials in the significance tests

COLORS = {"Grid search": "#2a78d6", "Random search": "#eb6834", "PSO": "#1baf7a", "GA": "#eda100",
          "SA": "#8e5bd0", "Bayesian (TPE)": "#d6336c"}

make_output_dirs()

X_train, X_test, y_train, y_test = load_data()
fitness = Fitness(X_train, y_train, use_my_svm=USE_MY_SVM)
methods = make_methods(include_bayesian=True)
print(f"Methods: {', '.join(methods)}")
if "Bayesian (TPE)" not in methods:
    print("(optuna is not installed, so the Bayesian baseline is skipped:  pip install optuna)")
print(f"{N_SEEDS} seeds per stochastic method{' (QUICK mode)' if QUICK else ''}, scoring = {fitness_module.SCORING}\n")

# ---------- 1. Run everything ----------
rows, curves, paths = [], {m: [] for m in methods}, {}
for name, run in methods.items():
    seeds = [0] if name == "Grid search" else SEEDS     # grid search has no randomness
    for seed in seeds:
        fitness.reset_history()
        best_x, _ = run(fitness.evaluate, seed)
        best_so_far = np.maximum.accumulate(fitness.history)  # best score seen up to each trial
        curves[name].append(best_so_far)
        if seed == 0:
            paths[name] = np.array(fitness.points)
        rep = test_report(X_train, y_train, X_test, y_test, *best_x, use_my_svm=USE_MY_SVM)
        rows.append({
            "method": name, "seed": seed,
            "best_C": 10 ** best_x[0], "best_gamma": 10 ** best_x[1],
            "cv_score": best_so_far[-1],     # the score the optimiser saw (includes the tie-break penalty)
            "cv_base_score": fitness.lookup(*best_x)[0],   # same point, without the tie-break penalty
            "test_acc": rep["acc"], "tp": rep["tp"], "fn": rep["fn"], "fp": rep["fp"], "tn": rep["tn"],
            "malignant_recall": rep["recall"],
            "repeated_trials": fitness.n_repeated(),
            "curve": best_so_far,
        })
    print(f"Finished {name}")

# "Target" = within TARGET_GAP of the best CV score any method found.
target = max(r["cv_score"] for r in rows) - TARGET_GAP
for r in rows:
    reached = np.where(r["curve"] >= target)[0]
    r["trials_to_target"] = int(reached[0]) + 1 if len(reached) else np.nan
    del r["curve"]

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS_DIR, "all_runs.csv"), index=False)

summary = df.groupby("method", sort=False).agg(
    runs=("seed", "count"),
    cv_score_mean=("cv_score", "mean"), cv_score_std=("cv_score", "std"),
    cv_base_score_mean=("cv_base_score", "mean"),
    test_acc_mean=("test_acc", "mean"),
    malignant_recall_mean=("malignant_recall", "mean"),
    malignant_missed_mean=("fn", "mean"),
    false_alarms_mean=("fp", "mean"),
    repeated_trials_mean=("repeated_trials", "mean"),
    trials_to_target_mean=("trials_to_target", "mean"),
    trials_to_target_median=("trials_to_target", "median"),
    runs_reaching_target=("trials_to_target", "count"),
).round(4)
summary.to_csv(os.path.join(RESULTS_DIR, "summary.csv"))
print(f"\nTarget CV score: {target:.4f}\n")
print(summary.to_string())

# ---------- 1b. Significance tests (Mann-Whitney U on trials-to-target) ----------
def trials(method):
    return df.loc[df.method == method, "trials_to_target"].fillna(NEVER).to_numpy()

pairs = [("PSO", "Random search"), ("GA", "Random search"), ("PSO", "GA"), ("SA", "Random search")]
if "Bayesian (TPE)" in methods:
    pairs.append(("Bayesian (TPE)", "Random search"))
sig_rows = []
for a, b in pairs:
    x, y = trials(a), trials(b)
    try:
        u, p = mannwhitneyu(x, y, alternative="two-sided")
    except ValueError:              # happens when every value is identical
        u, p = np.nan, 1.0
    sig_rows.append({"comparison": f"{a} vs {b}", "n_a": len(x), "n_b": len(y),
                     "median_trials_a": np.median(x), "median_trials_b": np.median(y),
                     "U": u, "p_value": p, "significant_at_0.05": bool(p < 0.05)})
sig = pd.DataFrame(sig_rows)
sig.to_csv(os.path.join(RESULTS_DIR, "significance.csv"), index=False)
print(f"\nMann-Whitney U test on trials-to-target ('never reached' counted as {NEVER}); two-sided:\n")
print(sig.round(4).to_string(index=False))
print("(p < 0.05 means the difference between the two methods is unlikely to be luck.)")

# ---------- 2. Convergence plot ----------
fig, ax = plt.subplots(figsize=(8, 4.5))
lowest = 100.0
for name, cs in curves.items():
    mean_curve = np.mean(cs, axis=0) * 100
    lowest = min(lowest, mean_curve[0])
    trials_axis = np.arange(1, len(mean_curve) + 1)
    ax.plot(trials_axis, mean_curve, color=COLORS[name], linewidth=2, label=name)
ax.set_xlabel("Trial (number of SVMs evaluated)")
ax.set_ylabel(f"Best CV {fitness_module.SCORING} so far (%)")
ax.set_title("Best score found vs. number of trials (average over runs)", loc="left")
ax.set_ylim(max(0, np.floor(lowest) - 1), 100)
ax.set_xlim(1, BUDGET)
ax.grid(axis="y", color="#e5e4e0")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "convergence.png"), dpi=150)
plt.close(fig)

# ---------- 3. Box plot: trials needed to reach the target ----------
stoch = [m for m in methods if m != "Grid search"]
fig, ax = plt.subplots(figsize=(1.6 * len(stoch) + 2, 4.8))
data_box = [df.loc[df.method == m, "trials_to_target"].dropna() for m in stoch]
bp = ax.boxplot([d if len(d) else [np.nan] for d in data_box], patch_artist=True, widths=0.5)
labels = [f"{m}\n(reached in {len(d)}/{N_SEEDS} runs)" for m, d in zip(stoch, data_box)]
ax.set_xticks(range(1, len(stoch) + 1), labels, fontsize=8)
for patch, m in zip(bp["boxes"], stoch):
    patch.set_facecolor(COLORS[m]); patch.set_alpha(0.35); patch.set_edgecolor(COLORS[m])
for med in bp["medians"]:
    med.set_color("#3d3d3a"); med.set_linewidth(2)
grid_trials = df.loc[df.method == "Grid search", "trials_to_target"].iloc[0]
if np.isnan(grid_trials):
    ax.text(0.01, 0.98, "Grid search: never reached the target", transform=ax.transAxes,
            va="top", fontsize=9, color="#3d3d3a")
else:
    ax.axhline(grid_trials, color=COLORS["Grid search"], linestyle="--", linewidth=1.5)
    ax.annotate(f"Grid search: {grid_trials:.0f} trials", (0.55, grid_trials), xytext=(0, 5),
                textcoords="offset points", fontsize=9, color="#3d3d3a")
ax.set_ylabel("Trials needed to reach target (lower is better)")
ax.set_title(f"Trials to reach {target*100:.2f}% CV score ({N_SEEDS} runs each)", loc="left")
ax.grid(axis="y", color="#e5e4e0")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "trials_to_target.png"), dpi=150)
plt.close(fig)

# ---------- 4. Search-space map with where PSO, GA and SA went ----------
log_C_axis = np.linspace(*BOUNDS[0], LANDSCAPE_N)
log_g_axis = np.linspace(*BOUNDS[1], LANDSCAPE_N)
landscape = Fitness(X_train, y_train, use_my_svm=False)
acc = np.array([[landscape.evaluate(c, g) for c in log_C_axis] for g in log_g_axis]) * 100

fig, ax = plt.subplots(figsize=(7.5, 5.5))
# imshow: rows = log_gamma (bottom to top), columns = log_C, matching `extent` below.
im = ax.imshow(acc, origin="lower", aspect="auto", cmap="Blues", vmin=max(0, np.floor(acc.min() / 10) * 10),
               vmax=100, extent=[*BOUNDS[0], *BOUNDS[1]])
fig.colorbar(im, ax=ax, label=f"CV {fitness_module.SCORING} (%)")
for name, marker in (("PSO", "o"), ("GA", "^"), ("SA", "s")):
    p = paths[name]
    ax.scatter(p[:, 0], p[:, 1], s=36, marker=marker, color=COLORS[name],
               edgecolor="white", linewidth=0.8, label=f"{name} trials (seed 0)")
ax.set_xlabel("log10(C)")
ax.set_ylabel("log10(gamma)")
ax.set_title("Where PSO, GA and SA searched, over the score landscape", loc="left")
ax.legend(loc="lower left", frameon=True)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "search_space.png"), dpi=150)
plt.close(fig)

print("\nSaved: results/all_runs.csv, results/summary.csv, results/significance.csv")
print("Saved: figures/convergence.png, figures/trials_to_target.png, figures/search_space.png")
