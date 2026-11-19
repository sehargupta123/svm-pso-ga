# STEP 11: Objective functions and convergence plots, at both levels of the problem.
#
#   INNER level: training one SVM = maximising the dual D(alpha) (equivalently minimising the primal P).
#                We run our own SMO and plot P, D and the duality gap against the SMO iteration.
#   OUTER level: choosing (C, gamma) = maximising the cross-validated accuracy F(log C, log gamma).
#                We plot every tuner's trials and the F "landscape" as contour plots.
#
# The formulas themselves live in objectives.py (explicit NumPy, one docstring per formula).
# This script only runs the experiments and draws the figures. It does not change steps 1-10.
#
# QUICK = True : 3 inner pairs and a 12 x 12 contour grid (under a minute).
# QUICK = False: 9 inner pairs and a 35 x 35 contour grid (a few minutes).
# Run it with:  python step11_objective_convergence.py

import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # save plots to files instead of opening windows
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from data import load_data
from fitness import Fitness, BOUNDS
from tuners import make_methods
from helpers import make_output_dirs, RESULTS_DIR, FIGURES_DIR
import objectives as obj

# ------------------------- SETTINGS -------------------------
QUICK = False
GRID_N = 35            # the contour grid is GRID_N x GRID_N points of (log10 C, log10 gamma)
TARGET = 0.9792        # CV accuracy that counts as "reached the best level" (dotted line in the plots)
# ------------------------------------------------------------
if QUICK:
    GRID_N = 12

# Same colours as step 6. Order: PSO and GA first, then the baselines, SA last.
COLORS = {"Grid search": "#2a78d6", "Random search": "#eb6834", "PSO": "#1baf7a", "GA": "#eda100",
          "SA": "#8e5bd0", "Bayesian (TPE)": "#d6336c"}
ORDER = ["PSO", "GA", "Grid search", "Random search", "Bayesian (TPE)", "SA"]
MARKERS = {"PSO": "o", "GA": "s", "Grid search": "^", "Random search": "v", "Bayesian (TPE)": "D", "SA": "P"}
CONTRAST_COLORS = {"Underfit": "#555555", "Middle": "#8c8c8c", "Overfit": "#222222"}
P_COLOR, D_COLOR = "#b3261e", "#1a5fb4"      # primal / dual lines in the inner plots

make_output_dirs()
t_start = time.time()

# ---------- a) Data: same split as every other step; scale on the 455 training patients ----------
X_train, X_test, y_train, y_test = load_data()
X_scaled = StandardScaler().fit_transform(X_train)
y = y_train.astype(float)
print(f"Training set: {X_train.shape[0]} patients, {X_train.shape[1]} features"
      f"{'   (QUICK mode)' if QUICK else ''}\n")

# ---------- b) Run every tuner once with seed 0 (sklearn SVC inside the fitness: fast) ----------
# Step 3 showed scikit-learn's SVC gives exactly the same predictions/accuracy as our own SVM,
# so using it here only saves time. F values below are the plain CV accuracy (no tie-break penalty).
fitness = Fitness(X_train, y_train, use_my_svm=False)
methods = make_methods(include_bayesian=True)
outer_rows, best_points, trial_points = [], {}, {}
for name in ORDER:
    if name not in methods:
        print(f"(skipping {name}: optuna not installed)")
        continue
    fitness.reset_history()
    best_x, _ = methods[name](fitness.evaluate, 0)
    pts = np.array(fitness.points)
    F = np.array([fitness.lookup(*p)[0] for p in pts])
    best_so_far = np.maximum.accumulate(F)
    best_points[name], trial_points[name] = np.asarray(best_x), pts
    for t in range(len(F)):
        outer_rows.append({"method": name, "trial": t + 1, "log_C": pts[t, 0], "log_gamma": pts[t, 1],
                           "F": F[t], "best_so_far": best_so_far[t]})
    print(f"  tuner finished: {name:15s} {len(F)} trials, best F = {F.max():.4f}")
outer = pd.DataFrame(outer_rows)
outer.to_csv(os.path.join(RESULTS_DIR, "outer_trials.csv"), index=False)
names = [m for m in ORDER if m in best_points]

# ---------- inner pairs ----------
pairs = []      # (label, method column, C, gamma)
for name in names:
    lc, lg = best_points[name]
    pairs.append((name, name, 10 ** lc, 10 ** lg))
pairs += [("Underfit", "contrast", 0.01, 1e-4), ("Overfit", "contrast", 100.0, 1.0),
          ("Middle", "contrast", 1.0, 0.03)]
if QUICK:
    pairs = [p for p in pairs if p[0] in ("PSO", "GA", "Overfit")]

# ---------- c) SMO on the full training set for each pair ----------
print("\nINNER problem: SMO on all 455 training patients\n")
hist_rows, results = [], {}
table = []
for label, method, C, gamma in pairs:
    K = obj.rbf_kernel(X_scaled, X_scaled, gamma)
    t0 = time.time()
    r = obj.smo_trace(K, y, C)
    secs = time.time() - t0
    h = r["history"]
    # Cross-check against the cvxopt QP solution of the SAME dual problem.
    a_qp = obj.solve_qp(K, y, C)
    D_qp = obj.dual_objective(a_qp, obj.q_matrix(K, y))
    results[label] = {"r": r, "K": K, "C": C, "gamma": gamma}
    for k in range(len(h["iteration"])):
        hist_rows.append({"pair": label, "method": method, "C": C, "gamma": gamma,
                          "iteration": h["iteration"][k], "primal": h["primal"][k], "dual": h["dual"][k],
                          "gap": h["gap"][k], "rel_gap": h["rel_gap"][k]})
    table.append({"pair": label, "C": C, "gamma": gamma, "iterations": r["n_updates"],
                  "final P": h["primal"][-1], "final D": h["dual"][-1], "final gap": h["gap"][-1],
                  "cvxopt D": D_qp, "converged": r["converged"],
                  "max box viol": h["box_violation"].max(), "max eq viol": h["eq_violation"].max(),
                  "time (s)": secs})
pd.DataFrame(hist_rows).to_csv(os.path.join(RESULTS_DIR, "inner_convergence.csv"), index=False)
tab = pd.DataFrame(table)
with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.6g}".format):
    print(tab.to_string(index=False))
print("\n(SMO and cvxopt solve the same dual problem, so 'final D' and 'cvxopt D' should agree closely."
      "\n The gap is P - D >= 0 and shrinks to ~0; 'max box/eq viol' ~ 0 means alpha stayed feasible.)")

# Our NumPy cv_objective (SMO inside) against the sklearn-based fitness: same F at the same point.
print("\nCheck: F from objectives.cv_objective (our SMO) vs F from the sklearn fitness")
for lc, lg in (list(best_points.values())[:1] if QUICK else list(best_points.values())[:3]):
    f_ours = obj.cv_objective(lc, lg, X_train, y_train, fitness.folds)
    f_skl = fitness.lookup(lc, lg)[0]
    print(f"  log C = {lc:6.3f}, log gamma = {lg:6.3f}:  ours = {f_ours:.6f}   sklearn = {f_skl:.6f}")

# ---------- e) Contour data: F on a grid ----------
print(f"\nOUTER problem: evaluating F on a {GRID_N} x {GRID_N} grid ...")
lc_axis = np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], GRID_N)
lg_axis = np.linspace(BOUNDS[1, 0], BOUNDS[1, 1], GRID_N)
Fgrid = np.zeros((GRID_N, GRID_N))            # Fgrid[i_gamma, i_C]
surf_rows = []
for ic, lc in enumerate(lc_axis):
    for ig, lg in enumerate(lg_axis):
        Fgrid[ig, ic] = fitness.lookup(lc, lg)[0]
        surf_rows.append({"log_C": lc, "log_gamma": lg, "F": Fgrid[ig, ic]})
pd.DataFrame(surf_rows).to_csv(os.path.join(RESULTS_DIR, "cv_surface.csv"), index=False)
ig_max, ic_max = np.unravel_index(np.argmax(Fgrid), Fgrid.shape)
grid_max = (lc_axis[ic_max], lg_axis[ig_max], Fgrid[ig_max, ic_max])
print(f"  grid maximum: F = {grid_max[2]:.4f} at log C = {grid_max[0]:.2f}, log gamma = {grid_max[1]:.2f}")

print("\nOUTER problem: tuners (seed 0)\n")
summary = []
for name in names:
    d = outer[outer.method == name]
    first_best = int(d.trial.values[d.F.values.argmax()])
    summary.append({"method": name, "trials": len(d), "best F": d.F.max(), "first reached best at trial": first_best,
                    "best log C": best_points[name][0], "best log gamma": best_points[name][1]})
with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
    print(pd.DataFrame(summary).to_string(index=False))

# ================================== FIGURES ==================================
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(os.path.join(FIGURES_DIR, name), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved figures/" + name)


def pair_label(label, C, gamma):
    return f"{label}: C = {C:.3g}, $\\gamma$ = {gamma:.3g}"


print("\nFigures")
# ---- 1. primal and dual vs iteration, one panel per pair ----
n_p = len(pairs)
ncols = 3
nrows = int(np.ceil(n_p / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False)
for ax, (label, method, C, gamma) in zip(axes.ravel(), pairs):
    h = results[label]["r"]["history"]
    ax.plot(h["iteration"], h["primal"], color=P_COLOR, lw=1.8, label="Primal $P$ (minimised)")
    ax.plot(h["iteration"], h["dual"], color=D_COLOR, lw=1.8, label="Dual $D$ (maximised)")
    top = 1.4 * max(h["primal"][-1], h["dual"][-1])
    ax.set_ylim(0, top)                  # P starts at about C*n (off the top): zoom on the converging part
    ax.set_xlabel("SMO iteration (pair updates)")
    ax.set_ylabel("Objective value")
    ax.text(0.97, 0.05, pair_label(label, C, gamma), transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    ax.grid(alpha=0.25)
for ax in axes.ravel()[n_p:]:
    ax.axis("off")
axes.ravel()[0].legend(loc="center right", fontsize=8, frameon=False)
fig.tight_layout()
save(fig, "inner_primal_dual_grid.png")

# ---- 2. relative duality gap, all pairs ----
fig, ax = plt.subplots(figsize=(7, 4.4))
for label, method, C, gamma in pairs:
    h = results[label]["r"]["history"]
    it, rg = h["iteration"][1:], np.maximum(h["rel_gap"][1:], 1e-9)
    if label in COLORS:
        ax.plot(it, rg, color=COLORS[label], lw=1.6, label=pair_label(label, C, gamma))
    else:
        ax.plot(it, rg, color=CONTRAST_COLORS[label], lw=1.4, ls="--", label=pair_label(label, C, gamma))
ax.axhline(1e-3, color="k", ls=":", lw=1)
x_right = max(results[l]["r"]["history"]["iteration"][-1] for l, *_ in pairs)
ax.text(x_right, 1.3e-3, "convergence threshold $10^{-3}$", fontsize=8, ha="right", va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5), zorder=5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("SMO iteration (pair updates, log scale)")
ax.set_ylabel("Relative duality gap $(P - D)\\,/\\,|P|$")
ax.grid(alpha=0.25, which="both")
ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
save(fig, "inner_duality_gap.png")

# ---- 3. dual objective over two multipliers (the others fixed at the SMO solution) ----
ref = "PSO" if "PSO" in results else pairs[0][0]
R = results[ref]
alpha, C, K = R["r"]["alpha"], R["C"], R["K"]
Q = obj.q_matrix(K, y)
margin = np.where((alpha > 1e-4 * C) & (alpha < C * (1 - 1e-4)))[0]
order = margin[np.argsort(np.abs(alpha[margin] - C / 2))]      # margin SVs, the most central ones first
i, j = int(order[0]), int(order[1])
rest = alpha.copy()
rest[[i, j]] = 0.0
base_D = np.sum(rest) - 0.5 * rest @ (Q @ rest)
lin_i, lin_j = 1.0 - Q[i] @ rest, 1.0 - Q[j] @ rest              # linear terms of D in alpha_i, alpha_j
pad = 0.08 * C
ai_ax = np.linspace(-pad, C + pad, 300)
aj_ax = np.linspace(-pad, C + pad, 300)
AI, AJ = np.meshgrid(ai_ax, aj_ax)
Dsl = base_D + lin_i * AI + lin_j * AJ - 0.5 * (Q[i, i] * AI ** 2 + 2 * Q[i, j] * AI * AJ + Q[j, j] * AJ ** 2)
fig, ax = plt.subplots(figsize=(6.2, 5.4))
cf = ax.contourf(AI, AJ, Dsl, levels=25, cmap="viridis")
ax.contour(AI, AJ, Dsl, levels=25, colors="white", linewidths=0.3, alpha=0.6)
fig.colorbar(cf, ax=ax, label="Dual objective $D(\\alpha)$")
ax.plot([0, C, C, 0, 0], [0, 0, C, C, 0], color="white", lw=1.6)           # the box 0 <= alpha <= C
ax.text(0.03 * C, 0.03 * C, "feasible box $[0, C]^2$", color="white", fontsize=8)
c_eq = y[i] * alpha[i] + y[j] * alpha[j]
line = np.array([-pad, C + pad])
aj_line = (c_eq - y[i] * line) / y[j]                                        # y_i a_i + y_j a_j = c
ax.plot(line, aj_line, color="#ffcf33", lw=1.8, ls="--",
        label="equality constraint $y_i\\alpha_i + y_j\\alpha_j = $ const")
ax.plot(alpha[i], alpha[j], marker="*", ms=17, color="#ff3b30", mec="white", ls="none",
        label="constrained optimum $(\\alpha_i^*, \\alpha_j^*)$")
ax.set_xlim(-pad, C + pad)
ax.set_ylim(-pad, C + pad)
ax.set_xlabel(f"$\\alpha_i$ (point {i}, label {int(y[i]):+d})")
ax.set_ylabel(f"$\\alpha_j$ (point {j}, label {int(y[j]):+d})")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=7, frameon=False)
save(fig, "dual_slice_contour.png")
print(f"    (dual slice: {ref} pair, C = {C:.3g}, margin support vectors i = {i}, j = {j})")

# ---- 4. outer convergence ----
fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.2))
for name in names:
    d = outer[outer.method == name]
    axl.plot(d.trial, d.best_so_far, color=COLORS[name], lw=1.8, label=name)
    axr.scatter(d.trial, d.F, s=10, color=COLORS[name], alpha=0.6, label=name, linewidths=0)
for ax in (axl, axr):
    ax.axhline(TARGET, color="k", ls=":", lw=1.2)
    ax.set_xlabel("Trial number (evaluations of $F$)")
    ax.grid(alpha=0.25)
axl.set_ylabel("Best CV accuracy so far (fraction)")
axl.set_ylim(0.93, 0.985)
axl.text(2, TARGET + 0.0004, f"target {TARGET}", fontsize=8)
axl.legend(fontsize=8, frameon=False, loc="lower right")
axr.set_ylabel("CV accuracy of each trial (fraction)")
fig.tight_layout()
save(fig, "outer_convergence.png")

# ---- 5. / 6. contour plots of F ----
fmin, fmax = Fgrid.min(), Fgrid.max()
fill_levels = [lv for lv in [0.0, 0.7, 0.8, 0.9, 0.93, 0.95, 0.965, 0.97, 0.975, 0.98, 0.985, 1.0]
               if fmin - 0.2 < lv <= fmax + 0.02]
fill_levels = sorted(set([fmin - 1e-6] + [lv for lv in fill_levels if lv > fmin] + [fmax + 1e-6]))
line_levels = [lv for lv in (0.90, 0.95, 0.97, 0.975, 0.98) if fmin < lv < fmax]
LC, LG = np.meshgrid(lc_axis, lg_axis)


def draw_surface(ax, labels=True):
    cf = ax.contourf(LC, LG, Fgrid, levels=fill_levels, cmap="viridis")
    if line_levels:
        cs = ax.contour(LC, LG, Fgrid, levels=line_levels, colors="white", linewidths=0.8)
        if labels:
            ax.clabel(cs, fmt="%.3f", fontsize=7)
    ax.set_xlabel("$\\log_{10} C$")
    ax.set_ylabel("$\\log_{10} \\gamma$")
    ax.set_xlim(BOUNDS[0])
    ax.set_ylim(BOUNDS[1])
    return cf


fig, ax = plt.subplots(figsize=(7, 5.4))
cf = draw_surface(ax)
fig.colorbar(cf, ax=ax, label="CV accuracy $F$ (fraction)")
for name in names:
    ax.scatter(*best_points[name], marker=MARKERS[name], s=85, color=COLORS[name], edgecolor="black",
               linewidths=1.0, label=f"{name} best", zorder=4)
ax.scatter(grid_max[0], grid_max[1], marker="*", s=330, color="#ff3b30", edgecolor="white", linewidths=1.2,
           label=f"grid maximum (F = {grid_max[2]:.4f})", zorder=5)
ax.legend(fontsize=7, frameon=True, framealpha=0.85, loc="lower left")
save(fig, "outer_contour_all.png")

fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharex=True, sharey=True)
for ax, name in zip(axes.ravel(), names):
    draw_surface(ax, labels=False)
    p = trial_points[name]
    ax.plot(p[:, 0], p[:, 1], color="white", lw=0.6, alpha=0.7, zorder=2)
    ax.scatter(p[:, 0], p[:, 1], s=6, color=COLORS[name], alpha=0.9, zorder=3, linewidths=0)
    ax.scatter(*p[0], marker="o", s=90, facecolor="none", edgecolor="white", linewidths=1.8, zorder=4, label="start")
    ax.scatter(*best_points[name], marker="*", s=230, color=COLORS[name], edgecolor="black", linewidths=1.0,
               zorder=5, label="best")
    ax.text(0.03, 0.96, name, transform=ax.transAxes, color="white", fontsize=10, fontweight="bold", va="top")
    ax.set_xlabel("$\\log_{10} C$")
    ax.set_ylabel("$\\log_{10} \\gamma$")
axes.ravel()[0].legend(fontsize=7, loc="lower left", frameon=True, framealpha=0.85)
fig.tight_layout()
save(fig, "outer_contour_paths.png")

print(f"\nDone in {time.time() - t_start:.0f} s. Tables: results/inner_convergence.csv, outer_trials.csv, cv_surface.csv")
