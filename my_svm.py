# my_svm.py
# A Support Vector Machine written from scratch.
# Training = solving the SVM "dual" problem, which is a QUADRATIC PROGRAM (QP).
#
# The dual problem (Unit 1 + Unit 2 of the syllabus):
#   maximise   sum(alpha) - 1/2 * sum_i sum_j alpha_i alpha_j y_i y_j K(x_i, x_j)
#   subject to 0 <= alpha_i <= C   and   sum(alpha_i * y_i) = 0
#
# Two solvers are available:
#   MySVM(solver="qp")  -> hand the QP to the cvxopt solver (default, accurate)
#   MySVM(solver="smo") -> Platt's (simplified) Sequential Minimal Optimisation, written by hand

import numpy as np
from cvxopt import matrix, solvers

solvers.options["show_progress"] = False   # stop cvxopt printing every iteration


def rbf_kernel(A, B, gamma):
    """RBF kernel: K(a, b) = exp(-gamma * ||a - b||^2), for every row of A against every row of B."""
    sq_dist = (
        np.sum(A ** 2, axis=1).reshape(-1, 1)
        + np.sum(B ** 2, axis=1).reshape(1, -1)
        - 2 * A @ B.T
    )
    sq_dist = np.maximum(sq_dist, 0)        # remove tiny negative values caused by rounding
    return np.exp(-gamma * sq_dist)


class MySVM:
    def __init__(self, C=1.0, gamma=0.1, solver="qp", smo_tol=1e-3, smo_max_passes=10,
                 smo_max_iter=200, seed=0):
        if solver not in ("qp", "smo"):
            raise ValueError("solver must be 'qp' or 'smo'")
        self.C = C            # penalty for misclassified points
        self.gamma = gamma    # how curvy the boundary can be
        self.solver = solver
        self.smo_tol = smo_tol                  # how wrong a point may be before SMO updates it
        self.smo_max_passes = smo_max_passes    # stop after this many full sweeps with no change
        self.smo_max_iter = smo_max_iter        # hard cap on sweeps so SMO can never loop forever
        self.seed = seed                        # SMO picks its second point at random

    # Tolerances scale with C, because the QP solver is only accurate to a few digits RELATIVE to C.
    def _a_tol(self):
        return 1e-5 * max(self.C, 1.0)          # alpha below this counts as zero

    def _c_tol(self):
        return 1e-3 * self.C                    # alpha above C - c_tol counts as "at the bound C"

    # ---------------- solver 1: cvxopt QP ----------------
    def _solve_qp(self, K, y):
        n = len(y)
        # cvxopt solves:  minimise 1/2 x'Px + q'x   subject to  Gx <= h  and  Ax = b
        # Maximising the dual  == minimising its negative, so P = y y' * K and q = -1.
        P = matrix(np.outer(y, y) * K + 1e-8 * np.eye(n))   # tiny value on the diagonal keeps the solver stable
        q = matrix(-np.ones(n))
        G = matrix(np.vstack([-np.eye(n), np.eye(n)]))       # -alpha <= 0   and   alpha <= C
        h = matrix(np.hstack([np.zeros(n), self.C * np.ones(n)]))
        A = matrix(y.reshape(1, -1))                         # sum(alpha * y) = 0
        b = matrix(0.0)
        result = solvers.qp(P, q, G, h, A, b)
        if result["x"] is None:
            raise RuntimeError("cvxopt failed: " + str(result["status"]))
        return np.ravel(result["x"])

    # ---------------- solver 2: simplified SMO (Platt 1998) ----------------
    def _solve_smo(self, K, y):
        """Optimise two alphas at a time (the equality constraint needs at least two to move)."""
        n, C, tol = len(y), self.C, self.smo_tol
        rng = np.random.default_rng(self.seed)
        alpha = np.zeros(n)
        b = 0.0
        f = np.zeros(n)               # f[i] = current decision value of point i (kept up to date)
        passes = sweeps = 0
        while passes < self.smo_max_passes and sweeps < self.smo_max_iter:
            changed = 0
            for i in range(n):
                Ei = f[i] - y[i]
                # Does point i break the KKT conditions (by more than tol)?
                if (y[i] * Ei < -tol and alpha[i] < C) or (y[i] * Ei > tol and alpha[i] > 0):
                    j = int(rng.integers(n - 1))
                    if j >= i:
                        j += 1        # pick any j different from i
                    Ej = f[j] - y[j]
                    ai_old, aj_old = alpha[i], alpha[j]
                    # Box limits for alpha_j so both alphas stay inside [0, C]
                    if y[i] != y[j]:
                        L, H = max(0.0, aj_old - ai_old), min(C, C + aj_old - ai_old)
                    else:
                        L, H = max(0.0, ai_old + aj_old - C), min(C, ai_old + aj_old)
                    if L == H:
                        continue
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]   # curvature along the line (<= 0 for a kernel)
                    if eta >= 0:
                        continue
                    aj = np.clip(aj_old - y[j] * (Ei - Ej) / eta, L, H)
                    if abs(aj - aj_old) < 1e-5:
                        continue
                    ai = ai_old + y[i] * y[j] * (aj_old - aj)   # keeps sum(alpha*y) unchanged
                    b1 = b - Ei - y[i] * (ai - ai_old) * K[i, i] - y[j] * (aj - aj_old) * K[i, j]
                    b2 = b - Ej - y[i] * (ai - ai_old) * K[i, j] - y[j] * (aj - aj_old) * K[j, j]
                    if 0 < ai < C:
                        b_new = b1
                    elif 0 < aj < C:
                        b_new = b2
                    else:
                        b_new = (b1 + b2) / 2
                    f += y[i] * (ai - ai_old) * K[:, i] + y[j] * (aj - aj_old) * K[:, j] + (b_new - b)
                    alpha[i], alpha[j], b = ai, aj, b_new
                    changed += 1
            passes = passes + 1 if changed == 0 else 0
            sweeps += 1
        return alpha

    def fit(self, X, y):
        y = y.astype(float)
        K = rbf_kernel(X, X, self.gamma)
        alpha = self._solve_qp(K, y) if self.solver == "qp" else self._solve_smo(K, y)

        # Support vectors = training points with alpha > 0 (we use a small tolerance).
        sv = alpha > self._a_tol()
        self.alpha = alpha[sv]
        self.sv_X = X[sv]
        self.sv_y = y[sv]
        self.n_support = int(np.sum(sv))

        # Bias b: for any margin SV (0 < alpha < C) we have y_i * f(x_i) = 1, so
        # b = y_i - sum_j alpha_j y_j K(x_j, x_i). Average over all of them for stability.
        on_margin = sv & (alpha < self.C - self._c_tol())
        if not np.any(on_margin):
            on_margin = sv       # every SV is at C: fall back to all SVs
        K_margin = rbf_kernel(X[on_margin], self.sv_X, self.gamma)
        self.b = np.mean(y[on_margin] - K_margin @ (self.alpha * self.sv_y))

        # Keep the full alpha vector for the KKT check.
        self._alpha_all, self._X_train, self._y_train = alpha, X, y
        return self

    def decision_function(self, X):
        """Raw score f(x). Positive -> class +1, negative -> class -1."""
        K = rbf_kernel(X, self.sv_X, self.gamma)
        return K @ (self.alpha * self.sv_y) + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def score(self, X, y):
        """Accuracy = fraction of correct predictions."""
        return np.mean(self.predict(X) == y)

    def check_kkt(self, tol=1e-3):
        """Percentage of training points that satisfy the KKT conditions.

        alpha = 0      ->  y * f(x) >= 1   (correct side, outside the margin)
        0 < alpha < C  ->  y * f(x) == 1   (exactly on the margin)
        alpha = C      ->  y * f(x) <= 1   (inside the margin or misclassified)
        """
        alpha, y = self._alpha_all, self._y_train
        yf = y * self.decision_function(self._X_train)
        zero = alpha <= self._a_tol()
        at_c = alpha >= self.C - self._c_tol()
        middle = ~zero & ~at_c
        ok = np.zeros_like(alpha, dtype=bool)
        ok[zero] = yf[zero] >= 1 - tol
        ok[at_c] = yf[at_c] <= 1 + tol
        ok[middle] = np.abs(yf[middle] - 1) <= tol
        return 100 * np.mean(ok)
