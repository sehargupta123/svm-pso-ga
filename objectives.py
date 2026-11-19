# objectives.py
# The objective functions of this project, written out explicitly as NumPy functions.
#
# There are TWO optimisation problems, one inside the other (a "bilevel" problem):
#
#   INNER problem (training one SVM, C and gamma fixed):
#       maximise  D(alpha) = sum_i alpha_i - 1/2 alpha^T Q alpha
#       s.t.      0 <= alpha_i <= C,   sum_i alpha_i y_i = 0
#       (equivalently minimise the primal P; the two meet at the optimum)
#
#   OUTER problem (choosing C and gamma):
#       maximise  F(theta) = cross-validated accuracy of the SVM trained with theta = (log10 C, log10 gamma)
#
# Everything below is plain NumPy. scikit-learn is NOT used inside any objective formula.
# (The tuning scripts use it only for fold splitting, and for a fast SVC when scoring many pairs.)

import numpy as np


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------
def rbf_kernel(A, B, gamma):
    """RBF (Gaussian) kernel matrix.

        K_ij = exp( -gamma * ||a_i - b_j||^2 )

    Symbols:
        A     : (n, d) array, rows a_i are points
        B     : (m, d) array, rows b_j are points
        gamma : kernel width parameter (> 0). Large gamma = very local, curvy boundary.
    Returns the (n, m) matrix K. It is a similarity: K_ij = 1 when a_i = b_j, and -> 0 far apart.
    (Not an optimisation problem, so neither minimised nor maximised.)
    """
    sq_dist = (np.sum(A ** 2, axis=1)[:, None] + np.sum(B ** 2, axis=1)[None, :] - 2.0 * A @ B.T)
    sq_dist = np.maximum(sq_dist, 0.0)          # rounding can give tiny negative numbers
    return np.exp(-gamma * sq_dist)


def q_matrix(K, y):
    """Label-weighted kernel matrix of the dual problem.

        Q_ij = y_i * y_j * K_ij

    Symbols:
        K : (n, n) kernel matrix on the training points
        y : (n,) labels, each -1 or +1
    Q is positive semi-definite, so the dual D below is a concave quadratic in alpha.
    """
    y = np.asarray(y, dtype=float)
    return y[:, None] * y[None, :] * K


# --------------------------------------------------------------------------------------
# INNER problem: dual and primal SVM objectives
# --------------------------------------------------------------------------------------
def dual_objective(alpha, Q):
    """Dual objective of the soft-margin SVM  (MAXIMISE this).

        D(alpha) = sum_i alpha_i  -  1/2 * alpha^T Q alpha

        subject to   0 <= alpha_i <= C   for every i      (box constraint)
                     sum_i alpha_i * y_i = 0              (equality constraint)

    Symbols:
        alpha : (n,) Lagrange multipliers, one per training point
        Q     : (n, n) matrix, Q_ij = y_i y_j K_ij  (see q_matrix)
        C     : penalty on margin violations (only appears in the constraints)
    For every feasible alpha, D(alpha) <= P(alpha', b') for every feasible primal point
    (weak duality), so D is always a LOWER bound on the optimal primal value.
    """
    alpha = np.asarray(alpha, dtype=float)
    return float(np.sum(alpha) - 0.5 * alpha @ (Q @ alpha))


def decision_values(alpha, b, K, y):
    """Decision function of the trained SVM, evaluated for every row of K.

        f(x_i) = sum_j alpha_j * y_j * K(x_i, x_j)  +  b

    Symbols:
        alpha : (n,) multipliers of the n TRAINING points
        b     : bias (a single number)
        K     : (m, n) kernel between the m points to score (rows) and the n training points (columns).
                For the training points themselves, K is the (n, n) training kernel.
        y     : (n,) training labels (-1 / +1)
    Positive f -> predict class +1, negative f -> class -1.
    """
    return K @ (np.asarray(alpha, dtype=float) * np.asarray(y, dtype=float)) + b


def primal_objective(alpha, b, K, y, C):
    """Primal objective of the soft-margin SVM, written in terms of alpha  (MINIMISE this).

        P(alpha, b) = 1/2 * alpha^T Q alpha  +  C * sum_i max(0, 1 - y_i * f(x_i))

    Where it comes from. The textbook primal is

        minimise   1/2 ||w||^2  +  C * sum_i xi_i
        subject to y_i (w . phi(x_i) + b) >= 1 - xi_i,   xi_i >= 0.

    Substitute  w = sum_j alpha_j y_j phi(x_j)  (the representer form). Then
        ||w||^2 = sum_ij alpha_i alpha_j y_i y_j K_ij = alpha^T Q alpha,
        w . phi(x_i) = sum_j alpha_j y_j K_ij, so  w . phi(x_i) + b = f(x_i).
    For fixed w and b, the slack xi_i is free to be any value that satisfies its two
    constraints, and since xi_i is PENALISED we take the smallest feasible one:
        xi_i = max(0, 1 - y_i f(x_i))     (the hinge loss).
    Plugging that in gives the formula above.

    Symbols:
        alpha : (n,) multipliers;  b : bias;  K : (n, n) training kernel;  y : (n,) labels;  C : penalty.
    Valid for any alpha (feasible for the dual or not), which is what makes the duality gap computable.
    """
    alpha = np.asarray(alpha, dtype=float)
    y = np.asarray(y, dtype=float)
    f = decision_values(alpha, b, K, y)
    quad = 0.5 * float(alpha @ (y * (K @ (y * alpha))))        # 1/2 alpha^T Q alpha (without building Q)
    hinge = np.maximum(0.0, 1.0 - y * f)
    return quad + C * float(np.sum(hinge))


def duality_gap(alpha, b, K, y, C):
    """Duality gap between the primal and the dual at the same alpha.

        gap     = P(alpha, b) - D(alpha)               (>= 0 by WEAK duality, for feasible alpha)
        rel_gap = (P - D) / (|P| + 1e-12)

    Strong duality (the SVM problem is a convex QP) says the gap -> 0 at the optimum,
    so a small gap proves the solver has converged, without knowing the true optimum.

    Returns (P, D, gap, rel_gap).
    """
    Q = q_matrix(K, y)
    P = primal_objective(alpha, b, K, y, C)
    D = dual_objective(alpha, Q)
    gap = P - D
    return P, D, gap, gap / (abs(P) + 1e-12)


def check_dual_feasibility(alpha, y, C):
    """How badly alpha breaks the dual constraints.

        box violation      = max_i  max(0, -alpha_i, alpha_i - C)     (want 0)
        equality violation = | sum_i alpha_i y_i |                    (want 0)

    Returns (max_box_violation, equality_violation).
    """
    alpha = np.asarray(alpha, dtype=float)
    box = max(0.0, float(np.max(-alpha)), float(np.max(alpha - C)))
    eq = abs(float(np.sum(alpha * np.asarray(y, dtype=float))))
    return box, eq


def bias_from_alpha(alpha, K, y, C):
    """Bias b used for classification: average of  y_i - sum_j alpha_j y_j K_ij  over the margin
    support vectors (0 < alpha_i < C), where KKT says y_i f(x_i) = 1 exactly."""
    alpha = np.asarray(alpha, dtype=float)
    y = np.asarray(y, dtype=float)
    s = K @ (alpha * y)
    margin = (alpha > 1e-5 * max(C, 1.0)) & (alpha < C - 1e-3 * C)
    if not np.any(margin):
        margin = alpha > 1e-5 * max(C, 1.0)         # all SVs at the bound C
    if not np.any(margin):
        return 0.0
    return float(np.mean(y[margin] - s[margin]))


def best_bias(alpha, K, y, C):
    """The b that makes the primal P(alpha, b) smallest for this alpha.

    P depends on b only through  C * sum_i max(0, 1 - y_i (s_i + b)),  s_i = sum_j alpha_j y_j K_ij.
    That is convex and piecewise linear in b, with kinks at b = y_i - s_i, so its minimum is at one
    of those n values. Using this b means the logged P is the tightest primal value for the given
    alpha (and still >= D), so the gap is not inflated by a half-converged bias.
    """
    alpha = np.asarray(alpha, dtype=float)
    y = np.asarray(y, dtype=float)
    s = K @ (alpha * y)
    cand = y - s                                              # (n,) candidate biases
    hinge = np.maximum(0.0, 1.0 - y[None, :] * (s[None, :] + cand[:, None]))   # (candidates, points)
    return float(cand[np.argmin(hinge.sum(axis=1))])


# --------------------------------------------------------------------------------------
# SMO (Platt 1998, full heuristics) with a logged history
# --------------------------------------------------------------------------------------
class _SMO:
    """Platt's SMO: error cache, second-choice heuristic |E_i - E_j|, non-bound loops.

    Each update moves TWO alphas along the line  y_i alpha_i + y_j alpha_j = const, clipped to the box,
    so alpha is feasible (box AND equality) after every single update.
    """

    def __init__(self, K, y, C, seed=0):
        self.K, self.y, self.C = K, np.asarray(y, dtype=float), C
        self.n = len(self.y)
        self.rng = np.random.default_rng(seed)
        self.alpha = np.zeros(self.n)
        self.b = 0.0
        self.E = -self.y.copy()          # E_i = f(x_i) - y_i  (f = 0 at the start)
        self.updates = 0                 # number of successful pair updates so far
        self.max_updates = np.inf
        self.on_update = None            # callback(self) after each successful update

    def _take_step(self, i, j):
        if i == j:
            return 0
        K, y, C, alpha = self.K, self.y, self.C, self.alpha
        ai, aj, yi, yj = alpha[i], alpha[j], y[i], y[j]
        Ei, Ej = self.E[i], self.E[j]
        if yi != yj:
            L, H = max(0.0, aj - ai), min(C, C + aj - ai)
        else:
            L, H = max(0.0, ai + aj - C), min(C, ai + aj)
        if H - L < 1e-12:
            return 0
        eta = K[i, i] + K[j, j] - 2.0 * K[i, j]       # curvature along the line (>= 0)
        if eta < 1e-12:
            return 0
        aj_new = min(max(aj + yj * (Ei - Ej) / eta, L), H)
        if abs(aj_new - aj) < 1e-9 * (aj + aj_new + 1e-3):
            return 0
        ai_new = ai + yi * yj * (aj - aj_new)         # keeps sum(alpha * y) unchanged
        ai_new = min(max(ai_new, 0.0), C)             # remove rounding at the edges
        dai, daj = ai_new - ai, aj_new - aj
        b1 = self.b - Ei - yi * dai * K[i, i] - yj * daj * K[i, j]
        b2 = self.b - Ej - yi * dai * K[i, j] - yj * daj * K[j, j]
        if 0 < ai_new < C:
            b_new = b1
        elif 0 < aj_new < C:
            b_new = b2
        else:
            b_new = 0.5 * (b1 + b2)
        self.E += yi * dai * K[:, i] + yj * daj * K[:, j] + (b_new - self.b)
        alpha[i], alpha[j], self.b = ai_new, aj_new, b_new
        self.updates += 1
        if self.on_update is not None:
            self.on_update(self)
        return 1

    def _examine(self, j, tol):
        C, alpha, E, y = self.C, self.alpha, self.E, self.y
        r = E[j] * y[j]
        if not ((r < -tol and alpha[j] < C) or (r > tol and alpha[j] > 0)):
            return 0                                      # point j already satisfies KKT
        nonbound = np.where((alpha > 0) & (alpha < C))[0]
        if len(nonbound) > 1:                             # second-choice heuristic: biggest |E_i - E_j|
            i = nonbound[np.argmax(np.abs(E[nonbound] - E[j]))]
            if self._take_step(i, j):
                return 1
        for pool in (nonbound, np.arange(self.n)):        # then non-bound points, then everything
            if len(pool) == 0:
                continue
            start = int(self.rng.integers(len(pool)))
            for k in range(len(pool)):
                if self._take_step(pool[(start + k) % len(pool)], j):
                    return 1
        return 0

    def run(self, tol):
        """Sweeps until no point violates KKT by more than tol (or the update cap is hit)."""
        examine_all, changed = True, 0
        while (changed > 0 or examine_all) and self.updates < self.max_updates:
            changed = 0
            if examine_all:
                idx = np.arange(self.n)
            else:
                idx = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
            for j in idx:
                changed += self._examine(j, tol)
                if self.updates >= self.max_updates:
                    break
            if examine_all:
                examine_all = False
            elif changed == 0:
                examine_all = True


def smo_trace(K, y, C, tol=1e-3, rel_gap_tol=1e-3, max_updates=60000, n_log=200, log=True, seed=0):
    """Train the SVM dual with SMO and record how D, P and the gap evolve.

    Every m successful pair updates we record: iteration number, D(alpha), P(alpha, b), gap,
    relative gap and feasibility. m is chosen so the history has about n_log points
    (a first, silent run counts the updates; SMO is deterministic given the seed, so the second
    run repeats it exactly). With log=False only the solution is computed (single pass).

    P is logged at the best bias for the current alpha (see best_bias), so P >= D always.
    KKT tolerance tol is tightened (divided by 10, up to 3 times) if the relative gap is still
    above rel_gap_tol when SMO stops. converged = (final relative gap < rel_gap_tol).

    Returns a dict: alpha, b, converged, n_updates, m, and 'history' (dict of arrays:
    iteration, dual, primal, gap, rel_gap, box_violation, eq_violation).
    """
    y = np.asarray(y, dtype=float)

    def solve(m, record):
        smo = _SMO(K, y, C, seed)
        smo.max_updates = max_updates
        rows = []

        def snapshot(s):
            b = best_bias(s.alpha, K, y, C)
            P, D, gap, rel = duality_gap(s.alpha, b, K, y, C)
            box, eq = check_dual_feasibility(s.alpha, y, C)
            rows.append((s.updates, D, P, gap, rel, box, eq))
            return rel, b

        if record:
            snapshot(smo)                                  # iteration 0: alpha = 0
            smo.on_update = lambda s: snapshot(s) if s.updates % m == 0 else None
        cur_tol, rel = tol, np.inf
        for _ in range(4):
            smo.run(cur_tol)
            b_opt = best_bias(smo.alpha, K, y, C)
            rel = duality_gap(smo.alpha, b_opt, K, y, C)[3]
            if rel < rel_gap_tol or smo.updates >= max_updates:
                break
            cur_tol /= 10.0
        if record and (not rows or rows[-1][0] != smo.updates):
            snapshot(smo)                                  # always end the history on the final alpha
        return smo, rel, rows

    if not log:
        smo, rel, _ = solve(1, False)
        return {"alpha": smo.alpha.copy(), "b": bias_from_alpha(smo.alpha, K, y, C),
                "converged": bool(rel < rel_gap_tol), "n_updates": smo.updates, "m": None, "history": None}

    n_total = solve(1, False)[0].updates
    m = max(1, int(np.ceil(n_total / n_log)))
    smo, rel, rows = solve(m, True)
    arr = np.array(rows)
    names = ["iteration", "dual", "primal", "gap", "rel_gap", "box_violation", "eq_violation"]
    history = {name: arr[:, k] for k, name in enumerate(names)}
    history["iteration"] = history["iteration"].astype(int)
    return {"alpha": smo.alpha.copy(), "b": bias_from_alpha(smo.alpha, K, y, C),
            "converged": bool(rel < rel_gap_tol), "n_updates": smo.updates, "m": m, "history": history}


# --------------------------------------------------------------------------------------
# OUTER problem: cross-validated accuracy
# --------------------------------------------------------------------------------------
def solve_smo(K, y, C):
    """Solver for cv_objective: our NumPy SMO (no logging). Tighter tolerances than the plots use, so a
    validation patient lying almost exactly on the boundary is not flipped by solver error."""
    return smo_trace(K, y, C, tol=1e-4, rel_gap_tol=1e-5, log=False)["alpha"]


def solve_qp(K, y, C):
    """Solver for cv_objective: the cvxopt QP from my_svm.py (same dual problem, interior-point)."""
    from my_svm import MySVM
    return MySVM(C=C)._solve_qp(K, np.asarray(y, dtype=float))


def cv_objective(log_C, log_gamma, X, y, folds, solver=solve_smo):
    """Cross-validated accuracy of the SVM trained with hyperparameters theta  (MAXIMISE this).

        F(theta) = (1/K) * sum_{k=1..K}  (1/|V_k|) * sum_{i in V_k}  1[ sign(f_theta^(k)(x_i)) = y_i ]

    Symbols:
        theta = (log10 C, log10 gamma), searched in [-2, 3] x [-4, 1]
                (so C = 10**log_C in [0.01, 1000] and gamma = 10**log_gamma in [1e-4, 10])
        K     : number of folds (5);  V_k = validation indices of fold k;  |V_k| = its size
        f_theta^(k) : decision function of the SVM trained on the OTHER folds (fold k held out)
        1[.]  : indicator, 1 if the statement is true else 0
    The scaler (mean and standard deviation) is fitted on the training part of each fold only, so
    nothing leaks from the validation part.

    Bilevel form. F is not a closed formula in theta: every evaluation solves an inner problem,

        max_theta   F( theta, alpha*(theta) )
        where       alpha*(theta) = argmax_alpha D_theta(alpha),  0 <= alpha_i <= C,  sum_i alpha_i y_i = 0.

    Because the accuracy counts correct patients, F is a step function of theta (piecewise constant),
    which is why gradient-free tuners (PSO, GA, ...) are used.

    X, y   : data (rows = samples), labels -1/+1;   folds : list of (train_idx, val_idx) pairs
    solver : function (K, y, C) -> alpha, for example solve_smo (default) or solve_qp
    """
    if solver == "smo":
        solver = solve_smo
    elif solver == "qp":
        solver = solve_qp
    C, gamma = 10.0 ** log_C, 10.0 ** log_gamma
    fold_acc = []
    for tr, va in folds:
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)                    # same convention as StandardScaler
        Xtr, Xva = (X[tr] - mu) / sd, (X[va] - mu) / sd
        ytr = np.asarray(y[tr], dtype=float)
        Ktr = rbf_kernel(Xtr, Xtr, gamma)
        alpha = solver(Ktr, ytr, C)                        # inner problem: argmax of the dual
        b = bias_from_alpha(alpha, Ktr, ytr, C)
        f = decision_values(alpha, b, rbf_kernel(Xva, Xtr, gamma), ytr)
        pred = np.where(f >= 0, 1, -1)
        fold_acc.append(np.mean(pred == y[va]))
    return float(np.mean(fold_acc))
