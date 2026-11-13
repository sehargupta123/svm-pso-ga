# my_svm.py
# A Support Vector Machine written from scratch.
# Training = solving the SVM "dual" problem, which is a QUADRATIC PROGRAM (QP).
# We hand the QP to the cvxopt solver.
#
# The dual problem (Unit 1 + Unit 2 of the syllabus):
#   maximise   sum(alpha) - 1/2 * sum_i sum_j alpha_i alpha_j y_i y_j K(x_i, x_j)
#   subject to 0 <= alpha_i <= C   and   sum(alpha_i * y_i) = 0

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
    def __init__(self, C=1.0, gamma=0.1):
        self.C = C            # penalty for misclassified points
        self.gamma = gamma    # how curvy the boundary can be

    def fit(self, X, y):
        n = X.shape[0]
        y = y.astype(float)
        K = rbf_kernel(X, X, self.gamma)

        # cvxopt solves:  minimise 1/2 x'Px + q'x   subject to  Gx <= h  and  Ax = b
        P = matrix(np.outer(y, y) * K + 1e-8 * np.eye(n))   # tiny value on the diagonal keeps the solver stable
        q = matrix(-np.ones(n))
        G = matrix(np.vstack([-np.eye(n), np.eye(n)]))       # -alpha <= 0   and   alpha <= C
        h = matrix(np.hstack([np.zeros(n), self.C * np.ones(n)]))
        A = matrix(y.reshape(1, -1))                         # sum(alpha * y) = 0
        b = matrix(0.0)

        result = solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(result["x"])

        # Support vectors = training points with alpha > 0 (we use a small tolerance).
        sv = alpha > 1e-5
        self.alpha = alpha[sv]
        self.sv_X = X[sv]
        self.sv_y = y[sv]
        self.n_support = int(np.sum(sv))

        # Bias b: average over support vectors that sit exactly on the margin (0 < alpha < C).
        on_margin = (alpha > 1e-5) & (alpha < self.C - 1e-5)
        if not np.any(on_margin):
            on_margin = sv
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
        a_tol = 1e-5
        zero = alpha <= a_tol
        at_c = alpha >= self.C - a_tol
        middle = ~zero & ~at_c
        ok = np.zeros_like(alpha, dtype=bool)
        ok[zero] = yf[zero] >= 1 - tol
        ok[at_c] = yf[at_c] <= 1 + tol
        ok[middle] = np.abs(yf[middle] - 1) <= tol
        return 100 * np.mean(ok)