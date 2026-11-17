# fitness.py
# The "fitness function": give it a (C, gamma) pair, it tells you how good that pair is.
# Score = average cross-validation score on the TRAINING data (accuracy by default).
# The test set is never touched here.
#
# We search in LOG scale because C and gamma range over many powers of 10:
#   log_C     from -2 to 3   ->  C     from 0.01   to 1000
#   log_gamma from -4 to 1   ->  gamma from 0.0001 to 10

import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from my_svm import MySVM

# ------------------------- SETTINGS (change here) -------------------------
# What number do we maximise?
#   "accuracy"         -> fraction of correct predictions (default)
#   "f1_malignant"     -> F1 score of the malignant class (label -1)
#   "recall_malignant" -> fraction of malignant cases that are caught (label -1)
SCORING = "accuracy"

# Number of times the 5-fold CV is repeated with different shuffles (1 = plain 5-fold CV).
N_REPEATS = 1

# Tie-breaking: if True, the score is   score - 1e-6 * (average number of support vectors),
# so between two settings with the same accuracy the SIMPLER model (fewer support vectors) wins.
#
# Why 1e-6 can never override a real accuracy difference:
#   The training set has 455 patients, so the smallest possible accuracy difference is
#   ONE patient = 1/455 = 0.0022. The most support vectors a fold model can have is the size
#   of its training part (about 364), so the penalty is at most 1e-6 * 364 = 0.00036, and two
#   settings can differ in penalty by at most that much. 0.00036 < 0.0022, so the penalty only
#   ever decides between settings whose accuracy is exactly equal.
TIE_BREAK_SV = True
TIE_BREAK_WEIGHT = 1e-6
# ---------------------------------------------------------------------------

# The search space: [lower, upper] for (log_C, log_gamma)
BOUNDS = np.array([[-2.0, 3.0],
                   [-4.0, 1.0]])

SCORING_CHOICES = ("accuracy", "f1_malignant", "recall_malignant")


def score_predictions(y_true, y_pred, scoring="accuracy"):
    """Turn predictions into one number. Malignant = label -1 (the 'positive' class here)."""
    if scoring == "accuracy":
        return float(np.mean(y_true == y_pred))
    tp = np.sum((y_true == -1) & (y_pred == -1))
    fn = np.sum((y_true == -1) & (y_pred == 1))
    fp = np.sum((y_true == 1) & (y_pred == -1))
    if scoring == "recall_malignant":
        return float(tp / (tp + fn)) if tp + fn > 0 else 0.0
    if scoring == "f1_malignant":
        return float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn > 0 else 0.0
    raise ValueError(f"SCORING must be one of {SCORING_CHOICES}, got {scoring!r}")


class Fitness:
    """Scores (log_C, log_gamma) pairs and remembers every score it gave.

    use_my_svm=True  -> uses our own SVM from my_svm.py (slower, ~1 minute per 100 trials)
    use_my_svm=False -> uses scikit-learn's SVC (much faster, same answers)

    Attributes after some calls:
      history -> the value the optimiser saw for each call (includes the tie-break penalty)
      points  -> the (log_C, log_gamma) of each call
      cache   -> {(log_C, log_gamma): (base_score, avg_support_vectors)}  one entry per DIFFERENT point
                 (base_score = the CV score without the tie-break penalty)
    """

    def __init__(self, X, y, use_my_svm=True, n_folds=5, scoring=None, n_repeats=None, tie_break=None):
        self.X, self.y = X, y
        self.use_my_svm = use_my_svm
        # Read the module settings when the object is created, so scripts can change them first.
        self.scoring = SCORING if scoring is None else scoring
        self.n_repeats = N_REPEATS if n_repeats is None else n_repeats
        self.tie_break = TIE_BREAK_SV if tie_break is None else tie_break
        if self.scoring not in SCORING_CHOICES:
            raise ValueError(f"SCORING must be one of {SCORING_CHOICES}, got {self.scoring!r}")
        if self.n_repeats == 1:
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        else:
            splitter = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=self.n_repeats, random_state=0)
        self.folds = list(splitter.split(X, y))   # the SAME folds for every (C, gamma): a fair comparison
        self.cache = {}      # remembers scores so the same point is never trained twice
        self.history = []    # one score per call, in order (used for convergence plots)
        self.points = []     # every (log_C, log_gamma) tried, in order (used for the search-space plot)

    def _make_model(self, C, gamma):
        if self.use_my_svm:
            return MySVM(C=C, gamma=gamma)
        return SVC(C=C, gamma=gamma, kernel="rbf")

    def _cv_score(self, log_C, log_gamma):
        """Run the cross-validation for one point. Returns (mean score, mean number of support vectors)."""
        C, gamma = 10 ** log_C, 10 ** log_gamma
        scores, n_svs = [], []
        for train_idx, val_idx in self.folds:
            # Scale INSIDE each fold so validation data never leaks into training.
            scaler = StandardScaler().fit(self.X[train_idx])
            X_tr = scaler.transform(self.X[train_idx])
            X_val = scaler.transform(self.X[val_idx])
            model = self._make_model(C, gamma).fit(X_tr, self.y[train_idx])
            scores.append(score_predictions(self.y[val_idx], model.predict(X_val), self.scoring))
            n_svs.append(model.n_support if self.use_my_svm else model.n_support_.sum())
        return float(np.mean(scores)), float(np.mean(n_svs))

    def evaluate(self, log_C, log_gamma):
        """Returns the fitness of this point (about 0..1, higher is better)."""
        key = (round(float(log_C), 4), round(float(log_gamma), 4))
        if key not in self.cache:
            self.cache[key] = self._cv_score(log_C, log_gamma)
        base, n_sv = self.cache[key]
        value = base - TIE_BREAK_WEIGHT * n_sv if self.tie_break else base
        self.history.append(value)      # the SAME value the optimiser saw, so plots stay consistent
        self.points.append((float(log_C), float(log_gamma)))
        return value

    def lookup(self, log_C, log_gamma):
        """Returns (base_score, avg_support_vectors) for a point, scoring it first if needed.

        Does not add anything to history, and has no tie-break penalty. Used for reporting.
        """
        key = (round(float(log_C), 4), round(float(log_gamma), 4))
        if key not in self.cache:
            self.cache[key] = self._cv_score(log_C, log_gamma)
        return self.cache[key]

    def n_repeated(self):
        """How many of the calls since the last reset were repeats of an already-tried point."""
        unique = {(round(a, 4), round(b, 4)) for a, b in self.points}
        return len(self.points) - len(unique)

    def reset_history(self):
        self.history = []
        self.points = []
