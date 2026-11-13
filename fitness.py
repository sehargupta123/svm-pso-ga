# fitness.py
# The "fitness function": give it a (C, gamma) pair, it tells you how good that pair is.
# Score = average accuracy over 5-fold cross-validation on the TRAINING data.
# The test set is never touched here.
#
# We search in LOG scale because C and gamma range over many powers of 10:
#   log_C     from -2 to 3   ->  C     from 0.01   to 1000
#   log_gamma from -4 to 1   ->  gamma from 0.0001 to 10

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from my_svm import MySVM

# The search space: [lower, upper] for (log_C, log_gamma)
BOUNDS = np.array([[-2.0, 3.0],
                   [-4.0, 1.0]])


class Fitness:
    """Scores (log_C, log_gamma) pairs and remembers every score it gave.

    use_my_svm=True  -> uses our own SVM from my_svm.py (slower, ~1 minute per 100 trials)
    use_my_svm=False -> uses scikit-learn's SVC (much faster, same answers)
    """

    def __init__(self, X, y, use_my_svm=True, n_folds=5):
        self.X, self.y = X, y
        self.use_my_svm = use_my_svm
        self.folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0).split(X, y))
        self.cache = {}      # remembers scores so the same point is never trained twice
        self.history = []    # one score per call, in order (used for convergence plots)
        self.points = []     # every (log_C, log_gamma) tried, in order (used for the search-space plot)

    def _make_model(self, C, gamma):
        if self.use_my_svm:
            return MySVM(C=C, gamma=gamma)
        return SVC(C=C, gamma=gamma, kernel="rbf")

    def evaluate(self, log_C, log_gamma):
        """Returns mean cross-validation accuracy (a number between 0 and 1)."""
        key = (round(float(log_C), 4), round(float(log_gamma), 4))
        if key not in self.cache:
            C, gamma = 10 ** log_C, 10 ** log_gamma
            accs, n_svs = [], []
            for train_idx, val_idx in self.folds:
                # Scale INSIDE each fold so validation data never leaks into training.
                scaler = StandardScaler().fit(self.X[train_idx])
                X_tr = scaler.transform(self.X[train_idx])
                X_val = scaler.transform(self.X[val_idx])
                model = self._make_model(C, gamma).fit(X_tr, self.y[train_idx])
                accs.append(model.score(X_val, self.y[val_idx]))
                n_svs.append(model.n_support if self.use_my_svm else model.n_support_.sum())
            self.cache[key] = (float(np.mean(accs)), float(np.mean(n_svs)))
        acc = self.cache[key][0]
        self.history.append(acc)
        self.points.append((float(log_C), float(log_gamma)))
        return acc

    def reset_history(self):
        self.history = []
        self.points = []