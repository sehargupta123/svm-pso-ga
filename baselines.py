# baselines.py
# The two simple methods we compare PSO and GA against.

import numpy as np


def grid_search(score_fn, bounds, points_per_axis=10):
    """Try every combination on an evenly spaced grid (10 x 10 = 100 trials)."""
    best_x, best_score = None, -np.inf
    for a in np.linspace(bounds[0, 0], bounds[0, 1], points_per_axis):
        for b in np.linspace(bounds[1, 0], bounds[1, 1], points_per_axis):
            s = score_fn(a, b)
            if s > best_score:
                best_x, best_score = np.array([a, b]), s
    return best_x, best_score


def random_search(score_fn, bounds, n_trials=100, seed=0):
    """Try random points in the search space."""
    rng = np.random.default_rng(seed)
    best_x, best_score = None, -np.inf
    for _ in range(n_trials):
        p = rng.uniform(bounds[:, 0], bounds[:, 1])
        s = score_fn(*p)
        if s > best_score:
            best_x, best_score = p, s
    return best_x, best_score