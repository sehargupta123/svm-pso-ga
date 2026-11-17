# baselines.py
# The simple methods we compare PSO, GA and SA against:
#   grid search, random search, and (if optuna is installed) Bayesian optimisation with TPE.

import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)    # silence the per-trial messages
    HAVE_OPTUNA = True
except ImportError:
    HAVE_OPTUNA = False


def grid_search(score_fn, bounds, points_per_axis=10):
    """Try every combination on an evenly spaced grid (10 x 10 = 100 trials).

    Scans row by row: log_C from small to large in the outer loop, log_gamma in the inner loop.
    """
    best_x, best_score = None, -np.inf
    for a in np.linspace(bounds[0, 0], bounds[0, 1], points_per_axis):
        for b in np.linspace(bounds[1, 0], bounds[1, 1], points_per_axis):
            s = score_fn(a, b)
            if s > best_score:
                best_x, best_score = np.array([a, b]), s
    return best_x, best_score


def random_search(score_fn, bounds, n_trials=100, seed=0):
    """Try random points in the search space (uniform in the log space)."""
    rng = np.random.default_rng(seed)
    best_x, best_score = None, -np.inf
    for _ in range(n_trials):
        p = rng.uniform(bounds[:, 0], bounds[:, 1])
        s = score_fn(*p)
        if s > best_score:
            best_x, best_score = p, s
    return best_x, best_score


def bayesian_tpe(score_fn, bounds, n_trials=100, seed=0):
    """Bayesian optimisation with optuna's TPE sampler: it learns from past trials where to look next.

    Needs:  pip install optuna.  `seed` must be a plain integer here.
    """
    if not HAVE_OPTUNA:
        raise ImportError("optuna is not installed. Run:  pip install optuna")
    def objective(trial):
        a = trial.suggest_float("log_C", bounds[0, 0], bounds[0, 1])
        b = trial.suggest_float("log_gamma", bounds[1, 0], bounds[1, 1])
        s = score_fn(a, b)
        return s

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=int(seed)))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    return np.array([best["log_C"], best["log_gamma"]]), study.best_value
