# tuners.py
# One place that lists every tuning method, so step5 and step6 use identical settings.
# Every method gets the SAME budget: 100 calls to the fitness function.
#
# SEEDING: each method gets its own random stream, seed = [run_seed, method_number].
# (If they all used the plain number 0, PSO, GA and random search would start from the
#  very same 10 random points, which would make the comparison unfair.)

import numpy as np

from fitness import BOUNDS
from pso import pso
from ga import ga
from sa import simulated_annealing
from baselines import grid_search, random_search, bayesian_tpe, HAVE_OPTUNA

BUDGET = 100

# method number -> used for seeding (never reorder these, it would change the results)
METHOD_ID = {"Grid search": 0, "Random search": 1, "PSO": 2, "GA": 3, "SA": 4, "Bayesian (TPE)": 5}


def make_methods(include_bayesian=True):
    """Returns {name: run(score_fn, run_seed) -> (best_point, best_score)}."""
    def sid(name, s):
        return [s, METHOD_ID[name]]

    methods = {
        "Grid search":   lambda f, s: grid_search(f, BOUNDS, points_per_axis=10),
        "Random search": lambda f, s: random_search(f, BOUNDS, n_trials=BUDGET, seed=sid("Random search", s)),
        "PSO":           lambda f, s: pso(f, BOUNDS, n_particles=10, n_iters=10, seed=sid("PSO", s)),
        "GA":            lambda f, s: ga(f, BOUNDS, pop_size=10, max_evals=BUDGET, seed=sid("GA", s)),
        "SA":            lambda f, s: simulated_annealing(f, BOUNDS, max_evals=BUDGET, seed=sid("SA", s)),
    }
    if include_bayesian and HAVE_OPTUNA:
        # optuna wants a single integer seed
        methods["Bayesian (TPE)"] = lambda f, s: bayesian_tpe(f, BOUNDS, n_trials=BUDGET,
                                                              seed=1000 * s + METHOD_ID["Bayesian (TPE)"])
    return methods
