# STEP 5: Tune our SVM with all four methods and compare them.
# Every method gets the same budget: 100 trials (100 calls to the fitness function).
# Uses OUR OWN SVM, so this takes a few minutes. Be patient!
# Run it with:  python step5_run_tuning.py

import time
import numpy as np

from data import load_data
from fitness import Fitness, BOUNDS
from pso import pso
from ga import ga
from baselines import grid_search, random_search
from helpers import test_accuracy

X_train, X_test, y_train, y_test = load_data()
fitness = Fitness(X_train, y_train, use_my_svm=True)

methods = {
    "Grid search":   lambda f: grid_search(f, BOUNDS),
    "Random search": lambda f: random_search(f, BOUNDS, seed=0),
    "PSO":           lambda f: pso(f, BOUNDS, n_particles=10, n_iters=10, seed=0),
    "GA":            lambda f: ga(f, BOUNDS, pop_size=10, max_evals=100, seed=0),
}

print(f"{'Method':<14} {'best C':>9} {'best gamma':>11} {'CV acc':>7} {'test acc':>9} {'found at trial':>15} {'time (s)':>9}")
print("-" * 80)
for name, run in methods.items():
    fitness.reset_history()
    start = time.time()
    best_x, best_cv = run(fitness.evaluate)
    seconds = time.time() - start
    found_at = int(np.argmax(fitness.history)) + 1   # first trial that reached this method's best score
    test_acc = test_accuracy(X_train, y_train, X_test, y_test, *best_x)
    C, gamma = 10 ** best_x[0], 10 ** best_x[1]
    print(f"{name:<14} {C:>9.3f} {gamma:>11.5f} {best_cv:>7.4f} {test_acc:>9.4f} "
          f"{found_at:>15} {seconds:>9.1f}")