# STEP 4: Test PSO and GA on a simple maths function before using them on the SVM.
# The "sphere" function -(x^2 + y^2) has its best value 0 at the point (0, 0).
# If PSO and GA land near (0, 0), they work.
# Run it with:  python step4_test_optimizers.py

import numpy as np
from pso import pso
from ga import ga


def sphere(x, y):
    return -(x ** 2 + y ** 2)          # negative because our optimizers MAXIMISE


bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])

best_x, best = pso(sphere, bounds, n_particles=20, n_iters=50, seed=1)
print(f"PSO found x = {best_x.round(4)}, value = {best:.6f}   (perfect answer: [0, 0], 0)")

best_x, best = ga(sphere, bounds, pop_size=20, max_evals=1000, seed=1)
print(f"GA  found x = {best_x.round(4)}, value = {best:.6f}   (perfect answer: [0, 0], 0)")