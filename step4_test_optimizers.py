# STEP 4: Test PSO, GA and SA on a simple maths function before using them on the SVM.
# The "sphere" function -(x^2 + y^2) has its best value 0 at the point (0, 0).
# If the optimizers land near (0, 0), they work.
# Run it with:  python step4_test_optimizers.py

import numpy as np
from pso import pso
from ga import ga
from sa import simulated_annealing


def sphere(x, y):
    return -(x ** 2 + y ** 2)          # negative because our optimizers MAXIMISE


bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])

best_x, best = pso(sphere, bounds, n_particles=20, n_iters=50, seed=1)
print(f"PSO found x = {best_x.round(4)}, value = {best:.6f}   (perfect answer: [0, 0], 0)")

best_x, best = ga(sphere, bounds, pop_size=20, max_evals=1000, seed=1)
print(f"GA  found x = {best_x.round(4)}, value = {best:.6f}   (perfect answer: [0, 0], 0)")

# The sphere scores range over ~50 points, so SA needs a hotter start than the SVM default (0.02).
best_x, best = simulated_annealing(sphere, bounds, max_evals=1000, start_temp=5.0, cooling=0.99, seed=1)
print(f"SA  found x = {best_x.round(4)}, value = {best:.6f}   (perfect answer: [0, 0], 0)")

# GA must never score the same point twice: check that on a counting wrapper.
seen = []
ga(lambda x, y: (seen.append((round(x, 4), round(y, 4))), sphere(x, y))[1], bounds,
   pop_size=10, max_evals=100, seed=1)
print(f"\nGA budget check: {len(seen)} evaluations, {len(set(seen))} different points (must be equal).")
