# sa.py
# Simulated Annealing, written from scratch.
#
# One single "walker" moves around the search space (log_C, log_gamma):
#   1. Propose a nearby point: current + Gaussian noise (step = 15% of the range).
#   2. If it scores better, always move there.
#   3. If it scores worse, still move there with probability exp((new - old) / T).
#      A high temperature T accepts bad moves often (explores); a low T almost never (refines).
#   4. After every step, cool down:  T = T * cooling.
# It uses exactly max_evals scored points (1 start point + max_evals - 1 proposals).
# We MAXIMISE the score (accuracy). The best point ever seen is returned.

import numpy as np


def simulated_annealing(score_fn, bounds, max_evals=100, start_temp=0.02, cooling=0.95,
                        step_fraction=0.15, seed=0):
    rng = np.random.default_rng(seed)
    low, high = bounds[:, 0], bounds[:, 1]
    step = step_fraction * (high - low)

    current = rng.uniform(low, high)
    current_score = score_fn(*current)
    best_x, best_score = current.copy(), current_score
    temp = start_temp

    for _ in range(max_evals - 1):
        candidate = np.clip(current + rng.normal(0, step), low, high)
        cand_score = score_fn(*candidate)
        delta = cand_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / temp):
            current, current_score = candidate, cand_score
        if current_score > best_score:
            best_x, best_score = current.copy(), current_score
        temp *= cooling

    return best_x, best_score
