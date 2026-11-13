# pso.py
# Particle Swarm Optimization, written from scratch.
#
# Each particle is one guess (log_C, log_gamma). Every iteration each particle moves using:
#   v = w*v + c1*r1*(personal_best - x) + c2*r2*(global_best - x)
#   x = x + v
#   w  = inertia   (keep going the same way)
#   c1 = memory    (pull back toward my own best spot)
#   c2 = social    (pull toward the swarm's best spot)
# We MAXIMISE the score (accuracy).

import numpy as np


def pso(score_fn, bounds, n_particles=10, n_iters=10, w=0.7, c1=1.5, c2=1.5, seed=0):
    rng = np.random.default_rng(seed)
    low, high = bounds[:, 0], bounds[:, 1]
    dim = len(bounds)
    v_max = 0.2 * (high - low)              # cap on step size: 20% of the range

    # 1. Start particles at random positions with small random velocities.
    x = rng.uniform(low, high, size=(n_particles, dim))
    v = rng.uniform(-v_max, v_max, size=(n_particles, dim))

    # 2. Score the starting positions.
    scores = np.array([score_fn(*p) for p in x])
    pbest_x, pbest_score = x.copy(), scores.copy()
    g = np.argmax(scores)
    gbest_x, gbest_score = x[g].copy(), scores[g]

    # 3. Move the swarm (the first round of scoring already counts as one iteration).
    for _ in range(n_iters - 1):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        v = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x)
        v = np.clip(v, -v_max, v_max)
        x = np.clip(x + v, low, high)       # stay inside the search space

        scores = np.array([score_fn(*p) for p in x])

        # Update each particle's personal best.
        improved = scores > pbest_score
        pbest_x[improved] = x[improved]
        pbest_score[improved] = scores[improved]

        # Update the swarm's global best.
        g = np.argmax(pbest_score)
        if pbest_score[g] > gbest_score:
            gbest_x, gbest_score = pbest_x[g].copy(), pbest_score[g]

    return gbest_x, gbest_score