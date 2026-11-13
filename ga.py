# ga.py
# Genetic Algorithm, written from scratch.
#
# Each individual is one guess (log_C, log_gamma). Every generation:
#   1. Elitism:   copy the best individuals unchanged
#   2. Selection: pick parents with "tournaments" (best of 3 random individuals)
#   3. Crossover: mix two parents into a child
#   4. Mutation:  add small random noise to some children
# It stops after exactly max_evals scored guesses, so it gets the same budget as the others.
# We MAXIMISE the score (accuracy).

import numpy as np


def ga(score_fn, bounds, pop_size=10, max_evals=100, crossover_rate=0.8,
       mutation_rate=0.15, n_elite=2, seed=0):
    rng = np.random.default_rng(seed)
    low, high = bounds[:, 0], bounds[:, 1]
    dim = len(bounds)
    sigma = 0.1 * (high - low)              # size of a mutation: 10% of the range

    # 1. Random starting population, and score it.
    pop = rng.uniform(low, high, size=(pop_size, dim))
    scores = np.array([score_fn(*ind) for ind in pop])
    evals = pop_size

    def tournament():
        picks = rng.choice(pop_size, size=3, replace=False)
        return pop[picks[np.argmax(scores[picks])]]

    # 2. Evolve until the budget is used up.
    while evals < max_evals:
        order = np.argsort(scores)[::-1]                 # best first
        new_pop = [pop[i].copy() for i in order[:n_elite]]
        new_scores = [scores[i] for i in order[:n_elite]]   # elites keep their known score

        while len(new_pop) < pop_size and evals < max_evals:
            parent1, parent2 = tournament(), tournament()
            if rng.random() < crossover_rate:
                mix = rng.random(dim)                    # blend crossover
                child = parent1 + mix * (parent2 - parent1)
            else:
                child = parent1.copy()
            if rng.random() < mutation_rate:
                child = child + rng.normal(0, sigma)     # Gaussian mutation
            child = np.clip(child, low, high)
            new_pop.append(child)
            new_scores.append(score_fn(*child))
            evals += 1

        # If the budget ran out mid-generation, fill the empty slots with the best old individuals.
        i = n_elite
        while len(new_pop) < pop_size:
            new_pop.append(pop[order[i]].copy())
            new_scores.append(scores[order[i]])
            i += 1

        pop, scores = np.array(new_pop), np.array(new_scores)

    best = np.argmax(scores)
    return pop[best], scores[best]