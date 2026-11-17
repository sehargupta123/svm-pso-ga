# CLAUDE.md

## Project decisions

- **Main comparison:** PSO vs GA for tuning C and gamma of our from-scratch RBF SVM. The project title is "SVM Hyperparameter Tuning using PSO and Genetic Algorithm".
- **Baselines (for comparison only, not main methods):** grid search, random search, Bayesian optimisation (optuna TPE), and Simulated Annealing (SA).
- **Why SA is kept as an extra baseline:**
  1. It is the third metaheuristic in Unit 5 of the course syllabus, alongside GA and PSO.
  2. It is a single-point search, unlike the population-based PSO and GA, so it tests whether a population is actually needed on this problem.
  3. If SA does about as well as PSO, that supports the No Free Lunch argument that this landscape is easy (one large region of good settings), not that PSO is always best.
- **Pre-revision SA result (indicative only):** in a pre-revision test (10 seeds, common starting points), SA reached the best CV level in 9/10 runs with a median of ~35 trials, similar to PSO. The post-revision step6 results supersede it.
- **Do not remove SA, and do not present it as a main method.**
- In tables and plots, list PSO and GA first, then the baselines, with SA last.
