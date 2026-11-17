# STEP 5: Tune our SVM with all five methods and compare them.
# Every method gets the same budget: 100 trials (100 calls to the fitness function).
# Uses OUR OWN SVM, so this takes a few minutes. Be patient!
# Run it with:  python step5_run_tuning.py

import time
import numpy as np

import data
import fitness as fitness_module
from data import load_data
from fitness import Fitness
from tuners import make_methods
from helpers import test_report

X_train, X_test, y_train, y_test = load_data()
fitness = Fitness(X_train, y_train, use_my_svm=True)
print(f"Dataset: {data.DATASET} | scoring: {fitness_module.SCORING} | CV repeats: {fitness_module.N_REPEATS} "
      f"| tie-break on support vectors: {fitness_module.TIE_BREAK_SV}\n")

methods = make_methods(include_bayesian=False)      # the Bayesian baseline is part of step 6

print(f"{'Method':<14} {'best C':>9} {'best gamma':>11} {'CV score':>9} {'test acc':>9} {'found at trial':>15} "
      f"{'repeats':>8} {'time (s)':>9}")
print("-" * 92)
reports = {}
for name, run in methods.items():
    fitness.reset_history()
    start = time.time()
    best_x, _ = run(fitness.evaluate, 0)
    seconds = time.time() - start
    found_at = int(np.argmax(fitness.history)) + 1   # first trial that reached this method's best score
    cv_score = fitness.lookup(*best_x)[0]            # CV score of the final point (without tie-break penalty)
    rep = test_report(X_train, y_train, X_test, y_test, *best_x, use_my_svm=True)
    reports[name] = rep
    C, gamma = 10 ** best_x[0], 10 ** best_x[1]
    print(f"{name:<14} {C:>9.3f} {gamma:>11.5f} {cv_score:>9.4f} {rep['acc']:>9.4f} "
          f"{found_at:>15} {fitness.n_repeated():>8} {seconds:>9.1f}")

print("\nFinal model on the TEST set (malignant = positive class):")
print(f"{'Method':<14} {'malignant caught':>17} {'malignant MISSED':>17} {'false alarms':>13} {'benign correct':>15} {'malignant recall':>17}")
print("-" * 98)
for name, r in reports.items():
    print(f"{name:<14} {r['tp']:>17} {r['fn']:>17} {r['fp']:>13} {r['tn']:>15} {r['recall']:>17.4f}")
print("\n'repeats' = how many of the 100 trials re-scored a point that was already tried (wasted trials).")
