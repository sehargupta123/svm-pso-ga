# CHANGES (branch `revisions`)

## Part 1: sanity check - problems found and fixed

Every file was read. Things that were checked and found **correct**: dual QP mapping to cvxopt
(`P = yy'K`, `q = -1`, `G/h` box constraints, `A = y'`, `b = 0`), the RBF kernel, the SVM decision
function, the bias formula, scaler fitted inside each CV fold only (no leakage), same folds for every
(C, gamma), equal budgets of 100 evaluations for grid / random / PSO / GA, PSO's 10 x 10 count.

| # | File | Problem | Fix |
|---|------|---------|-----|
| 1 | `my_svm.py` | **KKT / support-vector tolerances were absolute (1e-5) but cvxopt is only accurate relative to C.** With C = 100 about 3 extra "support vectors" (alpha ~ 1e-4) were counted, and bounded SVs sitting at C - 1e-4 were treated as margin SVs when averaging the bias. KKT pass rate was 95.4 % at (C=100, gamma=0.001). | Tolerances now scale with C (`alpha > 1e-5*max(C,1)` is a SV; `alpha > C - 1e-3*C` is "at C"). KKT now 98-100 % and SV counts match scikit-learn within a few. |
| 2 | `my_svm.py` | cvxopt failure (status not optimal) was silently used. | Raises a clear error if cvxopt returns no solution. |
| 3 | `step5/6` | **Seeding:** PSO, GA and random search all used `default_rng(seed)` with the same seed, so for the same seed they started from the *same 10 random points*. The runs were not independent, which biases the comparison. | Each method now gets its own stream `seed=[run_seed, method_id]` (see `tuners.py`). **Numbers from earlier runs will change** (trial counts especially). |
| 4 | `step6` | Hard-coded "10 runs each" and "/10 runs" in the plot text. | Uses the real seed count. |
| 5 | `step6` | If grid search never reached the target, `axhline(nan)` was drawn / annotation printed "nan trials". | Handled: prints "never reached". Empty boxes (a method that never reached the target) no longer crash `boxplot`. |
| 6 | `step6`, `step7` | Hard-coded plot limits (`ylim 90-99`, `xlim 20-260`, `vmin=60`) that clip data for other scoring options / datasets. | Limits computed from the data. |
| 7 | `step6` | CV used scikit-learn's SVC (default) but the final **test** model was always `MySVM`, so the "CV vs test" columns came from different SVM implementations. | `test_report(..., use_my_svm=USE_MY_SVM)` uses the same SVM as the CV. |
| 8 | `step2`, `step3` | Used the **test set** for demos / SVM comparison. Not used for selection, but the project rule is that the test set is only touched in `helpers`. | Both now use a validation split carved out of the training data. Only `helpers.test_report/test_accuracy` (and `test_accuracy_model`, step 9's twin) touch the test set. |
| 9 | all | Output folders were relative to the current directory (`results/`, `figures/`), so running from another folder wrote elsewhere (Windows path issue). | Paths are built from the script's own folder (`helpers.RESULTS_DIR`, `FIGURES_DIR`). |
| 10 | `data.py` | Comment claimed "the same 80/20 split" as the reference repo, which is not verified. | Comment now only says the dataset is loaded the same way. |
| 11 | `fitness.py` | History / cache mixed "the score" and "accuracy" under the name accuracy. | Cache stores the plain CV score + support vectors; history stores what the optimiser saw. Reporting uses `Fitness.lookup`. |
| 12 | `step1` | Did not check scipy (needed from step 6). | Checks scipy, and reports optuna / pymoo as optional. |

**Known and left as-is (documented, not bugs):**
* PSO clips particles at the border of the box, so a few of its 100 trials can repeat a boundary point (the task only asked to remove GA repeats). `step5` prints a `repeats` column for every method.
* SA can also revisit a point after a rejected move that is clipped to the same border value; counted in `repeats`.
* The `Fitness` cache is shared between methods inside one script, so wall-clock times in step 5 favour methods run later. Budgets (number of trials) are unaffected.
* `step3` KKT for the simplified SMO solver is lower (88-100 %) than QP: simplified SMO stops early by design.

## Part 2: revisions implemented

1. **GA no repeats** (`ga.py`): set of scored points (rounded to 4 decimals); a repeated child is nudged with Gaussian noise (up to 20 tries, then a fresh random point). Exactly 100 *unique* evaluations. `step4` checks this.
2. **step6**: `N_SEEDS = 30`, `QUICK = False` (True = 3 seeds + 10 x 10 landscape). Mann-Whitney U (PSO vs Random, GA vs Random, PSO vs GA, SA vs Random, plus TPE vs Random when optuna is present), "never reached" = 101, saved to `results/significance.csv`.
3. **Simulated annealing** (`sa.py`): start_temp 0.02, cooling 0.95, step 15 % of range, exactly 100 evaluations. In step5, step6 (colours, box plot, convergence, search-space plot, tables).
4. **Tie-break** (`fitness.py`, `TIE_BREAK_SV = True`): `score - 1e-6 * avg_SVs`, same value in `history`. The comment explains why it cannot beat a one-patient difference (max penalty 3.6e-4 < 1/455 = 2.2e-3).
5. **Scoring** (`SCORING`: accuracy / f1_malignant / recall_malignant). Confusion matrix and malignant recall in step5 and step6 (`all_runs.csv`, `summary.csv`).
6. **Repeated CV** (`N_REPEATS`, default 1 = identical to the old folds).
7. **step8_sensitivity.py** (PSO w, PSO c1 = c2, GA mutation rate; 10 seeds; SVC; table + one 3-panel plot; QUICK).
8. **Datasets** (`data.py`: `DATASET = "breast_cancer" | "sonar" | "ionosphere"`, OpenML download cached in `data/`, clear error offline, labels -1/+1: Mine and "bad" radar return = -1).
9. **step9_kernel_search.py** (GA over kernel, degree, log C, log gamma with SVC; QUICK).
10. **step10_nsga2.py** (pymoo NSGA-II, accuracy vs support vectors, compared with step 7's front, which is now saved to `results/pareto_step7.csv`). Import wrapped with an install hint.
11. **Bayesian optimisation** (`baselines.bayesian_tpe`, optuna TPE, 100 trials) included in step6 if optuna is installed.
12. **SMO solver** (`MySVM(solver="smo")`, Platt's simplified SMO; default stays `"qp"`). `step3` Part B compares QP vs SMO vs scikit-learn.

New small files: `sa.py`, `tuners.py` (single list of methods and seeds shared by step5/step6), `step8`-`step10`.
`requirements.txt` now lists scipy, optuna, pymoo.
