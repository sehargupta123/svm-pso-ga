# SVM Hyperparameter Tuning using PSO and Genetic Algorithm

*Optimization Techniques in Engineering (EEE1001), 2-person project.*

We train a Support Vector Machine (SVM) **written from scratch** on the UCI Breast Cancer Wisconsin
(Diagnostic) dataset. The main comparison is **Particle Swarm Optimization (PSO) vs a Genetic Algorithm (GA)**
for tuning the two hyperparameters `C` and `gamma`; both are written from scratch. We compare them with
baselines: **grid search**, **random search**, **Bayesian optimisation (TPE)** and **Simulated Annealing (SA)**.

Fair comparison rule: **every method gets exactly 100 evaluations**. One evaluation = train the SVM with
one `(C, gamma)` and measure its 5-fold stratified cross-validation (CV) score on the training data.

---

## 1. Project summary and results

* Data: 569 patients, 30 measurements each, 357 benign (+1) and 212 malignant (-1). 80 % train (455), 20 % test (114).
* Search space: $\log_{10} C \in [-2, 3]$ and $\log_{10}\gamma \in [-4, 1]$.
* Our own SVM: dual QP solved with `cvxopt` (also a hand-written SMO solver), RBF kernel, KKT check.

**Single run (seed 0, `step5_run_tuning.py`)**

| Method | best C | best gamma | CV accuracy | test accuracy | found at trial | repeated trials |
|---|---|---|---|---|---|---|
| PSO | 59.170 | 0.00053 | 0.9802 | 0.9825 | 76 | 0 |
| GA | 32.959 | 0.00098 | 0.9802 | 0.9825 | 23 | 0 |
| Grid search | 21.544 | 0.00129 | 0.9802 | 0.9825 | 63 | 0 |
| Random search | 6.879 | 0.00239 | 0.9780 | 0.9825 | 38 | 0 |
| SA | 31.953 | 0.00078 | 0.9802 | 0.9825 | 38 | 1 |

(One seed only: "found at trial" varies a lot between seeds, e.g. PSO's 76 is an unlucky run. See section 9.)

**Many seeds (`step6_experiments.py`, 30 seeds)**

| Method | mean CV score | mean test acc | malignant recall | mean trials to target | runs reaching target |
|---|---|---|---|---|---|
| PSO | 0.9800 | 0.9813 | 0.9762 | 36.7 | 28 / 30 |
| GA | 0.9796 | 0.9781 | 0.9762 | 37.6 | 22 / 30 |
| Grid search | 0.9802 | 0.9825 | 0.9762 | 63.0 | 1 / 1 |
| Random search | 0.9788 | 0.9792 | 0.9762 | 54.6 | 15 / 30 |
| Bayesian (TPE) | 0.9802 | 0.9810 | 0.9762 | 41.5 | 30 / 30 |
| SA | 0.9797 | 0.9813 | 0.9762 | 40.0 | 24 / 30 |

(Mean trials to target counts only the runs that reached the target. Target CV score = 0.9792.)

**Significance (Mann-Whitney U on trials-to-target, `results/significance.csv`)**

| Comparison | p-value |
|---|---|
| PSO vs Random | 0.0001 (significant) |
| GA vs Random | 0.0195 (significant) |
| PSO vs GA | 0.394 (not significant) |
| SA vs Random | 0.0065 (significant) |
| Bayesian (TPE) vs Random | < 0.0001 (significant) |

Plots are written to `figures/` (convergence, trials-to-target box plot, search-space map, Pareto fronts, sensitivity).

---

## 2. What is an SVM?

### 2.1 The maximum-margin idea
We have points $x_i$ with labels $y_i \in \{-1, +1\}$. Many straight lines (hyperplanes
$w^\top x + b = 0$) can separate the two classes. The SVM picks the one with the **largest margin**: the
widest empty street between the classes. The street has width $2/\lVert w\rVert$, so *maximising the margin
means minimising* $\tfrac12\lVert w\rVert^2$. A wide margin generalises better to new patients.

### 2.2 Hard vs soft margin
* **Hard margin**: no point may be inside the street or on the wrong side: $y_i(w^\top x_i + b)\ge 1$. This only works if the classes are perfectly separable, and one outlier can wreck it.
* **Soft margin**: we allow violations, measured by slack variables $\xi_i \ge 0$, and we pay a price $C$ for them.

### 2.3 The soft-margin primal problem
With a feature map $\phi$ (the kernel trick, section 3):

$$
\min_{w,b,\xi}\ \ \tfrac12\lVert w\rVert^2 + C\sum_{i=1}^{n}\xi_i
\quad\text{s.t.}\quad y_i\big(w^\top\phi(x_i)+b\big)\ \ge\ 1-\xi_i,\qquad \xi_i\ge 0 .
$$

### 2.4 Lagrangian and KKT conditions
Introduce multipliers $\alpha_i\ge0$ (for the margin constraints) and $\mu_i\ge0$ (for $\xi_i\ge0$):

$$
\mathcal L = \tfrac12\lVert w\rVert^2 + C\sum_i\xi_i
- \sum_i \alpha_i\big[y_i(w^\top\phi(x_i)+b)-1+\xi_i\big] - \sum_i\mu_i\xi_i .
$$

Setting the derivatives to zero (stationarity):

$$
\frac{\partial\mathcal L}{\partial w}=0\Rightarrow w=\sum_i\alpha_i y_i\phi(x_i),\qquad
\frac{\partial\mathcal L}{\partial b}=0\Rightarrow\sum_i\alpha_i y_i=0,\qquad
\frac{\partial\mathcal L}{\partial \xi_i}=0\Rightarrow C-\alpha_i-\mu_i=0 .
$$

The full **KKT conditions** are: stationarity (above), primal feasibility, dual feasibility
($\alpha_i,\mu_i\ge0$) and **complementary slackness**:

$$
\alpha_i\big[y_i f(x_i)-1+\xi_i\big]=0,\qquad \mu_i\xi_i=0,\qquad f(x)=w^\top\phi(x)+b .
$$

### 2.5 The dual problem (what our code solves)
Substituting back removes $w$, $b$ and $\xi$:

$$
\max_{\alpha}\ \ \sum_{i}\alpha_i-\tfrac12\sum_i\sum_j\alpha_i\alpha_j y_iy_j K(x_i,x_j)
\quad\text{s.t.}\quad 0\le\alpha_i\le C,\quad \sum_i\alpha_i y_i=0 .
$$

This is a quadratic program. To give it to `cvxopt` (which *minimises* $\tfrac12\alpha^\top P\alpha+q^\top\alpha$ with
$G\alpha\le h$, $A\alpha=b$) we use $P_{ij}=y_iy_jK(x_i,x_j)$, $q=-\mathbf 1$, $G=\begin{bmatrix}-I\\ I\end{bmatrix}$,
$h=\begin{bmatrix}\mathbf 0\\ C\mathbf 1\end{bmatrix}$, $A=y^\top$, $b=0$.

### 2.6 Support vectors: the three $\alpha$ cases
From complementary slackness and $C-\alpha_i-\mu_i=0$:

| $\alpha_i$ | where the point is | condition |
|---|---|---|
| $\alpha_i=0$ | correct side, outside the margin (not a support vector) | $y_if(x_i)\ge1$ |
| $0<\alpha_i<C$ | exactly **on the margin** (free support vector) | $y_if(x_i)=1$ |
| $\alpha_i=C$ | **inside the margin or misclassified** (bounded support vector) | $y_if(x_i)\le1$ |

Only points with $\alpha_i>0$ (support vectors) matter for predictions. `MySVM.check_kkt()` verifies these three rows on the training set.

### 2.7 Decision function and the bias
$$
f(x)=\sum_{i\in SV}\alpha_i y_i K(x_i,x)+b,\qquad \hat y=\operatorname{sign}f(x).
$$
For a free support vector $s$ ($0<\alpha_s<C$) we know $y_sf(x_s)=1$, hence
$b=y_s-\sum_j\alpha_jy_jK(x_j,x_s)$. Numerically we **average this over all free support vectors**
(if there are none, over all support vectors).

---

## 3. The RBF kernel, `gamma` and `C`

$$K(x,z)=\exp\!\big(-\gamma\lVert x-z\rVert^2\big)$$

It measures similarity: 1 for identical points, falling to 0 for far-apart points.

* **gamma** = how *local* the similarity is. Small gamma: each point influences a wide area, the boundary is smooth (can **underfit**). Large gamma: each point only influences its tiny neighbourhood, the boundary becomes a set of islands around training points (can **overfit**).
* **C** = how much we punish mistakes. Small C: tolerate mistakes, wide margin, simple model (**underfit**). Large C: almost no mistakes allowed on training data, a wiggly boundary (**overfit**).
* **They interact.** A large gamma with a large C overfits badly. A small gamma needs a large C to bend enough (the good region is a diagonal band in the $(\log C,\log\gamma)$ plane: "bigger C, smaller gamma" gives similar models). That is why tuning them one at a time is not enough.
* **Why log10 scale?** Both parameters matter by their *order of magnitude*: going from 0.01 to 0.1 changes the model as much as going from 100 to 1000. Searching $\log_{10}$ values uniformly gives every order of magnitude equal attention; searching $C$ linearly in $[0.01,1000]$ would spend 99.9 % of the points above $C=1$.

---

## 4. Grid search and random search (the two baselines)

### 4.1 Grid search
**What it is.** Choose a fixed list of values for each hyperparameter and try **every combination**.

**Our version.** A $10\times10$ lattice over $\log_{10}C\in[-2,3]$ and $\log_{10}\gamma\in[-4,1]$ (`numpy.linspace`,
10 evenly spaced values per axis) = **100 trials**, scanned row by row: the outer loop goes through $\log C$ from small to large, the inner loop
through $\log\gamma$ from small to large. The best score wins (ties go to the earlier point).

**Cost.** With $m$ values per axis and $d$ hyperparameters:

$$\text{trials}=m^{d}$$

2 hyperparameters x 10 values = 100 trials, 5 x 10 = 100 000 trials. This explosion is the **curse of dimensionality**.

**Strengths.** Simple; deterministic and reproducible; trivially parallel; guaranteed to cover the whole box evenly.

**Weaknesses.**
* Wastes trials on bad regions (it cannot learn from earlier trials).
* Resolution is limited by the grid spacing, so the true optimum between grid points can be missed (our spacing is 0.56 in $\log C$ and 0.56 in $\log\gamma$).
* The result depends on the scan order (ties) and on the chosen bounds.
* Scales badly with more hyperparameters.

scikit-learn does the same with `GridSearchCV` (the reference repo uses it). Ours is the same idea written by hand
so that every method uses the same fitness function and the same folds.

### 4.2 Random search
**What it is.** Sample each trial **uniformly at random** inside the bounds.

**Our version.** 100 uniform samples of $(\log C,\log\gamma)$ in the box, seeded for reproducibility.

**Why it often beats grid search.** Bergstra & Bengio (2012): when only *some* hyperparameters matter much, a grid of
$m\times m$ points tries only $m$ different values of each parameter (100 trials = 10 different values of the important one), but 100 random points try **100 different values** of each. With a fixed budget random search therefore explores each individual hyperparameter much more finely.

**Strengths.** Simple; no assumptions; trivially parallel; any budget works (not just $m^d$); easy to add more trials later.

**Weaknesses.** Does not learn from past trials; cannot refine around a good point; results change from run to run, so it needs **several seeds** to be judged fairly.

scikit-learn: `RandomizedSearchCV`.

### 4.3 Comparison table

| | Grid | Random | PSO | GA | SA |
|---|---|---|---|---|---|
| Uses past results? | no | no | **yes** | **yes** | **yes** |
| Deterministic? | yes | no (seeded) | no (seeded) | no (seeded) | no (seeded) |
| Number of settings | 2 (points per axis, bounds) | 1-2 (budget, seed) | 5 (swarm size, iters, w, c1, c2) | 5-6 (pop size, crossover, mutation, elites, tournament) | 3 (start temp, cooling, step size) |
| Main weakness | cost explodes with dimensions, fixed resolution | no learning | can converge early to a local optimum | needs population and tuning | single walker, sensitive to cooling schedule |

### 4.4 This project's numbers

| Method | trial at which the best accuracy was found (seed 0) | seeds that reached the target |
|---|---|---|
| PSO | 76 | 28 / 30 |
| GA | 23 | 22 / 30 |
| Grid | 63 | 1 / 1 |
| Random | 38 | 15 / 30 |
| SA | 38 | 24 / 30 |

"Target" = within 0.1 percentage points of the best CV score any method found. "Never reached" counts as 101 trials in the statistical tests.

### 4.5 Sketch: grid points vs random points in the $(\log C,\log\gamma)$ box

```
 log gamma                                    log gamma
  +1 |. . . . . . . . . .|                     +1 |  .     .   .      . |
     |. . . . . . . . . .|                        | .    .        .   . |
     |. . . . . . . . . .|                        |    .   .  .         |
     |. . . . . . . . . .|                        | .         .    .  . |
     |. . . . . . . . . .|                        |      .  .       .   |
     |. . . . . . . . . .|                        | .  .       .      . |
     |. . . . . . . . . .|                        |   .      .    .     |
     |. . . . . . . . . .|                        |.      .   .      .  |
     |. . . . . . . . . .|                        |  .  .        .    . |
  -4 |. . . . . . . . . .|                     -4 |.    .    .       .  |
      -2               +3  log C                    -2               +3  log C
        GRID (10 x 10)                                RANDOM (100 points)
   only 10 distinct log C values                 about 100 distinct log C values
   and 10 distinct log gamma values              and 100 distinct log gamma values
```

---

## 5. Other classical hyperparameter tuning methods

* **Manual tuning**: an expert tries values by hand and reasons about the results. Cheap for one or two parameters, not reproducible, depends on experience.
* **Cross-validation (CV)**: splits the training data into $k$ folds, trains on $k-1$ and validates on the remaining one, and averages. We use 5-fold stratified CV (each fold keeps the class ratio). The scaler is fitted inside each fold, otherwise validation data would leak into training.
* **Nested cross-validation**: the tuning itself runs inside an *inner* CV loop on each training part of an *outer* CV loop, and the outer folds estimate the final accuracy. This gives an almost unbiased estimate when the tuned score and the reported score would otherwise come from the same folds. We use a simpler, also valid scheme: tune with CV on the training set only and report accuracy once on a **separate test set**.
* **Bayesian optimisation**: builds a cheap *surrogate model* of "score as a function of the hyperparameters" and uses it to choose the next, most promising point. **Gaussian processes** (Snoek et al. 2012) model it directly; **TPE** (Tree-structured Parzen Estimator, Bergstra et al. 2011) models where good and bad trials lie. We use TPE through `optuna`.
* **Successive halving / Hyperband** (Jamieson & Talwalkar 2016; Li et al. 2017): start many configurations with a small resource (few data, few iterations), repeatedly drop the worst half, give survivors more. Excellent when training is expensive; an SVM on 455 points is cheap, so there is little to save.
* **Gradient-based tuning**: differentiate the validation loss with respect to the hyperparameters. It needs the whole pipeline to be differentiable; here the CV *accuracy* is a step function (piecewise constant) of $C$ and $\gamma$, so its gradient is zero almost everywhere. Smooth surrogates exist (e.g. validation loss through implicit differentiation), but they are not what this project is about.

---

## 6. How the three metaheuristics work

### 6.1 Genetic Algorithm (GA)
Inspired by evolution. Our settings: population 10, 100 evaluations (about 11 generations of 8 new children).
* **Encoding**: an individual is the real-number pair $(\log_{10}C,\log_{10}\gamma)$.
* **Selection**: *tournament of 3*: pick 3 random individuals, the best becomes a parent.
* **Crossover** (probability 0.8): blend $\;child=p_1+r\,(p_2-p_1)$ with a random $r\in[0,1]$ per gene; otherwise the child copies $p_1$.
* **Mutation** (probability 0.15 per child): add Gaussian noise with standard deviation 10 % of the range of each gene.
* **Elitism**: the best 2 individuals go to the next generation unchanged (and keep their known score).
* **No repeats**: if a child equals an already-scored point (rounded to 4 decimals), it is nudged with Gaussian noise, up to 20 tries, so all 100 evaluations are new points.
* Results are clipped into the search box.

### 6.2 Particle Swarm Optimization (PSO)
Inspired by flocks of birds. Our settings: 10 particles, 10 iterations (= 100 evaluations), $w=0.7$, $c_1=c_2=1.5$.
Each particle has a position $x$ (a guess) and a velocity $v$, and remembers its own best position $p$; the swarm remembers the best position $g$:

$$v\leftarrow w\,v+c_1r_1\,(p-x)+c_2r_2\,(g-x),\qquad x\leftarrow x+v$$

with random numbers $r_1,r_2\in[0,1]$ (new for each particle and each dimension).
* $w$ (inertia): keep moving the same way. High $w$ explores, low $w$ settles.
* $c_1$ (cognitive): pull toward the particle's own best.
* $c_2$ (social): pull toward the swarm's best.
* Velocity is capped at 20 % of the range and positions are clipped to the box.

### 6.3 Simulated Annealing (SA, extra baseline)
Inspired by slowly cooling metal (Kirkpatrick et al. 1983). One walker:
1. Propose $x'=x+\mathcal N(0,\,(0.15\cdot\text{range})^2)$.
2. If $f(x')>f(x)$ move there. Otherwise move there anyway with probability $\exp\big((f(x')-f(x))/T\big)$.
3. Cool: $T\leftarrow 0.95\,T$.

Start temperature 0.02 (accuracy is on a 0-1 scale, so $T=0.02$ accepts a 1-percentage-point loss with probability $e^{-0.5}\approx0.6$ at the start, and almost never after about 60 steps). Exactly 100 evaluations; the best point ever visited is returned.

**Why SA is here.** SA is an extra baseline, not a main method. It is the third metaheuristic in Unit 5 of the syllabus, next to GA and PSO. It is also a *single-point* search, unlike the population-based PSO and GA, so it tests whether a population is really needed on this problem. If SA does about as well as PSO, that supports the No Free Lunch argument that this landscape is easy (one large region of good settings), not that PSO is always best.

---

## 7. Benchmarks for comparison

**(a) Optimizer benchmarks on test functions.** Optimisers are usually first tested on functions whose best point is known:
*Sphere* $\sum x_i^2$ (easy, one valley), *Rastrigin* (many regular local minima), *Rosenbrock* (long curved narrow valley),
*Ackley* (many small bumps on a big funnel). Standard collections are the **CEC benchmark suites** (IEEE Congress on Evolutionary Computation competitions). `step4` uses only the sphere, as a smoke test.

**(b) Hyperparameter-tuning baselines in this project.** Grid search, random search and Bayesian optimisation (optuna TPE, 100 trials).

**(c) Evaluation metrics used here.**
* *Best CV accuracy*: the score used for selection (optionally F1 or recall of the malignant class).
* *Test accuracy*: measured once, on data never used for tuning (plus confusion matrix and malignant recall).
* *Trials-to-target*: how many evaluations until the best-so-far score is within 0.1 percentage points of the best score found by any method.
* *Success rate over seeds*: how many of the N seeds reached the target within 100 trials.
* *Mann-Whitney U test*: a rank test (no normal-distribution assumption) asking whether one method's trials-to-target tends to be smaller than another's. $p<0.05$ means the difference is unlikely to be luck.

**(d) No Free Lunch theorem.** Wolpert & Macready (1997): averaged over *all possible* problems, every optimisation algorithm performs equally well. Therefore "PSO is better than GA" can only mean "better on this kind of problem". Our results apply to smooth, low-dimensional SVM tuning, not to everything.

**(e) Other standard datasets for SVM tuning comparisons.** UCI **Sonar** (208 samples, 60 features, harder), **Ionosphere** (351, 34), **Pima Indians Diabetes** (768, 8), **Wine** (178, 13), **Iris**, **Heart disease**. Sonar and Ionosphere are built in (set `DATASET` in `data.py`).

---

## 8. Code organisation and how to run

| File | What it does |
|---|---|
| `data.py` | Loads the dataset (`DATASET`: breast_cancer / sonar / ionosphere), maps labels to -1/+1, stratified 80/20 split |
| `my_svm.py` | Our SVM: RBF kernel, dual QP via cvxopt (`solver="qp"`) or Platt's simplified SMO (`solver="smo"`), KKT check |
| `fitness.py` | Cross-validated score of one `(log C, log gamma)`. Settings: `SCORING`, `N_REPEATS`, `TIE_BREAK_SV` |
| `helpers.py` | Test-set evaluation (the only place the test set is used), output folders |
| `pso.py`, `ga.py`, `sa.py` | The three metaheuristics |
| `baselines.py` | Grid search, random search, Bayesian TPE (optuna) |
| `tuners.py` | One list of all methods with their settings and seeds (used by step 5 and 6) |
| `step1_check_setup.py` | Check the libraries |
| `step2_explore_data.py` | Look at the data, effect of scaling |
| `step3_test_my_svm.py` | Our SVM vs scikit-learn; QP vs SMO vs scikit-learn |
| `step4_test_optimizers.py` | PSO/GA/SA on the sphere function |
| `step5_run_tuning.py` | One run of every method with our own SVM, with confusion matrices |
| `step6_experiments.py` | Many seeds, plots, tables, Mann-Whitney tests (`N_SEEDS`, `QUICK`) |
| `step7_pareto.py` | Accuracy vs number of support vectors (Pareto front) |
| `step8_sensitivity.py` | Sensitivity of PSO (w, c1, c2) and GA (mutation rate) |
| `step9_kernel_search.py` | GA also chooses the kernel (linear / poly / rbf) and degree |
| `step10_nsga2.py` | NSGA-II multi-objective search with pymoo |
| `objectives.py` | The objective functions as explicit NumPy code (kernel, $Q$, dual $D$, primal $P$, duality gap, CV accuracy $F$) and SMO with a logged history |
| `step11_objective_convergence.py` | Convergence plots and contour plots for the inner (SVM training) and outer (tuning) problems (`QUICK`) |
| `CHANGES.md` | List of problems found and revisions made |
| `results/`, `figures/` | Output tables and plots |

### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt

python step1_check_setup.py
python step2_explore_data.py
python step3_test_my_svm.py
python step4_test_optimizers.py
python step5_run_tuning.py          # slow: uses our own SVM
python step6_experiments.py         # set QUICK = True inside the file for a fast check
python step7_pareto.py
python step8_sensitivity.py
python step9_kernel_search.py
python step10_nsga2.py
```

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 step1_check_setup.py        # then step2 ... step10 the same way
```

Settings you may change (edit the top of the file): `DATASET` in `data.py`; `SCORING`, `N_REPEATS`, `TIE_BREAK_SV` in `fitness.py`; `USE_MY_SVM`, `N_SEEDS`, `QUICK` in `step6_experiments.py`; `QUICK` in steps 8 and 9.
The tie-break `score - 1e-6 * (average support vectors)` only decides between *exactly equal* accuracies: one patient is worth $1/455=0.0022$, while the largest possible penalty is about $0.0004$.

---

## 9. Results

All numbers come from the log files in `results/` (`step5_log.txt` ... `step10_log.txt`): Breast Cancer Wisconsin dataset, accuracy scoring, 100 evaluations per run.

### 9.1 Main result: PSO vs GA over 30 seeds (step 6)

"Target" = a CV score of **0.9792**, i.e. within 0.1 percentage points of the best score any method found (0.9802).

| Method | Runs that reached the target | Median trials (successful runs only) | p-value vs random search |
|---|---|---|---|
| **PSO** | **28 / 30** | 34.5 | 0.0001 |
| **GA** | **22 / 30** | 31 | 0.0195 |
| Grid search | 1 / 1 (at trial 63) | 63 | not applicable |
| Random search | 15 / 30 | 59 | |
| Bayesian (TPE) | 30 / 30 | 39 | < 0.0001 |
| SA | 24 / 30 | 36.5 | 0.0065 |

Two kinds of median appear in the result files, and they differ on purpose:
* `summary.csv` takes the median **only over the runs that reached the target**. That is why random search shows 59.
* `significance.csv` counts a run that **never reached the target as 101 trials**, so that failures count against a method in the Mann-Whitney test. With 15 of 30 random-search runs failing, its median there is 100.5. In the same way PSO shows 35.5, GA 42.5 and SA 48.5.

What the statistics say:
* PSO, GA, SA and TPE are **all significantly faster than random search** (Mann-Whitney U, two-sided, p = 0.0001, 0.0195, 0.0065 and < 0.0001).
* **PSO vs GA is not significant (p = 0.394).** PSO was more *reliable* (28 vs 22 successful runs), while GA was slightly *faster when it succeeded* (median 31 vs 34.5 trials). With 30 seeds we cannot say that either one wins.
* **Bayesian optimisation (TPE) was the most reliable method: 30 / 30.** As a modern baseline it beat both metaheuristics on reliability. That is an honest result worth stating: a method that learns a model of the score surface is hard to beat on a smooth 2-D problem.
* Repeated trials are negligible: PSO 0.03 and SA 0.87 repeated points per run (GA 0 by design, random 0).

### 9.2 Final models on the test set (step 5 and step 6)

Every method ends with a model of about the same quality. The mean test accuracy over 30 seeds lies between **97.8 % and 98.25 %** (grid 98.25, PSO 98.13, SA 98.13, TPE 98.10, random 97.92, GA 97.81), and **every method's final model misses exactly 1 of the 42 malignant test cases on average (malignant recall 0.9762)**. One test patient is worth 0.88 percentage points, so these differences are one patient or less.
Final accuracy therefore does **not** separate the methods; **speed and reliability** do.

Step 5 (single seed 0, our own SVM) is used only for the test-set confusion matrix: all five methods give 41 malignant caught, 1 missed, 1 false alarm, 71 benign correct, test accuracy 0.9825. Its "found at trial" values (grid 63, random 38, PSO 76, GA 23, SA 38) come from one seed: PSO's 76 is just one unlucky run, not a typical value (its median over 30 seeds is 34.5).

### 9.3 Other experiments

* **Kernel search (step 9, GA over kernel, degree, C and gamma, 5 seeds).** A **linear kernel reaches 98.02 % CV accuracy (C about 0.03), the same as the best RBF model**, and the GA chose linear in 4 of 5 seeds (RBF once; polynomial never won, its best was 97.36 %). The data is nearly linearly separable.
* **Multi-objective (step 10).** NSGA-II with only 300 evaluations produced an 11-point Pareto front that **dominates 7 of the 9 points** of the step 7 front (which was built from 1000 PSO and GA evaluations). Both reach 98.02 % accuracy at best, but the most accurate NSGA-II point uses 51.2 support vectors, while step 7 found 98.02 % with 33.
* **Sensitivity (step 8, 10 seeds per setting).** The defaults ($w=0.7$, $c_1=c_2=1.5$, mutation 0.15) are reasonable but not optimal.

| Setting | Runs reaching the target | Mean trials to target |
|---|---|---|
| PSO w = 0.4 | 7 / 10 | 34.4 |
| PSO w = 0.7 (default) | 9 / 10 | 33.1 |
| PSO w = 0.9 | 9 / 10 | 53.1 (slower) |
| PSO c1 = c2 = 1.0 | 10 / 10 | 47.8 |
| PSO c1 = c2 = 1.5 (default) | 9 / 10 | 33.1 |
| PSO c1 = c2 = 2.0 | 10 / 10 | 35.4 |
| GA mutation 0.05 | 6 / 10 (worst) | 32.7 |
| GA mutation 0.15 (default) | 9 / 10 | 39.3 |
| GA mutation 0.30 | 8 / 10 | 31.0 |

---

## 10. Discussion

* **PSO vs GA.** On this problem they are statistically tied. PSO succeeded more often (28 vs 22 of 30); GA was a little quicker when it did succeed. One plausible reason for PSO's reliability is that its particles are pulled toward the best point found so far, so the swarm settles on the good region, while the GA relies more on random mutation to keep moving (step 8 fits this: the lowest mutation rate, 0.05, was the worst GA setting with 6 / 10). We did not test this explanation directly.
* **Guided search beats random search.** Random search reached the target in only 15 of 30 runs and is significantly slower than all four guided methods. Using the results of earlier trials helps.
* **Why every method ends up equally good.** The score surface has one large region of good settings (a wide band in $(\log C,\log\gamma)$), and step 9 shows that even a **linear** kernel is as good as RBF. When the problem is that easy, any reasonable search finds a good point sooner or later, so the final accuracy is the same for everyone. This agrees with the **No Free Lunch theorem**: no algorithm wins on all problems, and on an easy landscape the differences between algorithms mostly disappear. We must therefore not conclude that PSO (or GA) is "the best optimiser"; we only learned how they behave on this kind of problem.
* **Single-point vs population search.** SA keeps only one point, yet reached the target in 24 of 30 runs, close to GA (22) and PSO (28). A population helps reliability a little but is not essential here.
* **TPE.** The Bayesian baseline was the most reliable (30 / 30), although not the fastest. Learning a model of the score surface pays off on a smooth, low-dimensional problem like this one.
* **Multi-objective view.** If a small model matters, tune accuracy and support vectors together. NSGA-II found better trade-offs than the step 7 collection of single-objective runs, with about a third of the evaluations. Its most accurate point is a bigger model, though: the simplest top-accuracy model (98.02 % with 33 support vectors) came from step 7.
* **Settings.** The defaults are not optimal (for example c1 = c2 = 1.0 or 2.0 reached the target in 10 / 10 runs, w = 0.9 was much slower), but no setting changes the final accuracy.

---

## 11. Limitations and future work

* **One small dataset** (569 patients, nearly linearly separable) and one data split (seed 42). Conclusions about PSO vs GA may not carry over to harder problems (Sonar and Ionosphere are built in but were not used for the reported results).
* **Small test set:** 114 patients, so **one patient = 0.88 percentage points**. The final test accuracies of all methods differ by about one patient, which is too small to rank them.
* **Fixed CV split:** every setting is scored on the same 5 folds (seed 0), so the best settings partly fit those folds. Repeated or nested CV would be fairer (`N_REPEATS` exists but was not used for the reported results).
* **Metaheuristic settings were not tuned.** PSO and GA use fixed defaults (step 8 shows they are not optimal), SA settings were not studied, and the budget is only 100 evaluations.
* **Only 30 seeds**, so PSO vs GA (p = 0.394) might separate with more runs. The target level (0.9792) was defined from the best score found, which is convenient but arbitrary.
* The SMO solver is the *simplified* version (random second point); it is less exact than the QP solver and is only used for checks.
* Only the RBF kernel is implemented in our own SVM; other kernels use scikit-learn (step 9).
* Future work: nested CV, more datasets and benchmark functions, more seeds, successive halving, adaptive PSO/GA parameters, GP-based Bayesian optimisation.

---

## Objective functions and convergence (Step 11)

Tuning is a **bilevel** problem. `objectives.py` writes both levels as plain NumPy functions (each docstring states the formula and its symbols).

**Inner problem (train one SVM, $C$ and $\gamma$ fixed).** With $K_{ij}=\exp(-\gamma\lVert x_i-x_j\rVert^2)$ and $Q_{ij}=y_iy_jK_{ij}$:

* Dual, **maximised**: $D(\alpha)=\sum_i\alpha_i-\tfrac12\alpha^\top Q\alpha$, subject to $0\le\alpha_i\le C$ and $\sum_i\alpha_iy_i=0$.
* Primal, **minimised**: $P(\alpha,b)=\tfrac12\alpha^\top Q\alpha+C\sum_i\max\bigl(0,\,1-y_if(x_i)\bigr)$, with $f(x_i)=\sum_j\alpha_jy_jK_{ij}+b$. This is $\tfrac12\lVert w\rVert^2+C\sum\xi_i$ with $w=\sum_j\alpha_jy_j\phi(x_j)$ substituted and each $\xi_i$ set to its smallest feasible value.
* Duality gap: $P-D\ge 0$ (weak duality), $\to 0$ at the optimum (strong duality); relative gap $(P-D)/(|P|+10^{-12})$. In the plots $P$ uses the best $b$ for the current $\alpha$, so $P\ge D$ always.

**Outer problem (choose $\theta=(\log_{10}C,\log_{10}\gamma)\in[-2,3]\times[-4,1]$), maximised:**

$$F(\theta)=\frac1K\sum_{k=1}^{K}\frac1{|V_k|}\sum_{i\in V_k}\mathbf 1\bigl[\operatorname{sign} f^{(k)}_\theta(x_i)=y_i\bigr],\qquad \max_\theta F\bigl(\theta,\alpha^*(\theta)\bigr),\ \ \alpha^*(\theta)=\arg\max_\alpha D_\theta(\alpha).$$

$F$ is a step function of $\theta$ (it counts patients), so gradient-free tuners are used.

`smo_trace()` is Platt's full SMO (error cache, $|E_i-E_j|$ second choice). Every update moves two $\alpha$ along the equality constraint and clips to the box, so $\alpha$ is feasible after every update. It logs about 200 points per run and flags convergence (relative gap $<10^{-3}$); its final $D$ is compared with the cvxopt solution.

**Run:** `python step11_objective_convergence.py` (set `QUICK = True` at the top for a check in under a minute). It reruns each tuner once with seed 0 and writes `results/inner_convergence.csv`, `outer_trials.csv`, `cv_surface.csv`. The contour grid uses scikit-learn's SVC for speed (Step 3 showed it matches our SVM).

| Figure | Shows |
|---|---|
| `inner_primal_dual_grid.png` | $P$ falling and $D$ rising to the same value, one panel per $(C,\gamma)$ pair (the y-axis is zoomed; $P$ starts near $C\cdot n$, off the top) |
| `inner_duality_gap.png` | Relative gap vs SMO iteration (log-log), all pairs |
| `dual_slice_contour.png` | $D$ over two margin-SV multipliers (others fixed), the box $[0,C]^2$, the equality line, and the constrained optimum |
| `outer_convergence.png` | Left: best-so-far $F$ per method (y-axis starts at 0.93). Right: $F$ of every trial. Dotted line = 0.9792 |
| `outer_contour_all.png` | $F$ over $(\log C,\log\gamma)$ with each method's best point and the grid maximum (star) |
| `outer_contour_paths.png` | Each method's 100 trials in evaluation order, with start and best |

## 12. References

1. C. Cortes and V. Vapnik, "Support-vector networks", *Machine Learning* 20(3), 273-297, 1995.
2. J. Kennedy and R. Eberhart, "Particle swarm optimization", *Proc. IEEE International Conference on Neural Networks*, 1942-1948, 1995.
3. J. H. Holland, *Adaptation in Natural and Artificial Systems*, University of Michigan Press, 1975; D. E. Goldberg, *Genetic Algorithms in Search, Optimization and Machine Learning*, Addison-Wesley, 1989.
4. S. Kirkpatrick, C. D. Gelatt and M. P. Vecchi, "Optimization by simulated annealing", *Science* 220(4598), 671-680, 1983.
5. J. Bergstra and Y. Bengio, "Random search for hyper-parameter optimization", *Journal of Machine Learning Research* 13, 281-305, 2012.
6. D. H. Wolpert and W. G. Macready, "No free lunch theorems for optimization", *IEEE Transactions on Evolutionary Computation* 1(1), 67-82, 1997.
7. J. C. Platt, "Sequential Minimal Optimization: A fast algorithm for training support vector machines", Microsoft Research Technical Report MSR-TR-98-14, 1998.
8. J. Bergstra, R. Bardenet, Y. Bengio and B. Kegl, "Algorithms for hyper-parameter optimization", *Advances in Neural Information Processing Systems* 24, 2011 (TPE).
9. J. Snoek, H. Larochelle and R. P. Adams, "Practical Bayesian optimization of machine learning algorithms", *Advances in Neural Information Processing Systems* 25, 2012.
10. L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh and A. Talwalkar, "Hyperband: a novel bandit-based approach to hyperparameter optimization", *Journal of Machine Learning Research* 18(185), 1-52, 2018.
11. K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II", *IEEE Transactions on Evolutionary Computation* 6(2), 182-197, 2002.
12. H. B. Mann and D. R. Whitney, "On a test of whether one of two random variables is stochastically larger than the other", *Annals of Mathematical Statistics* 18(1), 50-60, 1947.
13. T. Akiba, S. Sano, T. Yanase, T. Ohta and M. Koyama, "Optuna: a next-generation hyperparameter optimization framework", *Proc. ACM SIGKDD*, 2019.
14. J. Blank and K. Deb, "pymoo: Multi-objective optimization in Python", *IEEE Access* 8, 89497-89509, 2020.
15. F. Pedregosa et al., "Scikit-learn: machine learning in Python", *Journal of Machine Learning Research* 12, 2825-2830, 2011.
16. D. Dua and C. Graff, *UCI Machine Learning Repository*, University of California, Irvine, https://archive.ics.uci.edu (Breast Cancer Wisconsin (Diagnostic), Sonar, Ionosphere).
17. Reference repository: https://github.com/abhipatel35/svm-hyperparameter-optimization-for-breast-cancer. It uses scikit-learn's `SVC` with `GridSearchCV` (cv=2) and the kernel fixed to `'linear'`. Its `param_grid` lists `'gamma'` twice, so the second entry (`['scale','auto']`) overwrites the first, and gamma has no effect with a linear kernel anyway, so effectively only C in {0.1, 1, 10, 100} is tuned. Its `train_test_split` has no `random_state` or `stratify`, and it does not scale the features. **Our implementation is different**: the SVM, the optimisers and the comparison framework are written from scratch, with an RBF kernel and an equal-budget comparison of several tuners.
18. K. Jamieson and A. Talwalkar, "Non-stochastic best arm identification and hyperparameter optimization", *Proc. AISTATS*, 2016 (successive halving).
