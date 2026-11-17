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
| PSO | | | | | | |
| GA | | | | | | |
| Grid search | *(fill)* | *(fill)* | *(fill)* | *(fill)* | *(fill)* | *(fill)* |
| Random search | | | | | | |
| SA | | | | | | |

**Many seeds (`step6_experiments.py`, N seeds = *(fill)*)**

| Method | mean CV score | mean test acc | malignant recall | mean trials to target | runs reaching target |
|---|---|---|---|---|---|
| PSO | | | | | / N |
| GA | | | | | / N |
| Grid search | *(fill)* | *(fill)* | *(fill)* | *(fill)* | *(fill)* / 1 |
| Random search | | | | | / N |
| Bayesian (TPE) | | | | | / N |
| SA | | | | | / N |

**Significance (Mann-Whitney U on trials-to-target, `results/significance.csv`)**

| Comparison | p-value |
|---|---|
| PSO vs Random | *(fill)* |
| GA vs Random | *(fill)* |
| PSO vs GA | *(fill)* |
| SA vs Random | *(fill)* |

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

### 4.4 This project's numbers (fill in from `results/summary.csv`)

| Method | trial at which the best accuracy was found (seed 0) | seeds that reached the target |
|---|---|---|
| PSO | *(fill)* | *(fill)* / N |
| GA | *(fill)* | *(fill)* / N |
| Grid | *(fill)* | *(fill)* / 1 |
| Random | *(fill)* | *(fill)* / N |
| SA | *(fill)* | *(fill)* / N |

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

## 9. Limitations and future work

* One data split (seed 42) and one set of CV folds. The test set has only 114 patients, so **one patient = 0.88 percentage points**; test accuracies of different methods are often identical.
* Several methods reach the same best CV accuracy: this landscape is easy (a wide plateau of good settings), so differences are about *speed to get there*, not final accuracy.
* The budget is only 100 evaluations; results could change with other budgets.
* PSO and SA settings were not tuned per method (step 8 studies PSO and GA only).
* The SMO solver is the *simplified* version (random second point); it is less exact than the QP solver and is only used for checks.
* Only the RBF kernel is implemented in our own SVM; other kernels use scikit-learn (step 9).
* Future work: nested CV, more datasets and benchmark functions, more seeds, successive halving, adaptive PSO/GA parameters, GP-based Bayesian optimisation.

---

## 10. References

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
