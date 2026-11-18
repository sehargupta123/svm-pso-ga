# The Mathematics Behind the Project: SVM, GA and PSO

This document explains, step by step and with every symbol defined, the three pieces of mathematics the project relies on:

0. **The search ranges**: why we search $\log_{10} C$ and $\log_{10}\gamma$, and where their limits come from (Part 0.5).
1. **The support vector machine (SVM)**: the primal problem, the Lagrangian, the KKT conditions, the dual problem, and how `my_svm.py` solves and checks it.
2. **The genetic algorithm (GA)**: how `ga.py` turns evolution into an optimiser.
3. **Particle swarm optimisation (PSO)**: how `pso.py` moves a swarm, and why its settings are stable.

The equations use standard LaTeX math, which renders on GitHub and in VS Code's Markdown preview.

---

## Part 0. The setting

We have $n$ training patients. Patient $i$ is described by a feature vector and a label:

| Symbol | Meaning | In this project |
|---|---|---|
| $n$ | number of training points | 455 (or 364 inside one CV fold) |
| $d$ | number of features | 30 |
| $\mathbf{x}_i \in \mathbb{R}^d$ | feature vector of patient $i$ (after scaling) | 30 measurements of cell nuclei |
| $y_i \in \{-1, +1\}$ | label of patient $i$ | $-1$ = malignant, $+1$ = benign |

The labels are $\pm 1$ rather than $0/1$ because it lets a single expression, $y_i \cdot (\text{score})$, mean "correct and confident" whenever it is large and positive, whichever class the point belongs to.

The two hyperparameters we tune are $C$ and $\gamma$. Every optimiser in the project searches over

$$
\boldsymbol{\theta} = (\log_{10} C,\ \log_{10} \gamma) \in [-2, 3] \times [-4, 1],
$$

and scores each $\boldsymbol{\theta}$ with the **fitness function**

$$
F(\boldsymbol{\theta}) = \frac{1}{5} \sum_{k=1}^{5} \text{Accuracy}_k(\boldsymbol{\theta}),
$$

the mean accuracy over 5 cross-validation folds, where in fold $k$ the SVM is trained on 4/5 of the training set and tested on the remaining 1/5 (with feature scaling fitted on the 4/5 only). When the tie-break option is on, the value returned is $F(\boldsymbol{\theta}) - 10^{-6} \cdot \overline{n}_{SV}$, where $\overline{n}_{SV}$ is the average number of support vectors; the penalty is at most $3.6 \times 10^{-4}$, smaller than one patient's worth of accuracy ($1/455 \approx 0.0022$), so it only decides between exact ties.

---

## Part 0.5. Where the search ranges for log C and log γ come from

Every optimiser searches the box $\log_{10} C \in [-2, 3]$ and $\log_{10}\gamma \in [-4, 1]$, that is $C$ from $0.01$ to $1000$ and $\gamma$ from $0.0001$ to $10$. This part explains (1) why we search the **logarithms** instead of $C$ and $\gamma$ directly, and (2) where the **numbers** $-2, 3, -4, 1$ come from. All numbers below were computed on the project's own scaled training data.

### 0.5.1 Why search on a log scale

$C$ and $\gamma$ matter by their **order of magnitude**, not by their absolute size. Changing $C$ from 1 to 10 changes the model about as much as changing it from 100 to 1000; changing it from 100 to 101 changes almost nothing. So the natural unit of change is "times 10", and taking $\log_{10}$ turns "times 10" into "plus 1":

$$
\theta_1 = \log_{10} C \quad\Longleftrightarrow\quad C = 10^{\theta_1}, \qquad \theta_2 = \log_{10}\gamma \quad\Longleftrightarrow\quad \gamma = 10^{\theta_2}.
$$

This has three practical benefits:

1. **Equal steps mean equal effects.** A PSO or GA step of $+0.5$ always multiplies $C$ by $10^{0.5} \approx 3.2$, wherever the particle is. Searching $C$ directly on $[0.01, 1000]$, a step of $+0.5$ would be enormous at $C = 0.01$ and meaningless at $C = 900$.
2. **Uniform sampling is fair.** Random search draws $\theta$ uniformly, so each decade ($0.01$ to $0.1$, $0.1$ to $1$, and so on) gets the same share of trials. Drawing $C$ uniformly on $[0.01, 1000]$ would put 90% of the trials above $C = 100$.
3. **Positivity is automatic.** $C$ and $\gamma$ must be positive. $10^{\theta}$ is positive for every real $\theta$, so the optimisers never produce an invalid value.

This is the same advice given in the standard practical guide to SVMs (Hsu et al., 2003), which recommends exponentially spaced values such as $C = 2^{-5}, 2^{-3}, \dots, 2^{15}$ and $\gamma = 2^{-15}, 2^{-13}, \dots, 2^{3}$. We use powers of 10 instead of powers of 2, which is the same idea in a different base.

### 0.5.2 Where the γ range comes from: what γ does to the kernel

$\gamma$ only ever appears in the product $\gamma \lVert \mathbf{x} - \mathbf{z} \rVert^2$ inside $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \lVert \mathbf{x} - \mathbf{z} \rVert^2)$. So a sensible range for $\gamma$ depends on how far apart patients typically are.

After standard scaling, every one of the $d = 30$ features has mean 0 and variance 1. For two independent patients, each feature contributes on average $\mathbb{E}[(x_k - z_k)^2] = 2$ to the squared distance, so we expect

$$
\mathbb{E}\lVert \mathbf{x} - \mathbf{z} \rVert^2 \approx 2d = 60.
$$

On the real training data, the measured values are:

| Distance between two patients | Squared distance $\lVert \mathbf{x} - \mathbf{z} \rVert^2$ |
|---|---|
| Average over all pairs | 60.1 (matches $2d = 60$) |
| Median pair | 40.6 |
| Median distance to a patient's nearest neighbour | 4.45 |

Now look at what the kernel value becomes at the two ends of our range:

| $\gamma$ | $K$ for a typical pair (median distance) | $K$ for nearest neighbours | What the SVM "sees" |
|---|---|---|---|
| $10^{-4}$ (lower bound) | 0.996 | 1.000 | every patient looks almost identical to every other: a very smooth, nearly linear boundary |
| $10^{-3}$ | 0.96 | 0.996 | smooth boundary |
| $10^{-2}$ | 0.67 | 0.96 | moderate curvature |
| $10^{-1}$ | 0.017 | 0.64 | only fairly close patients count |
| $1$ | $\approx 10^{-18}$ | 0.012 | almost only exact neighbours count |
| $10$ (upper bound) | $\approx 10^{-177}$ | $\approx 10^{-20}$ | every patient is similar only to itself: the model memorises the training data |

So the range **brackets everything from "all patients look the same" to "every patient is unique"**:

- **Below $10^{-4}$**, $K$ is already about 1 for all pairs, so making $\gamma$ even smaller cannot change the model in any meaningful way.
- **Above $10$**, $K$ is already about 0 for all pairs, so the model has fully memorised the training data; making $\gamma$ larger changes nothing. In the project's accuracy map, this end gives 62.6% accuracy, which is exactly the share of benign patients (357 of 569): the model gives up and predicts "benign" for everyone.

A useful way to state the same range: $\gamma \cdot 60$ (gamma times the average squared distance) runs from $0.006$ to $600$.

As a sanity check, scikit-learn's own default, `gamma="scale"`, is $1/(d \cdot \operatorname{Var}(\mathbf{x})) = 1/30 \approx 0.033$, so $\log_{10}\gamma \approx -1.5$, comfortably inside our range.

### 0.5.3 Where the C range comes from: what C does to the solution

$C$ is the price of a margin violation, and in the dual it is the upper limit of every multiplier: $0 \le \alpha_i \le C$.

- **At $C = 0.01$ (lower bound)** violations are almost free and every $\alpha_i$ is squeezed into $[0, 0.01]$. The SVM barely tries to fit the data. Combined with a small $\gamma$, it underfits completely: the accuracy map gives 62.6%, the same "predict benign for everyone" result.
- **At $C = 1000$ (upper bound)** violations are so expensive that the SVM behaves almost like a hard-margin SVM. Because the scaled data are close to separable, few violations are left to remove, so raising $C$ further changes very little.
- scikit-learn's default, $C = 1$ ($\log_{10} C = 0$), sits inside the range.

Cross-validated accuracy along one row of the map ($\gamma = 10^{-2.5}$) shows this saturation clearly:

| $\log_{10} C$ | $-2$ | $-1.2$ | $-0.3$ | $0.5$ | $1.4$ | $2.2$ |
|---|---|---|---|---|---|---|
| CV accuracy | 0.626 | 0.884 | 0.952 | 0.974 | 0.976 | 0.976 |

Accuracy rises steeply through the first decades of $C$ and then goes flat, which is exactly what the reasoning above predicts.

### 0.5.4 Checking that the box was a good choice

A search range is right when (a) its edges are clearly poor or clearly flat, so nothing better lies outside, and (b) the best values found sit inside it. The project's accuracy map (60 by 60 grid, scikit-learn's SVC) shows:

| Corner of the box | CV accuracy |
|---|---|
| low $C$, low $\gamma$ $(0.01, 10^{-4})$ | 0.626 |
| low $C$, high $\gamma$ $(0.01, 10)$ | 0.626 |
| high $C$, high $\gamma$ $(1000, 10)$ | 0.626 |
| high $C$, low $\gamma$ $(1000, 10^{-4})$ | 0.971 |

Down the $\gamma$ axis at $C = 10$, accuracy goes 0.947, 0.976, 0.974, 0.971, 0.923, 0.626 at $\log_{10}\gamma = -4, -3.2, -2.3, -1.5, -0.6, 0.2$, so accuracy has already collapsed well before the upper edge.

- Three corners are at the "predict benign for everyone" floor, so the box is wide enough to include the clearly bad regions on those sides.
- The good region does reach the high-$C$, low-$\gamma$ corner, but accuracy there is already flat (97.1% at the corner, 97.6% to 98.0% just inside), so extending the box in that direction would not find anything measurably better.
- The best settings the tuners actually found (for example $C \approx 20$ to $90$, $\gamma \approx 0.0005$ to $0.004$, that is $\log_{10} C \approx 1.3$ to $1.95$ and $\log_{10}\gamma \approx -3.3$ to $-2.4$) lie well inside the box.
- Only about 7% of the box reaches 97.5% CV accuracy or more, so the box is wide enough that **where** you search still matters, which is what makes it a fair test of the optimisers.

### 0.5.5 How to choose the ranges for a different dataset

1. **Scale the features** first; otherwise distances, and therefore $\gamma$, are dominated by the largest-valued features.
2. **Compute the typical squared distance** $\bar{D}$ between scaled points (about $2d$ for $d$ standardised features).
3. **Pick the $\gamma$ range** so that $\gamma \bar{D}$ runs from about $0.01$ (everything similar) to about $1000$ (everything unique).
4. **Pick the $C$ range** to span several decades around 1, for example $10^{-2}$ to $10^{3}$, wider if the data are noisy.
5. **Run a coarse search and check the edges.** If the best point found sits on a boundary where accuracy is still rising, widen the box in that direction and search again.

---

## Part 1. The support vector machine

### 1.1 The geometry: what $\mathbf{w}$ and $b$ are

A linear classifier draws a **hyperplane** (a line in 2-D, a flat sheet in 3-D, its analogue in 30-D):

$$
\mathbf{w}^\top \mathbf{x} + b = 0.
$$

| Symbol | Meaning |
|---|---|
| $\mathbf{w} \in \mathbb{R}^d$ | the **weight vector**, perpendicular (normal) to the hyperplane. Its direction sets the orientation of the boundary; each component says how much one feature pushes the score up or down. |
| $b \in \mathbb{R}$ | the **bias** (offset). It slides the hyperplane parallel to itself without rotating it. |
| $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$ | the **decision score**. Its sign is the predicted class: $\hat{y} = \operatorname{sign}(f(\mathbf{x}))$. |

The perpendicular distance from a point $\mathbf{x}_i$ to the hyperplane is

$$
\text{dist}(\mathbf{x}_i) = \frac{|\mathbf{w}^\top \mathbf{x}_i + b|}{\lVert \mathbf{w} \rVert}.
$$

The pair $(\mathbf{w}, b)$ has a scale freedom: multiplying both by 5 gives the same hyperplane. We remove it by a convention called the **canonical form**: scale $(\mathbf{w}, b)$ so that the training points closest to the boundary satisfy $|\mathbf{w}^\top \mathbf{x}_i + b| = 1$. Those closest points then lie on two parallel planes, $\mathbf{w}^\top \mathbf{x} + b = +1$ and $\mathbf{w}^\top \mathbf{x} + b = -1$, and the gap between them, the **margin**, has width

$$
\text{margin} = \frac{2}{\lVert \mathbf{w} \rVert}.
$$

So **making the margin wide is the same as making $\lVert \mathbf{w} \rVert$ small.** That single fact turns "find the widest gap" into an optimisation problem.

### 1.2 The hard-margin primal problem

If the two classes can be separated perfectly, the widest-margin hyperplane solves

$$
\min_{\mathbf{w},\, b} \ \ \frac{1}{2} \lVert \mathbf{w} \rVert^2
\qquad \text{subject to} \qquad
y_i \left( \mathbf{w}^\top \mathbf{x}_i + b \right) \ge 1, \quad i = 1, \dots, n.
$$

- **Objective.** Minimising $\tfrac12 \lVert \mathbf{w} \rVert^2$ maximises the margin $2/\lVert \mathbf{w} \rVert$. The square and the $\tfrac12$ change nothing about where the minimum is; they make the objective a smooth quadratic whose gradient is simply $\mathbf{w}$.
- **Constraints.** $y_i f(\mathbf{x}_i) \ge 1$ says each point is on its correct side **and** at least on the margin plane (never inside the gap).

### 1.3 The soft-margin primal problem (what the project uses)

Real data overlap, so the hard constraints may be impossible to satisfy. We allow each point a **slack variable** $\xi_i \ge 0$ and charge for it:

$$
\boxed{
\min_{\mathbf{w},\, b,\, \boldsymbol{\xi}} \ \ \frac{1}{2} \lVert \mathbf{w} \rVert^2 + C \sum_{i=1}^{n} \xi_i
\quad \text{s.t.} \quad
y_i \left( \mathbf{w}^\top \phi(\mathbf{x}_i) + b \right) \ge 1 - \xi_i, \quad \xi_i \ge 0, \quad i = 1, \dots, n.
}
$$

| Symbol | Meaning |
|---|---|
| $\xi_i$ | **slack**: how far point $i$ violates the margin. $\xi_i = 0$: on or outside the margin, correct side. $0 < \xi_i < 1$: inside the gap but still correctly classified. $\xi_i = 1$: exactly on the boundary. $\xi_i > 1$: misclassified. |
| $\sum_i \xi_i$ | total violation; an upper bound on the number of training mistakes. |
| $C > 0$ | the **penalty weight**, the price per unit of violation. Large $C$: violations are expensive, so the model shrinks the margin to fit the training data (risk of overfitting). Small $C$: violations are cheap, so the model prefers a wide, simple margin (risk of underfitting). |
| $\phi(\mathbf{x})$ | a (possibly very high-dimensional) **feature map**. With $\phi(\mathbf{x}) = \mathbf{x}$ this is the linear SVM; Section 1.7 explains why we never compute $\phi$ explicitly. |

This is the "penalty method" idea from the syllabus: instead of forbidding constraint violations, the objective pays $C$ for each unit of violation.

**Why the problem is well-behaved.** The objective is convex (a convex quadratic plus a linear term), every constraint is linear, and a feasible point always exists (take $\mathbf{w} = 0$, $b = 0$, $\xi_i = 1$). So the problem is a **convex quadratic programme** with a unique optimal $\mathbf{w}$, and the strong duality and KKT results below apply.

### 1.4 The Lagrangian

To handle the constraints, we attach one **Lagrange multiplier** to each of them. Rewrite the constraints in the "$\ge 0$" form:

$$
g_i = y_i \left( \mathbf{w}^\top \phi(\mathbf{x}_i) + b \right) - 1 + \xi_i \ \ge 0, \qquad h_i = \xi_i \ \ge 0.
$$

| Multiplier | Attached to | Meaning |
|---|---|---|
| $\alpha_i \ge 0$ | the margin constraint $g_i \ge 0$ | how hard point $i$ "pushes" on the boundary. Most points end with $\alpha_i = 0$. |
| $\mu_i \ge 0$ | the slack constraint $\xi_i \ge 0$ | enforces non-negative slack. |

The **Lagrangian** subtracts each multiplier times its constraint from the objective:

$$
L(\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\alpha}, \boldsymbol{\mu})
= \frac{1}{2} \lVert \mathbf{w} \rVert^2 + C \sum_{i=1}^{n} \xi_i
- \sum_{i=1}^{n} \alpha_i \left[ y_i \left( \mathbf{w}^\top \phi(\mathbf{x}_i) + b \right) - 1 + \xi_i \right]
- \sum_{i=1}^{n} \mu_i \xi_i .
$$

Intuition: if any constraint were violated ($g_i < 0$), the adversary choosing $\alpha_i$ could make $L$ arbitrarily large, so minimising over $(\mathbf{w}, b, \boldsymbol{\xi})$ and maximising over the multipliers recovers exactly the constrained problem:

$$
\text{primal} = \min_{\mathbf{w}, b, \boldsymbol{\xi}} \ \max_{\boldsymbol{\alpha} \ge 0,\, \boldsymbol{\mu} \ge 0} L,
\qquad
\text{dual} = \max_{\boldsymbol{\alpha} \ge 0,\, \boldsymbol{\mu} \ge 0} \ \min_{\mathbf{w}, b, \boldsymbol{\xi}} L .
$$

For a convex problem with linear constraints and a feasible point (Slater's condition), these two values are equal: **strong duality holds**, so we may solve whichever is easier.

### 1.5 The KKT conditions

At the optimum, the Karush-Kuhn-Tucker conditions must all hold. For a convex problem like this they are both **necessary and sufficient**: any point that satisfies them is optimal.

**(1) Stationarity**: the gradient of $L$ with respect to each primal variable is zero.

$$
\frac{\partial L}{\partial \mathbf{w}} = \mathbf{w} - \sum_{i=1}^{n} \alpha_i y_i \phi(\mathbf{x}_i) = 0
\quad\Longrightarrow\quad
\boxed{\mathbf{w} = \sum_{i=1}^{n} \alpha_i y_i \phi(\mathbf{x}_i)}
$$

$$
\frac{\partial L}{\partial b} = -\sum_{i=1}^{n} \alpha_i y_i = 0
\quad\Longrightarrow\quad
\boxed{\sum_{i=1}^{n} \alpha_i y_i = 0}
$$

$$
\frac{\partial L}{\partial \xi_i} = C - \alpha_i - \mu_i = 0
\quad\Longrightarrow\quad
\boxed{\mu_i = C - \alpha_i}
$$

The first line is the key insight: **the optimal $\mathbf{w}$ is a weighted sum of training points**, and only points with $\alpha_i > 0$ contribute. Those are the **support vectors**.

**(2) Primal feasibility**: the original constraints hold.

$$
y_i \left( \mathbf{w}^\top \phi(\mathbf{x}_i) + b \right) \ge 1 - \xi_i, \qquad \xi_i \ge 0.
$$

**(3) Dual feasibility**: multipliers are non-negative.

$$
\alpha_i \ge 0, \qquad \mu_i \ge 0 .
$$

Combined with $\mu_i = C - \alpha_i$, the condition $\mu_i \ge 0$ becomes $\alpha_i \le C$. So $\boxed{0 \le \alpha_i \le C}$, the **box constraint**.

**(4) Complementary slackness**: for each constraint, either the constraint is tight or its multiplier is zero.

$$
\alpha_i \left[ y_i f(\mathbf{x}_i) - 1 + \xi_i \right] = 0, \qquad \mu_i \xi_i = (C - \alpha_i)\, \xi_i = 0 .
$$

### 1.6 The three cases of $\alpha_i$ (what the KKT conditions say about each point)

Writing $f(\mathbf{x}_i)$ for the decision score of training point $i$, complementary slackness splits every training point into exactly one of three cases:

| Case | What the conditions force | Where the point sits | Role |
|---|---|---|---|
| $\alpha_i = 0$ | $\mu_i = C > 0 \Rightarrow \xi_i = 0$, so $y_i f(\mathbf{x}_i) \ge 1$ | on the correct side, on or outside the margin | **Not** a support vector. Removing it would not change the boundary. |
| $0 < \alpha_i < C$ | $\mu_i > 0 \Rightarrow \xi_i = 0$; and $\alpha_i > 0 \Rightarrow$ the margin constraint is tight | **exactly on the margin**: $y_i f(\mathbf{x}_i) = 1$ | **Margin support vector.** Pins the boundary in place; used to compute $b$. |
| $\alpha_i = C$ | $\mu_i = 0$, so $\xi_i \ge 0$ is allowed; margin constraint tight: $y_i f(\mathbf{x}_i) = 1 - \xi_i$ | inside the margin or misclassified: $y_i f(\mathbf{x}_i) \le 1$ | **Bound support vector.** Pays the full penalty $C$. |

### 1.7 The dual problem and the kernel trick

Substitute the stationarity results back into $L$. Using $\mathbf{w} = \sum_i \alpha_i y_i \phi(\mathbf{x}_i)$:

$$
\frac{1}{2} \lVert \mathbf{w} \rVert^2 = \frac{1}{2} \sum_{i}\sum_{j} \alpha_i \alpha_j y_i y_j\, \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j),
\qquad
\sum_i \alpha_i y_i \mathbf{w}^\top \phi(\mathbf{x}_i) = \sum_{i}\sum_{j} \alpha_i \alpha_j y_i y_j\, \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j).
$$

The $b$ term vanishes because $\sum_i \alpha_i y_i = 0$, and the $\xi$ terms cancel because $C - \alpha_i - \mu_i = 0$. What is left depends on $\boldsymbol{\alpha}$ alone:

$$
\boxed{
\max_{\boldsymbol{\alpha}} \ \ \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)
\quad \text{s.t.} \quad
0 \le \alpha_i \le C, \quad \sum_{i=1}^{n} \alpha_i y_i = 0,
}
$$

where $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$ is the **kernel**.

**Why the dual is what we solve:**

1. **It only needs dot products.** $\phi$ never appears on its own, only inside $\phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$. So we can pick a function $K$ that equals a dot product in some huge (even infinite-dimensional) feature space without ever computing $\phi$. This is the **kernel trick**.
2. **Its constraints are simple**: a box $[0, C]$ for each $\alpha_i$ plus one linear equality. Standard QP solvers handle this directly.
3. **It has $n$ variables** (one per training point), regardless of how many features there are.

**The RBF (Gaussian) kernel** used in the project is

$$
K(\mathbf{x}, \mathbf{z}) = \exp\!\left( -\gamma \lVert \mathbf{x} - \mathbf{z} \rVert^2 \right), \qquad \gamma = \frac{1}{2\sigma^2}.
$$

$K = 1$ when two points coincide and falls smoothly toward 0 as they move apart. $\gamma$ controls how fast: large $\gamma$ (narrow bell) lets each point influence only its close neighbours, giving a wiggly boundary; small $\gamma$ (wide bell) gives a smooth one. The RBF kernel corresponds to an infinite-dimensional $\phi$, which is why it can fit almost any boundary shape.

The matrix with entries $Q_{ij} = y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$ is positive semi-definite for any valid kernel, so the dual objective is concave and the dual is also a convex QP.

### 1.8 Recovering the classifier: prediction and the bias $b$

After solving for $\boldsymbol{\alpha}$, we never form $\mathbf{w}$ explicitly. Substituting $\mathbf{w} = \sum_i \alpha_i y_i \phi(\mathbf{x}_i)$ into $f(\mathbf{x}) = \mathbf{w}^\top \phi(\mathbf{x}) + b$ gives

$$
\boxed{ f(\mathbf{x}) = \sum_{i \in SV} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b, \qquad \hat{y} = \operatorname{sign} f(\mathbf{x}) }
$$

where $SV = \{ i : \alpha_i > 0 \}$. With the RBF kernel, the decision score is literally a weighted sum of Gaussian bumps centred on the support vectors.

The bias comes from the margin support vectors. For any $i$ with $0 < \alpha_i < C$, Section 1.6 says $y_i f(\mathbf{x}_i) = 1$. Multiplying both sides by $y_i$ (and using $y_i^2 = 1$):

$$
b = y_i - \sum_{j \in SV} \alpha_j y_j K(\mathbf{x}_j, \mathbf{x}_i).
$$

Each margin support vector gives an estimate of $b$; `my_svm.py` **averages over all of them** for numerical stability. If no point falls strictly between $0$ and $C$, it falls back to averaging over all support vectors.

### 1.9 How the code solves the dual (`my_svm.py`, QP solver)

The solver `cvxopt.solvers.qp` minimises

$$
\min_{\mathbf{z}} \ \frac{1}{2} \mathbf{z}^\top P \mathbf{z} + \mathbf{q}^\top \mathbf{z}
\quad \text{s.t.} \quad G\mathbf{z} \le \mathbf{h}, \quad A\mathbf{z} = \mathbf{b}_{\text{eq}}.
$$

Maximising the dual is the same as minimising its negative, so with $\mathbf{z} = \boldsymbol{\alpha}$:

| cvxopt term | Value in the code | Corresponds to |
|---|---|---|
| $P$ ($n \times n$) | `outer(y, y) * K + 1e-8 * I` | $Q_{ij} = y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$, the quadratic term. The tiny $10^{-8}$ on the diagonal keeps the matrix numerically positive definite. |
| $\mathbf{q}$ | vector of $-1$ | the $-\sum_i \alpha_i$ term (sign flipped because we minimise) |
| $G$ ($2n \times n$) | $\begin{bmatrix} -I \\ I \end{bmatrix}$ | the two halves of the box: $-\alpha_i \le 0$ and $\alpha_i \le C$ |
| $\mathbf{h}$ | $\begin{bmatrix} \mathbf{0} \\ C\,\mathbf{1} \end{bmatrix}$ | right-hand sides of the box |
| $A$ ($1 \times n$) | $\mathbf{y}^\top$ | $\sum_i \alpha_i y_i$ |
| $\mathbf{b}_{\text{eq}}$ | $0$ | $\sum_i \alpha_i y_i = 0$ |

cvxopt uses an interior-point method, so it returns values that are only accurate to a few digits **relative to $C$**. That is why the code uses tolerances that scale with $C$:

- `_a_tol` $= 10^{-5} \cdot \max(C, 1)$: any $\alpha_i$ below this counts as zero (not a support vector).
- `_c_tol` $= 10^{-3} \cdot C$: any $\alpha_i$ above $C - $ `_c_tol` counts as "at the bound $C$".

**The SMO alternative.** `MySVM(solver="smo")` uses Platt's (1998) simplified sequential minimal optimisation. Because of the equality constraint $\sum_i \alpha_i y_i = 0$, no single $\alpha_i$ can change alone, so SMO picks **two** multipliers $(\alpha_i, \alpha_j)$ at a time, solves that two-variable QP analytically along the line that keeps the equality satisfied, clips the result to the box $[L, H]$, updates $b$, and repeats until no point violates the KKT conditions by more than a tolerance.

### 1.10 The KKT check (`check_kkt`)

Since the KKT conditions are necessary and sufficient here, checking them on the returned $\boldsymbol{\alpha}$ is a direct test that the solver reached the true optimum. For every training point, the code computes $m_i = y_i f(\mathbf{x}_i)$ and tests the condition that matches its $\alpha_i$ (with tolerance $\varepsilon = 10^{-3}$):

| Classified as | Condition tested |
|---|---|
| $\alpha_i \le$ `_a_tol` (zero) | $m_i \ge 1 - \varepsilon$ |
| $\alpha_i \ge C -$ `_c_tol` (at bound) | $m_i \le 1 + \varepsilon$ |
| otherwise (strictly between) | $\lvert m_i - 1 \rvert \le \varepsilon$ |

It reports the **percentage of points that pass**. In Step 3, 98.2% to 100% of points passed, which is the evidence that the hand-written SVM is correct.

### 1.11 Summary: how primal and dual fit together

| | Primal | Dual |
|---|---|---|
| Variables | $\mathbf{w} \in \mathbb{R}^{\dim \phi}$, $b$, $\boldsymbol{\xi} \in \mathbb{R}^n$ | $\boldsymbol{\alpha} \in \mathbb{R}^n$ |
| Objective | minimise $\tfrac12 \lVert \mathbf{w} \rVert^2 + C \sum \xi_i$ | maximise $\sum \alpha_i - \tfrac12 \boldsymbol{\alpha}^\top Q \boldsymbol{\alpha}$ |
| Constraints | $y_i f(\mathbf{x}_i) \ge 1 - \xi_i$, $\xi_i \ge 0$ | $0 \le \alpha_i \le C$, $\sum \alpha_i y_i = 0$ |
| Needs $\phi$ explicitly? | yes | no, only $K$ |
| Link between them | $\mathbf{w} = \sum_i \alpha_i y_i \phi(\mathbf{x}_i)$; optimal values equal (strong duality); KKT conditions hold at the shared optimum | |

---

## Part 2. The genetic algorithm (`ga.py`)

A GA treats each candidate solution as an individual in a population, and improves the population by selection, recombination and mutation, as in natural evolution (Holland, 1975; Goldberg, 1989).

### 2.1 Representation

| Concept | In this project |
|---|---|
| **Chromosome** (individual) | a real vector $\boldsymbol{\theta} = (\theta_1, \theta_2) = (\log_{10} C, \log_{10}\gamma)$ |
| **Gene** | one coordinate $\theta_k$ |
| **Search box** | $\boldsymbol{\ell} = (-2, -4)$ (lower bounds), $\mathbf{u} = (3, 1)$ (upper bounds) |
| **Fitness** | $F(\boldsymbol{\theta})$, the 5-fold CV accuracy (higher is better) |
| **Population** | $N = 10$ individuals $\{\boldsymbol{\theta}^{(1)}, \dots, \boldsymbol{\theta}^{(N)}\}$ |

This is a **real-coded** GA: genes are continuous numbers, not bit strings, so crossover and mutation work directly on real values.

### 2.2 The operators

**Initialisation.** Each gene is drawn uniformly in its range: $\theta_k \sim \mathcal{U}(\ell_k, u_k)$.

**Tournament selection (size $s = 3$).** To choose one parent, pick 3 distinct individuals at random and keep the fittest:

$$
\text{parent} = \arg\max_{m \in T} F(\boldsymbol{\theta}^{(m)}), \qquad T = \text{3 random distinct indices}.
$$

Fitter individuals win more tournaments, but weaker ones still win sometimes, which keeps diversity. Only the two weakest individuals can never win, because any tournament containing one of them also contains at least one fitter member. The best individual wins every tournament it enters, and it enters each one with probability $s/N = 0.3$.

**Blend crossover** (applied with probability $p_c = 0.8$). Given parents $\mathbf{a}$ and $\mathbf{b}$, draw an independent random weight per gene and mix:

$$
\boldsymbol{\theta}^{\text{child}} = \mathbf{a} + \mathbf{r} \odot (\mathbf{b} - \mathbf{a}), \qquad r_k \sim \mathcal{U}(0, 1) \ \text{independently},
$$

where $\odot$ is element-wise multiplication. Because each gene gets its own $r_k$, the child lands **anywhere inside the rectangle spanned by the two parents**, not just on the line between them. With probability $1 - p_c = 0.2$ the child is a copy of parent $\mathbf{a}$.

**Gaussian mutation** (applied with probability $p_m = 0.15$). Add normal noise to every gene:

$$
\boldsymbol{\theta}^{\text{child}} \leftarrow \boldsymbol{\theta}^{\text{child}} + \boldsymbol{\epsilon}, \qquad \epsilon_k \sim \mathcal{N}(0, \sigma_k^2), \quad \sigma_k = 0.1\,(u_k - \ell_k) = 0.5.
$$

So a typical mutation moves a gene by about half an order of magnitude in $C$ or $\gamma$.

**Repair (clipping).** After crossover and mutation, each gene is clipped to its range: $\theta_k \leftarrow \min(\max(\theta_k, \ell_k), u_k)$.

**Elitism ($e = 2$).** The 2 fittest individuals are copied unchanged into the next generation, keeping their known fitness. This guarantees the best fitness found so far **never decreases** from one generation to the next.

**No repeated evaluations (`make_unique`).** Every scored point is remembered (rounded to 4 decimals). If a child repeats a known point, it is nudged by $\mathcal{N}(0, \sigma_k^2)$ noise up to 20 times until it is new; in the very rare case that all 20 nudges fail, it is replaced by a fresh uniform random point. This fixed a real problem found in the project: before the fix, the GA wasted about 49 of its 100 evaluations re-scoring points it had already tried.

### 2.3 One generation, step by step

1. Sort the population by fitness and copy the top $e = 2$ into the new population.
2. Until the new population has $N = 10$ members (or the budget runs out): pick two parents by tournament, apply crossover with probability 0.8, apply mutation with probability 0.15, clip, make unique, score it (one evaluation), add it.
3. If the budget ran out mid-generation, fill the remaining slots with the best of the previous generation (unscored, so no budget is used).
4. Replace the old population with the new one.

### 2.4 The budget

The initial population costs 10 evaluations; each later generation costs $N - e = 8$ (elites are not re-scored). The GA stops at exactly **100 unique evaluations**: $10 + 11 \times 8 = 98$, plus 2 more in a final partial generation.

### 2.5 What each parameter controls

| Parameter | Value | Effect of increasing it |
|---|---|---|
| $N$ (population) | 10 | more exploration per generation, fewer generations within the budget |
| $s$ (tournament size) | 3 | stronger selection pressure: faster convergence, less diversity |
| $p_c$ (crossover rate) | 0.8 | more mixing of good solutions |
| $p_m$ (mutation rate) | 0.15 | more exploration of new areas. Step 8 found 0.05 too low (only 6 of 10 runs reached the target). |
| $\sigma$ (mutation size) | 10% of range | bigger jumps when mutating |
| $e$ (elites) | 2 | safer, but more exploitation of the current best |

---

## Part 3. Particle swarm optimisation (`pso.py`)

PSO moves a swarm of candidate solutions ("particles") through the search space. Each particle remembers its own best position and is told the swarm's best position, and steers between them (Kennedy & Eberhart, 1995).

### 3.1 Variables

For particle $j = 1, \dots, M$ (here $M = 10$) at iteration $t = 0, 1, \dots, T-1$ (here $T = 10$):

| Symbol | Meaning |
|---|---|
| $\mathbf{x}_j^t \in \mathbb{R}^2$ | **position**: the particle's current guess $(\log_{10} C, \log_{10}\gamma)$ |
| $\mathbf{v}_j^t \in \mathbb{R}^2$ | **velocity**: the step it will take next |
| $\mathbf{p}_j^t$ | **personal best**: the best position particle $j$ has ever visited |
| $\mathbf{g}^t$ | **global best**: the best position any particle has visited |
| $w$ | **inertia weight**, 0.7: how much of the previous velocity is kept |
| $c_1$ | **cognitive coefficient**, 1.5: strength of the pull toward the particle's own best |
| $c_2$ | **social coefficient**, 1.5: strength of the pull toward the swarm's best |
| $\mathbf{r}_1, \mathbf{r}_2$ | random vectors with each component $\sim \mathcal{U}(0, 1)$, drawn fresh for every particle, dimension and iteration |
| $\mathbf{v}_{\max}$ | velocity cap, $0.2\,(\mathbf{u} - \boldsymbol{\ell}) = (1, 1)$ |

### 3.2 The update equations

Each iteration, every particle updates its velocity and then its position:

$$
\boxed{
\mathbf{v}_j^{t+1} = w\, \mathbf{v}_j^{t}
+ c_1 \mathbf{r}_1 \odot \left( \mathbf{p}_j^{t} - \mathbf{x}_j^{t} \right)
+ c_2 \mathbf{r}_2 \odot \left( \mathbf{g}^{t} - \mathbf{x}_j^{t} \right)
}
$$

$$
\mathbf{v}_j^{t+1} \leftarrow \operatorname{clip}\left( \mathbf{v}_j^{t+1},\ -\mathbf{v}_{\max},\ \mathbf{v}_{\max} \right),
\qquad
\mathbf{x}_j^{t+1} = \operatorname{clip}\left( \mathbf{x}_j^{t} + \mathbf{v}_j^{t+1},\ \boldsymbol{\ell},\ \mathbf{u} \right).
$$

The three terms of the velocity are three "pulls":

| Term | Name | Meaning |
|---|---|---|
| $w\,\mathbf{v}_j^t$ | **inertia** | keep flying roughly the same way; drives exploration |
| $c_1 \mathbf{r}_1 \odot (\mathbf{p}_j - \mathbf{x}_j)$ | **cognitive / memory** | move back toward where *I* did best |
| $c_2 \mathbf{r}_2 \odot (\mathbf{g} - \mathbf{x}_j)$ | **social** | move toward where *the swarm* did best; drives convergence |

The random factors $\mathbf{r}_1, \mathbf{r}_2$ make each particle take a different, partly random path, so the swarm does not collapse into a single line.

### 3.3 Updating the bests

After all particles move and are scored (synchronous update):

$$
\mathbf{p}_j^{t+1} =
\begin{cases}
\mathbf{x}_j^{t+1} & \text{if } F(\mathbf{x}_j^{t+1}) > F(\mathbf{p}_j^{t}) \\
\mathbf{p}_j^{t} & \text{otherwise}
\end{cases}
\qquad
\mathbf{g}^{t+1} = \arg\max_{j} F\left( \mathbf{p}_j^{t+1} \right) \ \text{(kept only if it improves on } \mathbf{g}^t\text{)}.
$$

Because $\mathbf{g}$ only changes when a strictly better point is found, the best fitness found so far never decreases.

### 3.4 Initialisation and budget

- $\mathbf{x}_j^0 \sim \mathcal{U}(\boldsymbol{\ell}, \mathbf{u})$ and $\mathbf{v}_j^0 \sim \mathcal{U}(-\mathbf{v}_{\max}, \mathbf{v}_{\max})$.
- Initially $\mathbf{p}_j^0 = \mathbf{x}_j^0$ and $\mathbf{g}^0$ is the best of them.
- Scoring the initial swarm is iteration 1; then 9 more moves. Total: $M \times T = 10 \times 10 = 100$ evaluations.

### 3.5 Why these settings are stable

If $w$ is too large or $c_1 + c_2$ too big, particle velocities grow without bound and the swarm never settles. A standard theoretical result (Poli, 2009) gives the region in which the particles' positions converge (in mean and variance) while stagnating:

$$
-1 < w < 1, \qquad 0 < c_1 + c_2 < \frac{24\,(1 - w^2)}{7 - 5w}.
$$

With $w = 0.7$ the right-hand side is $24 \times 0.51 / 3.5 \approx 3.50$, and $c_1 + c_2 = 3.0$ lies inside it, so the project's settings are stable. The velocity cap $\mathbf{v}_{\max}$ and the position clipping add a practical safety net on top.

### 3.6 What each parameter controls

| Parameter | Value | Effect of increasing it |
|---|---|---|
| $M$ (particles) | 10 | broader coverage per iteration, fewer iterations within the budget |
| $w$ (inertia) | 0.7 | more exploration, slower settling. Step 8: $w = 0.9$ was clearly slower (53 trials on average vs 33). |
| $c_1$ (cognitive) | 1.5 | particles revisit their own good areas more |
| $c_2$ (social) | 1.5 | faster convergence on the leader, higher risk of settling on a local peak |
| $\mathbf{v}_{\max}$ | 20% of range | larger maximum step |

Step 8 also found that $c_1 = c_2 = 1.0$ and $c_1 = c_2 = 2.0$ both reached the target in 10 of 10 runs, so the default 1.5 is reasonable but not the best on this dataset.

---

## Part 4. GA and PSO side by side

| | GA | PSO |
|---|---|---|
| Inspiration | natural evolution | flocking birds |
| Each candidate has | a position (chromosome) and a fitness | a position, a velocity and a memory ($\mathbf{p}_j$) |
| How information is shared | indirectly, through selection and crossover | directly, through the global best $\mathbf{g}$ |
| Source of exploration | mutation, and crossover across distant parents | inertia and the random factors $\mathbf{r}_1, \mathbf{r}_2$ |
| Source of exploitation | selection pressure and elitism | the pulls toward $\mathbf{p}_j$ and $\mathbf{g}$ |
| Best found ever lost? | no (elitism) | no ($\mathbf{g}$ only improves) |
| Result in this project (30 runs) | reached the best level in 22/30 runs, median 31 trials when successful | reached the best level in 28/30 runs, median 34.5 trials |

Both are **derivative-free**: they only need $F(\boldsymbol{\theta})$ values, never gradients. That is exactly why they suit SVM tuning, where cross-validated accuracy moves in jumps of one patient and has no usable gradient.

---

## References

- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*(3), 273–297.
- Goldberg, D. E. (1989). *Genetic algorithms in search, optimization, and machine learning*. Addison-Wesley.
- Holland, J. H. (1975). *Adaptation in natural and artificial systems*. University of Michigan Press.
- Hsu, C.-W., Chang, C.-C., & Lin, C.-J. (2003). *A practical guide to support vector classification* [Technical report]. Department of Computer Science, National Taiwan University.
- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. In *Proceedings of ICNN'95* (Vol. 4, pp. 1942–1948). IEEE.
- Platt, J. C. (1998). *Sequential minimal optimization: A fast algorithm for training support vector machines* (Technical Report MSR-TR-98-14). Microsoft Research.
- Poli, R. (2009). Mean and variance of the sampling distribution of particle swarm optimizers during stagnation. *IEEE Transactions on Evolutionary Computation, 13*(4), 712–721.
