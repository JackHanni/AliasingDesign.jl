# Project Plan: Optimal Sampling to Control Aliasing

Theory of Predictive Modeling class project · Jack Hanni · started 2026-10-16

Choose where to sample so that a linear model's fit suffers as little as possible from
aliasing, measured either by the aliasing operator or by the full prediction risk. Calculus of variations gives the gradient and optimality condition of a cost over
sampling densities; a Fisher–Rao gradient flow finds the optimal density; quantiles turn it
into $n$ points; a Wasserstein particle flow polishes them at finite $n$. The deliverable is
the Julia package [AliasingDesign.jl](../README.md): given basis functions, it returns the
optimal density, sample points, or design matrix.

## The learning problem

Predict the formation energy $E(\sigma)$ of an alloy configuration from a few computed (DFT)
energies with a linear cluster expansion, $E \approx M\theta$, fit by least squares. The
question is *which* configurations to put in the training set. A 1D polynomial benchmark,
where the answer (Gauss-like points) is known, comes first; an equation-of-state fit (which
volumes to compute) tests the method on a continuous physical input before the discrete
cluster expansion.

| Course requirement           | Where it appears                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------------------ |
| Models something             | Alloy formation energy (cluster expansion); 1D polynomial benchmark                        |
| Learning problem + algorithm | Least-squares fit on a training set chosen by gradient flows                               |
| Identifiability              | $I-B$ = unidentifiable directions; aliasing = confounding of modeled and unmodeled terms |
| Uncertainty quantification   | The prior $\Sigma$ gives expected error and predictive error bars                        |
| Dynamics                     | The design flows: fixed points, convergence rates, multiple optima                         |
| Abstraction                  | Density $\rho$ abstracts $n$ points; truncated basis abstracts the true model          |

## Background: the GAD

**Basis functions.** An input $t$ is a point of the domain: a number in $[-1, 1]$ for the 1D
benchmark, a volume for the equation of state, or an alloy configuration for the cluster
expansion. The true model is linear in a
set of basis functions, split into $p$ modeled ($\mathcal M$, kept in the fit) and $q$ unmodeled
($\mathcal U$, left out). Collect them into vector-valued functions of $t$:

$$
\phi_{\mathcal M}(t) = \begin{bmatrix} \phi_1(t) \\ \vdots \\ \phi_p(t) \end{bmatrix} \in \mathbb R^{p},
\qquad
\phi_{\mathcal U}(t) = \begin{bmatrix} \phi_{p+1}(t) \\ \vdots \\ \phi_{p+q}(t) \end{bmatrix} \in \mathbb R^{q},
\qquad
y(t) = \phi_{\mathcal M}(t)^\top \theta_{\mathcal M} + \phi_{\mathcal U}(t)^\top \theta_{\mathcal U} \tag{1}
$$

**Design matrices.** Evaluate the basis at $n$ training points $t_1, \dots, t_n$ ($\mathcal T$)
and $n_{\mathcal P}$ prediction points $x_1, \dots, x_{n_{\mathcal P}}$ ($\mathcal P$). Rows are
samples and columns are basis functions: row $i$ of $M_{\mathcal{TM}}$ is
$\phi_{\mathcal M}(t_i)^\top$ and row $i$ of $M_{\mathcal{TU}}$ is $\phi_{\mathcal U}(t_i)^\top$, so
$M_{\mathcal{TM}}$ is $n \times p$ and $M_{\mathcal{TU}}$ is $n \times q$; the prediction blocks
$M_{\mathcal{PM}}$, $M_{\mathcal{PU}}$ use the $x_k$ the same way. Then:

$$
\begin{bmatrix} M_{\mathcal{TM}} & M_{\mathcal{TU}} \\ M_{\mathcal{PM}} & M_{\mathcal{PU}} \end{bmatrix}
\begin{bmatrix} \theta_{\mathcal M} \\ \theta_{\mathcal U} \end{bmatrix}
= \begin{bmatrix} y_{\mathcal T} \\ y_{\mathcal P} \end{bmatrix} \tag{2}
$$

Fitting only the modeled block to the training rows gives

$$
\hat\theta_{\mathcal M} = M_{\mathcal{TM}}^{+} y_{\mathcal T} = B\,\theta_{\mathcal M} + A\,\theta_{\mathcal U},
\qquad B = M_{\mathcal{TM}}^{+} M_{\mathcal{TM}}, \qquad A = M_{\mathcal{TM}}^{+} M_{\mathcal{TU}} \tag{3}
$$

**Gram matrices.** Define the Gram matrices of the basis functions on the training samples ($\{\{\phi_i(t_j)\}_i\}_j$),

$$
G_{\mathcal{MM}} = \frac1n M_{\mathcal{TM}}^\top M_{\mathcal{TM}} = \frac1n \sum_{i=1}^{n} \phi_{\mathcal M}(t_i)\,\phi_{\mathcal M}(t_i)^\top,
\qquad
G_{\mathcal{MU}} = \frac1n M_{\mathcal{TM}}^\top M_{\mathcal{TU}} = \frac1n \sum_{i=1}^{n} \phi_{\mathcal M}(t_i)\,\phi_{\mathcal U}(t_i)^\top \tag{4}
$$

$G_{\mathcal{MM}}$ ($p \times p$) holds the average overlaps of the modeled basis functions over
the training points. $G_{\mathcal{MU}}$ ($p \times q$) holds the overlaps between modeled and
unmodeled functions; it is what lets unmodeled signal leak into the fit. When $n \ge p$ and
$M_{\mathcal{TM}}$ has full column rank, $M_{\mathcal{TM}}^{+} = (M_{\mathcal{TM}}^\top M_{\mathcal{TM}})^{-1} M_{\mathcal{TM}}^\top$,
the factors of $1/n$ cancel, and

$$
A = G_{\mathcal{MM}}^{-1} G_{\mathcal{MU}}, \qquad B = I \tag{5}
$$

When $n < p$, $G_{\mathcal{MM}}$ is singular: $B$ is a projection, $I - B \ne 0$, and $A$ must be
computed with the pseudoinverse as in (3).

Predicting with $\hat y_{\mathcal P} = M_{\mathcal{PM}}\hat\theta_{\mathcal M}$ and substituting (3) gives the prediction error

$$
y_{\mathcal P} - \hat y_{\mathcal P}
= M_{\mathcal{PM}}\theta_{\mathcal M} + M_{\mathcal{PU}}\theta_{\mathcal U} - M_{\mathcal{PM}}\big(B\theta_{\mathcal M} + A\theta_{\mathcal U}\big)
= M_{\mathcal{PM}}(I - B)\,\theta_{\mathcal M} + \big(M_{\mathcal{PU}} - M_{\mathcal{PM}}A\big)\,\theta_{\mathcal U}
= M_{\mathcal P} E_\theta\, \theta \tag{6}
$$

with $M_{\mathcal P} = [\,M_{\mathcal{PM}}\ \ M_{\mathcal{PU}}\,]$ and

$$
E_\theta = \begin{bmatrix} I - B & -A \\ 0 & I_{\mathcal U} \end{bmatrix} \tag{7}
$$

| Term                | Operator           | Meaning                                                                        |
| ------------------- | ------------------ | ------------------------------------------------------------------------------ |
| Data insufficiency  | $I - B$          | Modeled directions the data cannot pin down (zero when $n \ge p$, full rank) |
| Aliasing            | $A$              | Unmodeled signal leaking into the fitted parameters                            |
| Model insufficiency | $I_{\mathcal U}$ | Signal the model cannot represent                                              |

All three depend only on the design, not the labels, so the design can be optimized before any
data is collected. Reference: Transtrum, Hart, Jarvis, Whitehead, arXiv:2408.08294.

## The cost

Put a prior on all coefficients: $\mathbb E[\theta] = 0$ and $\operatorname{Cov}(\theta) = \Sigma$,
with no other distributional assumption ($\Sigma_{\mathcal U}$ = its unmodeled block). The
project studies two costs, built from the prior-weighted aliasing operator and the risk
operator:

$$
X_{\text{alias}} = A\,\Sigma_{\mathcal U}^{1/2},
\qquad
X_{\text{risk}} = R = \frac{1}{\sqrt{n_{\mathcal P}}}\, M_{\mathcal P} E_\theta\, \Sigma^{1/2} \tag{8}
$$

$$
S_\mu = \lVert X_\mu \rVert^2, \tag{9}
$$

where $S$ is the cost which is equal to the norm of either $X$, and $||\cdot||$ is either the Frobenius or spectral norm.

**Why both.** Aliasing measures how badly the fitted parameters are contaminated. Risk measures
the prediction error itself: it also includes data insufficiency ($I-B$, which matters for
$n < p$), model insufficiency, and where predictions are made ($M_{\mathcal P}$). The design that
minimizes aliasing need not minimize risk; comparing the two optima is a result of the project.

**Choosing the prediction points.** Optimizing the risk is not fitting to the prediction data:
the cost uses only the prediction *inputs* $x_k$ (through $M_{\mathcal P}$) and the prior, never
the labels $y_{\mathcal P}$, and the model is still fit to the training rows alone. $M_{\mathcal P}$
only says where prediction error matters, as in I-optimal design. The design can, however,
specialize to a small or unrepresentative prediction set, so $M_{\mathcal P}$ should cover where
future predictions will be made: a dense grid over the domain for the 1D benchmark, and the full
candidate pool for the cluster expansion. To check, evaluate each optimized design on a second,
independent set of prediction points; its advantage should persist.

The two norms read differently. Frobenius is an expectation that uses only the first two moments
of $\theta$, since $\mathbb E\lVert Y\theta\rVert^2 = \operatorname{tr}(Y\Sigma Y^\top)$ for any
matrix $Y$. The operator norm is not probabilistic: with $\theta = \Sigma^{1/2}z$, it is the worst
case over the ellipsoid of plausible coefficients,
$\max_{\theta^\top\Sigma^{-1}\theta = 1} \lVert Y\theta\rVert^2$.
$S$ plays the role of an action: it is the functional minimized by calculus of variations.
The operator norm is not differentiable where the top two singular values cross. The Frobenius
version is smooth, so use it as a check and as a fallback when the operator-norm flow stalls at
a crossing.

**Expected aliasing, and why not the paper's $\lVert A\rVert$.** The GAD paper measures aliasing
by the label-free aliasing norm $\lVert A\rVert_2 = \sigma_{\max}(A)$ (as in JuCE's reproduction of
its figures). That is the most a *unit-size* vector of unmodeled coefficients can leak into the fit:
it weights every unmodeled coefficient equally. Here the unmodeled coefficients have a prior, and the
aliasing error in the fitted parameters is $A\theta_{\mathcal U}$, so its expected size is

$$
\mathbb E\lVert A\theta_{\mathcal U}\rVert^2 = \operatorname{tr}\big(A\Sigma_{\mathcal U}A^\top\big)
= \lVert A\Sigma_{\mathcal U}^{1/2}\rVert_F^2 \tag{9′}
$$

the Frobenius cost (9). This reduces to $\lVert A\rVert_F^2$ only when $\Sigma_{\mathcal U} = I$. In the
cluster expansion the prior shrinks with cluster size and diameter, so an unweighted norm lets a
column carrying a tiny 8-body coefficient count as much as one carrying a large nearest-neighbor
triplet; $\Sigma_{\mathcal U}^{1/2}$ scales each column of $A$ by the size of the coefficient that leaks
through it, turning "how much *could* leak" into "how much leaks on average". In the experiments,
**expected aliasing** is this quantity plus the noise term of (9a),
$\lVert A\Sigma_{\mathcal U}^{1/2}\rVert_F^2 + \sigma_\epsilon^2\lVert M_{\mathcal{TM}}^+\rVert_F^2$, which is exactly
the expectation of the **parameter aliasing** $\lVert\hat\theta_{\mathcal M} - B\theta_{\mathcal M}\rVert^2$
measured on fits to synthetic data (subtracting $B\theta_{\mathcal M}$ removes data insufficiency, so
this isolates aliasing for any $n$ and $p$). The paper's $\lVert A\rVert_2$ is plotted alongside, for
direct comparison with its figures.

**Designs as distributions.** A design of $n$ points is the distribution that puts weight $1/n$
on each point, and (4) is the Gram matrix under that distribution. The plan optimizes over *all*
distributions instead: densities $\rho$ on a continuous domain, or weights $w$ on a finite
pool. This is a relaxation of the real problem, choosing $n$ points, and it is justified
differently for the two kinds of domain (see *Why the relaxation is justified* below). For a
density, the Gram matrices become averages over $\rho$:

$$
G_{\mathcal{MM}} \to G_{\mathcal{MM}}[\rho] = \int \phi_{\mathcal M}(t)\,\phi_{\mathcal M}(t)^\top \rho(t)\,dt, \qquad
G_{\mathcal{MU}} \to G_{\mathcal{MU}}[\rho] = \int \phi_{\mathcal M}(t)\,\phi_{\mathcal U}(t)^\top \rho(t)\,dt \tag{10}
$$

The meanings are unchanged; the overlaps are now weighted by $\rho$ instead of counted over
points. As long as $G_{\mathcal{MM}}[\rho]$ is invertible (at least $p$ support points in general
position), $B = I$ and the design enters only through $\rho$:

$$
A[\rho] = G_{\mathcal{MM}}[\rho]^{-1} G_{\mathcal{MU}}[\rho], \qquad B = I \tag{11}
$$

On a grid or candidate pool with weights $w_i \ge 0$, $\sum_i w_i = 1$, the integrals become
$\sum_i w_i\,\phi(t_i)\phi(t_i)^\top$. Equal weights $w_i = 1/n$ on $n \ge p$ points recover the
sample Gram matrices *exactly*, so (11) is then the finite-$n$ aliasing operator (3), not an
approximation of it. The risk operator keeps only its unmodeled block,
$R = n_{\mathcal P}^{-1/2}\,(M_{\mathcal{PU}} - M_{\mathcal{PM}} A)\,\Sigma_{\mathcal U}^{1/2}$.

**Why the relaxation is justified.** Assume $G_{\mathcal{MM}}$ is invertible at the designs
considered. Then $A$ and every cost are continuous functions of the Gram matrices, so it is
enough to know when the Gram matrices of $n$-point designs approach those of the relaxed
optimum, and how much the optimum moves.

- *Continuous domain (1D benchmark, Fourier, equation of state): a limit.* If the $n$ points
  are drawn independently from $\rho$, the strong law of large numbers gives
  $G_n \to G[\rho]$ almost surely (finite second moments of $\phi$), and the continuous mapping
  theorem carries this to $A$ and $S$; the central limit theorem gives errors of order
  $n^{-1/2}$, and $A_n$ has a bias of order $1/n$. If the points are placed deterministically
  (step 3), the empirical distribution converges weakly to $\rho$, and on a compact domain with
  continuous $\phi$ this gives the same limits; in 1D the Koksma–Hlawka inequality bounds the
  error by (variation of $\phi_j\phi_k$) $\times$ (discrepancy), and midpoint quantiles have
  discrepancy $1/(2n)$, so the error is $O(1/n)$. That the *optimal* designs also converge is a
  Γ-convergence argument: probability measures on a compact domain are weakly compact, $n$-point
  designs become dense in them, and $S$ is weakly continuous wherever $G_{\mathcal{MM}}$ stays
  invertible. It fails near designs with fewer than $p$ support points, which is exactly the
  $n \approx p$ regime that step 4 handles with the exact finite-$n$ cost.
  Caveat: almost-sure convergence of $S$ does not give convergence of its expectation over
  random designs; near $n = p$, $G_{\mathcal{MM}}^{-1}$ can be large with non-negligible
  probability.
- *Finite pool (cluster expansion): no limit, a relaxation.* The pool is finite, so the sum
  over it is exact and there is no $n \to \infty$. The real problem is to choose $n$ distinct
  structures: weights $w_t \in \{0, 1/n\}$. The relaxation lets each weight vary continuously
  in $[0, 1/n]$ (the capped simplex of the *Discrete domains* section). It is exact at the
  vertices: an $n$-structure design's relaxed cost *is* its finite-$n$ cost (for $n \ge p$).
  The approximation is only in rounding the relaxed optimum back to a vertex. Because every
  $n$-structure design is feasible for the relaxation, the relaxed minimum is a lower bound on
  the best $n$-structure cost, so (rounded cost) $-$ (relaxed minimum) bounds how far the rounded
  design is from optimal. Caveat: this needs the *global* relaxed minimum, and unlike the
  classical A- and D-optimal criteria, $S$ is not convex in $w$ (checked: on `monomial_basis(7)`,
  $p = 4$, the midpoint of two random weight vectors had a higher cost than the endpoints'
  average in about 60% of 20,000 trials). Run the flow from several starts and treat the bound as
  empirical. Rounding guarantees from the optimal-design literature (e.g. Allen-Zhu et al.) are
  proved for convex criteria and do not transfer directly.

**Adding measurement noise.** The costs (8)–(9) contain only the error from unmodeled terms
(bias). Without a noise term the relaxed problem can be degenerate (see *Findings from the first
optimizer runs*): monomial aliasing has no minimizer, since collapsing the design onto $t = 0$
drives it to zero while $G_{\mathcal{MM}}$ becomes singular, and Frobenius risk is minimized
trivially by sampling the prediction distribution. With independent zero-mean noise of variance
$\sigma_\epsilon^2$ and $n$ samples in proportions $\rho$, $\operatorname{Cov}(\hat\theta_{\mathcal M}) = \frac{\sigma_\epsilon^2}{n}G_{\mathcal{MM}}^{-1}$, so the
expected squared error gains a variance term. With $\tau = \sigma_\epsilon^2/n$:

$$
S_\tau = S + \tau V, \qquad
V_{\text{alias}} = \operatorname{tr}\big(G_{\mathcal{MM}}^{-1}\big), \qquad
V_{\text{risk}} = \operatorname{tr}\big(G_{\mathcal{MM}}^{-1}G_{\mathcal P}\big), \quad G_{\mathcal P} = \tfrac{1}{n_{\mathcal P}}M_{\mathcal{PM}}^\top M_{\mathcal{PM}} \tag{9a}
$$

$V_{\text{alias}}$ is the A-optimality criterion and $V_{\text{risk}}$ the I-optimality criterion,
so $S_\tau$ trades aliasing (bias) against classical optimal design (variance). The noise term
is what the uncertainty-quantification formula (30) adds. Its first variations (derived in the
[primer](calculus_of_variations_primer.md), Problem 4) are

$$
\frac{\delta V_{\text{alias}}}{\delta\rho}(t) = -\big\lVert G_{\mathcal{MM}}^{-1}\phi_{\mathcal M}(t)\big\rVert^2, \qquad
\frac{\delta V_{\text{risk}}}{\delta\rho}(t) = -\tfrac{1}{n_{\mathcal P}}\big\lVert M_{\mathcal{PM}}G_{\mathcal{MM}}^{-1}\phi_{\mathcal M}(t)\big\rVert^2 \tag{9b}
$$

Any $\tau > 0$ rules out the collapse, because $V \to \infty$ as $G_{\mathcal{MM}}$ becomes
singular. The design depends on $\tau$, so report results for a range of $\tau$ (a physical
$\sigma_\epsilon$ and $n$ where one exists), with $\tau \to 0$ the pure-aliasing limit and
$\tau \to \infty$ the classical A-/I-optimal design.

## Step 1 — Calculus of variations: gradient and optimality condition

Step 1 supplies the gradient for step 2 and the condition
that certifies step 2's answer. A line-by-line derivation of (12)–(16), (20) and (22), with
practice problems, is in the [calculus of variations primer](calculus_of_variations_primer.md).

**First variation.** $\delta S/\delta\rho(t)$ is the change in $S$ from adding sampling weight
at $t$:

$$
S[\rho + \varepsilon\eta] = S[\rho] + \varepsilon \int \frac{\delta S}{\delta\rho}(t)\,\eta(t)\,dt + O(\varepsilon^2) \tag{12}
$$

$$
\delta A = G_{\mathcal{MM}}^{-1}\,\delta G_{\mathcal{MU}} - G_{\mathcal{MM}}^{-1}\,\delta G_{\mathcal{MM}}\,G_{\mathcal{MM}}^{-1}G_{\mathcal{MU}}
= G_{\mathcal{MM}}^{-1}\big(\delta G_{\mathcal{MU}} - \delta G_{\mathcal{MM}}\,A\big)
= \varepsilon \int G_{\mathcal{MM}}^{-1}\phi_{\mathcal M}(t)\big(\phi_{\mathcal U}(t)^\top - \phi_{\mathcal M}(t)^\top A\big)\,\eta(t)\,dt \tag{13}
$$

Comparing with the definition (12), the integrand is the first variation:

$$
\frac{\delta A}{\delta\rho}(t) = G_{\mathcal{MM}}^{-1}\phi_{\mathcal M}(t)\, e_{\mathcal U}(t)^\top,
\qquad e_{\mathcal U}(t)^\top = \phi_{\mathcal U}(t)^\top - \phi_{\mathcal M}(t)^\top A \tag{14}
$$

where $e_{\mathcal U}(t)$ is what the model cannot reproduce of the unmodeled features at $t$.
Both norms reduce to one formula, with $K = X$ (Frobenius) or $K = \sigma_1 u_1 v_1^\top$
(operator norm, top singular triple of $X$):

$$
\frac{\delta S_{\text{alias}}}{\delta\rho}(t) = 2\,\big(G_{\mathcal{MM}}^{-1}\phi_{\mathcal M}(t)\big)^{\!\top} K\, \Sigma_{\mathcal U}^{1/2} e_{\mathcal U}(t),
\qquad
\frac{\delta S_{\text{risk}}}{\delta\rho}(t) = -\frac{2}{\sqrt{n_{\mathcal P}}}\,\big(M_{\mathcal{PM}} G_{\mathcal{MM}}^{-1}\phi_{\mathcal M}(t)\big)^{\!\top} K\, \Sigma_{\mathcal U}^{1/2} e_{\mathcal U}(t) \tag{15}
$$

For Frobenius aliasing this is $2\,\phi_{\mathcal M}^\top G_{\mathcal{MM}}^{-1} A \Sigma_{\mathcal U} e_{\mathcal U}(t)$.

**Optimality (KKT) condition**, with multiplier $\lambda$ for $\int \rho(t)\,dt = 1$:

$$
\frac{\delta S}{\delta\rho}(t) = \lambda \ \text{ where } \rho > 0, \qquad
\frac{\delta S}{\delta\rho}(t) \ge \lambda \ \text{ where } \rho = 0 \tag{16}
$$

Same form as the Kiefer–Wolfowitz equivalence theorem. It cannot be solved in closed form in
general, because $S$ depends on $\rho$ through $G_{\mathcal{MM}}^{-1}$.

## Step 2 — Fisher–Rao gradient flow: find the optimal density

$\delta S/\delta\rho$ is not yet a direction: a metric must say which $\eta$ is a small step.
Fisher–Rao measures relative change in weight:

$$
\lVert \eta \rVert_{\rho}^2 = \int \frac{\eta(t)^2}{\rho(t)}\,dt
\qquad\Longrightarrow\qquad
\partial_\tau \rho(t) = -\rho(t)\left(\frac{\delta S}{\delta\rho}(t) - \mathbb E_\rho\!\left[\frac{\delta S}{\delta\rho}\right]\right) \tag{17}
$$

**Derivation of (17).** Write $g(t) = \frac{\delta S}{\delta\rho}(t)$.

*1. Tangent directions.* A small change of $\rho$ that keeps it a density is a direction $\eta$
with $\int \eta(t)\,dt = 0$, as in step 1. These directions form the tangent space at $\rho$, and
the metric is the inner product on it,
$\langle \eta, \xi \rangle_\rho = \int \eta(t)\,\xi(t)/\rho(t)\,dt$.

*2. The metric is the Fisher information.* For a parameterized family of densities $\rho_\vartheta$,
the tangent directions are $\partial_i \rho = \partial\rho/\partial\vartheta_i$, and

$$
\langle \partial_i\rho, \partial_j\rho \rangle_\rho
= \int \frac{\partial_i\rho(t)\,\partial_j\rho(t)}{\rho(t)}\,dt
= \int \rho(t)\,\partial_i \log\rho(t)\,\partial_j \log\rho(t)\,dt
= F_{ij}(\vartheta) \tag{18}
$$

so the Fisher–Rao metric is the Fisher information matrix of the design density, used as the
metric tensor. (This is the FIM of $\rho$ as a distribution over inputs, not the regression FIM.)

*3. Gradient = inverse metric applied to the differential.* The differential of $S$ in direction
$\eta$ is the first variation, $dS[\eta] = \int g(t)\,\eta(t)\,dt$ from (12). The gradient is the
tangent vector $v$ that represents it through the metric, $\langle v, \eta\rangle_\rho = dS[\eta]$
for every tangent $\eta$:

$$
\int \frac{v(t)}{\rho(t)}\,\eta(t)\,dt = \int g(t)\,\eta(t)\,dt
\qquad \text{for all } \eta \text{ with } \int \eta(t)\,dt = 0 \tag{19}
$$

The difference $v/\rho - g$ is orthogonal to every zero-mean $\eta$, so it is a constant $c$:
$v = \rho\,(g - c)$. Requiring $v$ to be tangent ( $\int v(t)\,dt = 0$ ), with $\int\rho(t)\, dt = 1$ gives
$c = \int \rho(t)\,g(t)\,dt = \mathbb E_\rho[g]$:

$$
\operatorname{grad}_\rho S\,(t) = \rho(t)\big(g(t) - \mathbb E_\rho[g]\big) \tag{20}
$$

This is the infinite-dimensional form of $(\operatorname{grad} S)^i = F^{ij}\,\partial_j S$:
inverting the metric (dividing by $1/\rho$) multiplies $g$ by $\rho$, and subtracting the mean
projects onto the tangent space. Equivalently, the steepest step of size $h$ solves

$$
\eta^\star = \arg\min_{\int\eta\,dt = 0} \left[\int g(t)\,\eta(t)\,dt + \frac{1}{2h}\int \frac{\eta(t)^2}{\rho(t)}\,dt\right]
= -h\,\operatorname{grad}_\rho S \tag{21}
$$

*4. The flow.* Gradient descent in continuous time, $\partial_\tau\rho = -\operatorname{grad}_\rho S$,
is (17).

*5. Checks.* Mass is conserved, $\frac{d}{d\tau}\int\rho\,dt = -\int \rho\,(g - \mathbb E_\rho g)\,dt = 0$,
and the factor $\rho$ keeps zero weights at zero, so $\rho$ stays a density. The cost decreases:

$$
\frac{dS}{d\tau} = \int g(t)\,\partial_\tau\rho(t)\,dt
= -\int \rho(t)\big(g(t) - \mathbb E_\rho[g]\big)^2 dt
= -\operatorname{Var}_\rho(g) \le 0 \tag{22}
$$

(the $\mathbb E_\rho[g]$ term integrates to zero). The flow stops exactly when $g$ is constant on
the support of $\rho$: the equality part of the KKT condition (16). The inequality part, where
$\rho = 0$, is not enforced, because zero weights never grow back. So start with $\rho > 0$
everywhere and check (16) at the end.

On a grid or candidate pool with weights $w_i$ and $g_i = \delta S/\delta\rho(t_i)$, this is
exponentiated gradient (mirror descent). To first order in $h$, the update below is
$w_i\big(1 - h\,(g_i - \sum_j w_j g_j)\big)$, a forward-Euler step of (17) that keeps the weights
exactly positive:

$$
w_i \leftarrow \frac{w_i\, e^{-h g_i}}{\sum_j w_j\, e^{-h g_j}} \tag{23}
$$

Why Fisher–Rao: weights stay positive and normalized with no projection; the steps do not
depend on how $\rho$ is parameterized (it is the unique invariant metric, Chentsov); and it
works on discrete pools, where points cannot move (cluster expansion). Stop when the KKT
condition holds to tolerance.

**Step size in practice** (from the first optimizer runs). Set $h = \eta/\max_i\lvert g_i - \lambda\rvert$
once, from the first gradient, and keep it fixed; halve it whenever a step would raise the cost.
Recomputing this normalization every step keeps the step length constant as $g \to 0$, and the
weights then oscillate around the optimum without converging. Iterations scale like $1/\eta$;
$\eta \approx 8$–$16$ was safe and fast on the 1D problems. Measure convergence against
$\max(\max_i\lvert g_i\rvert,\ S)$, not $\max_i\lvert g_i\rvert$ alone: at an exact optimum $g$
can be pure rounding noise. Levenberg–Marquardt (in log-weights, Frobenius costs only) and an
active-set Newton method were tested as faster alternatives and not adopted: from uniform weights
both failed or fell into degenerate designs, and as polishers after Fisher–Rao they jumped to
different, worse local optima instead of refining the solution.

Not to be confused with the regression Fisher information
$M_{\mathcal{TM}}^\top M_{\mathcal{TM}}/\sigma^2$, which lives on parameter space and is used
in the identifiability analysis.

**Comparison geometry: Wasserstein** measures how far mass moves, and its flow transports
points instead of reweighting them:

$$
\partial_\tau \rho(t) = \partial_t\!\left(\rho(t)\,\partial_t \frac{\delta S}{\delta\rho}(t)\right) \tag{24}
$$

When the optimum is not unique, the two geometries can reach different optimal designs.
$S$ is not convex in $\rho$ in general (see *Why the relaxation is justified*), so a flow can
stop at a local optimum; run it from several starts.

## Step 3 — Discretize: density to $n$ points

In 1D, use evenly spaced quantiles of $F(t) = \int_{-\infty}^{t} \rho(s)\,ds$, not random draws.
Random draws add sampling noise, which is largest near $n \approx p$:

$$
t_i = F^{-1}\!\left(\frac{i - 1/2}{n}\right), \qquad i = 1,\dots,n \tag{25}
$$

Example: the arcsine density $\rho(t) = 1/(\pi\sqrt{1-t^2})$ gives exactly the Chebyshev nodes
$t_i = \cos\big((2i-1)\pi/2n\big)$.

A discrete pool has no quantile function; see *Discrete domains* below for rounding a pmf to $n$
distinct structures.

## Step 4 — Finite-$n$ polish and evaluation

The density is optimal only for $n \gg p$. Starting from the step-3 points, minimize the exact
finite-$n$ cost $S(t_1,\dots,t_n)$, with $A = M_{\mathcal{TM}}^{+}M_{\mathcal{TU}}$ and
$I - B$ computed from the points, by gradient descent on the points. This is the particle form
of the Wasserstein flow:

$$
t_i \leftarrow t_i - h\,\frac{\partial S}{\partial t_i} \tag{26}
$$

It needs no relaxation to densities, so it handles $n \approx p$ and the $I - B$ term.
The pseudoinverse is differentiable only while $\operatorname{rank} M_{\mathcal{TM}}$ is
constant; add a small ridge near rank changes. Then build the design matrix and evaluate.

## Discrete domains: the cluster expansion

Alloy configurations are discrete, so there is no density over them. The fix is to replace the
density by a probability mass function (pmf) on the candidate pool $\mathcal C$ ($N = 2{,}346$
configurations), not to bin a continuous pdf: configurations are not points of a continuum, so
there is nothing to round to. Steps 1–2 carry over with every integral $\int \cdot\,\rho(t)\,dt$
replaced by a sum $\sum_{t\in\mathcal C} \cdot\, w_t$:

$$
G_{\mathcal{MM}}[w] = \sum_{t\in\mathcal C} w_t\,\phi_{\mathcal M}(t)\,\phi_{\mathcal M}(t)^\top, \qquad
G_{\mathcal{MU}}[w] = \sum_{t\in\mathcal C} w_t\,\phi_{\mathcal M}(t)\,\phi_{\mathcal U}(t)^\top, \qquad
\sum_{t\in\mathcal C} w_t = 1 \tag{27}
$$

The first variation becomes the partial derivative $g_t = \partial S/\partial w_t$, given by the
same formula (15) evaluated at $t$. The Fisher–Rao metric becomes
$\lVert\eta\rVert_w^2 = \sum_t \eta_t^2 / w_t$ on the probability simplex, and (17) becomes
$\dot w_t = -w_t\,(g_t - \sum_s w_s g_s)$, which is exactly the mirror-descent update (23).
Fisher–Rao is the natural geometry here: it was built for pmfs.

**Each configuration counts once.** DFT energies are deterministic, so computing the same
structure twice adds no information. A design of $n$ distinct structures puts weight $1/n$ on
each, so cap the weights at $w_t \le 1/n$. The KKT condition (16) gains a case for the cap:

$$
g_t = \lambda \ \text{ where } 0 < w_t < \tfrac1n, \qquad
g_t \ge \lambda \ \text{ where } w_t = 0, \qquad
g_t \le \lambda \ \text{ where } w_t = \tfrac1n \tag{28}
$$

Enforce the cap after each update by clipping weights at $1/n$ and renormalizing the rest.

**Step 3 for a pool: rounding.** There are no quantiles. Select the $n$ configurations with the
largest weights; with the cap, the optimum is expected to put most weight at exactly 0 or
$1/n$, so this rounding should lose little. Compare with sampling $n$ distinct structures from
$w$. Report the gap (rounded cost) $-$ (relaxed minimum) after rounding and after exchange: it
bounds the distance to the best $n$-structure design, provided the relaxed minimum is global
(see *Why the relaxation is justified*).

**Step 4 for a pool: exchange.** Points cannot move, so the particle flow is replaced by the
discrete analogue from classical optimal design (Fedorov exchange): repeatedly swap a selected
structure for an unselected one when the swap lowers the exact finite-$n$ cost, until no swap
helps. The first variation ranks the candidates, so only the most promising swaps need checking.

## Worked toy example

Model a constant ($\phi_{\mathcal M} = 1$) when the truth also has a slope ($\phi_{\mathcal U} = t$,
$\operatorname{Var}\theta_1 = \sigma^2$). With $\bar t = \mathbb E_\rho[t]$:

$$
A = \bar t, \qquad S_{\text{alias}} = \sigma^2 \bar t^{\,2}, \qquad
\frac{\delta S_{\text{alias}}}{\delta\rho}(t) = 2\sigma^2 \bar t\,(t - \bar t), \qquad
S_{\text{risk}} = \sigma^2\, \frac{1}{n_{\mathcal P}}\sum_k (x_k - \bar t)^2 \tag{29}
$$

Every density with $\bar t = 0$ is optimal. The two flows reach different ones:

| Flow        | Mean dynamics                                                    | End state                                  |
| ----------- | ---------------------------------------------------------------- | ------------------------------------------ |
| Fisher–Rao | $\dot{\bar t} = -2\sigma^2 \operatorname{Var}_\rho(t)\,\bar t$ | Same points, reweighted to $\bar t = 0$  |
| Wasserstein | $\dot{\bar t} = -2\sigma^2 \bar t$                             | Same shape, slid rigidly to $\bar t = 0$ |

## Findings from the first optimizer runs

Run on 2026-11-12 with the Fisher–Rao flow on a 101-point grid over $[-1, 1]$, Frobenius costs,
$p = 4$ modeled and 3 unmodeled functions, prior $1/k^2$; risk predictions on 50 evenly spaced
points. Each problem was run from 6 starts (uniform weights and 5 random ones). These are
observations on these problems, not general results.

1. **Monomial aliasing has no minimizer.** Shrinking a 4-point Chebyshev design toward $t = 0$ by
   a factor $\varepsilon$ sends the aliasing cost to zero (0.085 at $\varepsilon = 1$,
   $6.8\times10^{-10}$ at $\varepsilon = 0.01$; checked in 512-bit arithmetic) while
   $\operatorname{cond}(G_{\mathcal{MM}})$ grows from 65 to $3\times10^{13}$. Reason: $t^k$ with
   $k \ge p$ vanishes to high order at 0, so it barely aliases there. The noiseless optimum is a
   degenerate collapse, and fast optimizers find it; the 6 starts ended at 6 different costs
   ($6.4\times10^{-6}$ to $4.8\times10^{-3}$), all clustered within $|t| \le 0.1$.
2. **Frobenius risk is trivial without noise.** Each column of $A[\rho]$ is the least-squares
   projection under $\rho$, and the Frobenius risk is that same least-squares error under the
   prediction distribution, so training on the prediction distribution is a global optimum
   (first variation $2.7\times10^{-18}$ there). This is the covariate-shift result: without noise,
   sample where you will predict.
3. **The noise term (9a) fixes both.** Along the shrinking Chebyshev design, the best scale moves
   from $\varepsilon = 0.02$ at $\tau = 0$ to 0.3 at $\tau = 10^{-8}$, 0.5 at $10^{-6}$, and 1 for
   $\tau \ge 10^{-4}$. Full optimization:

   | Problem | $\tau$ | Distinct optima (of 6 starts) | Support ($t \ge 0$) | $\operatorname{cond}(G)$ | bias / $\tau V$ |
   |---|---|---|---|---|---|
   | Monomial aliasing | 0 | 6 | $0$–$0.10$ (collapse) | $5\times10^{7}$ | — |
   | | $10^{-6}$ | 1 | $\approx 0.17,\ 0.45$ | $2\times10^{3}$ | $3.4\times10^{-3}$ / $2.7\times10^{-3}$ |
   | | $10^{-4}$ | 1 | $\approx 0.27,\ 0.71$ | $2\times10^{2}$ | $2.3\times10^{-2}$ / $2.1\times10^{-2}$ |
   | | $\ge 10^{-2}$ | 1 | $\approx 0.43$–$0.48,\ 1$ | 30 | variance-dominated |
   | Monomial risk | $\le 10^{-4}$ | 1 | whole grid (prediction distribution) | 60 | — |
   | | $\ge 10^{-2}$ | 1 | $\approx 0.42$–$0.46,\ 1$ | 30 | variance-dominated |
   | Bumps aliasing | 0 | 6 | one numerically singular start ($\operatorname{cond} \sim 10^{113}$) | — | — |
   | | $10^{-6}$ | 6 | 35 points | $7\times10^{3}$ | $2.1\times10^{-2}$ / $4.2\times10^{-3}$ |
   | | $10^{-4}$ | 4 | 7 points | $6\times10^{2}$ | $4.5\times10^{-2}$ / $4.8\times10^{-2}$ |
   | | $\ge 10^{-2}$ | 1 | 5–6 points | $2\times10^{2}$ | variance-dominated |

   With noise, the monomial designs move out from the origin as $\tau$ grows and reach 4 symmetric
   atoms at about $\pm0.45$ and $\pm1$, close to the classical cubic-regression designs (the
   D-optimal design is $\pm1, \pm1/\sqrt5$). At moderate $\tau$ bias and variance are of the same
   size, as expected where the trade-off is active. The bumps problem is asymmetric (the modeled
   bumps are centered on the left half), and its support sits on the left.
4. **Local optima remain at small $\tau$.** Noise removed them for the monomial problems, but the
   bumps problem still had 6 distinct optima at $\tau = 10^{-6}$ and 4 at $10^{-4}$. Several
   starts are necessary.
5. **Convergence is slow when an atom falls between grid points.** At $\tau > 0$ most runs did
   not meet the KKT tolerance within 20,000 iterations, although all starts agreed on the cost to
   $10^{-4}$. The optimal atoms sit between grid nodes (e.g. at 0.16–0.18), and splitting weight
   between neighbors is a nearly flat direction. Refining the grid near the support, or handing
   the atoms to the step-4 particle flow, addresses this better than a stricter tolerance.

## Experiments

**1D benchmark.** Legendre (orthogonal; known answer) and monomial or Gaussian-bump bases
(non-orthogonal; unknown answer), prior variance decaying with degree. For orthogonal bases with
uniform $\rho$, $G_{\mathcal{MU}} = 0$ and the continuum problem is trivial, so non-orthogonal
bases carry steps 1–2. Run steps 1–4 for $n$ below, at, and above $p$; compare final points with
Gauss nodes and random points. Expect the biggest gain at $n \approx p$.

**Fourier fit (signal processing).** The most familiar picture of aliasing, used to check that
the GAD reproduces it and to build intuition before the physical examples. Domain: the circle
$t \in [0, 2\pi)$. Basis: the real Fourier modes $1, \cos kt, \sin kt$; modeled frequencies
$k \le K$ ($p = 2K + 1$), unmodeled $K < k \le K_{\max}$, with prior variance decaying like
$k^{-2s}$ ($s$ sets how smooth the true signal is). Prediction points: a dense uniform grid on
the circle.

- *Textbook aliasing.* With $n$ equally spaced samples $t_i = 2\pi i/n$, $A$ is a 0/$\pm1$
  matrix: with $r = m \bmod n$, frequency $m$ lands exactly on frequency $r$, or on $n - r$ if
  $r > n/2$, with a sign flip on the sine in that folded case ($m \equiv -k$). The first unmodeled frequency to alias
  is $m = n - K$: the Nyquist condition, read off $A$. Show $A$ as a heatmap. Checked: $K = 3$,
  $n = 9$ gives exactly this pattern, and frequencies 4 and 5 do not alias.
- *Nyquist sweep.* Cost vs. $n$ for equally spaced points, from $n < p$ (the $I - B$ term)
  through $n = p$ to $n \gg p$. Aliasing falls as the first aliased frequency $n - K$ moves
  into the tail of the prior; the rate depends on $s$.
- *Known answers.* Uniform $\rho$ makes the modes orthogonal, so $G_{\mathcal{MU}} = 0$ and
  $A[\rho] = 0$: the continuum optimum is uniform, and the Fisher–Rao flow from a nonuniform start
  should reach it. At finite $n$, equally spaced points are the circle's analog of Gauss nodes
  (they integrate trigonometric polynomials of degree $< n$ exactly), so the Wasserstein flow from
  random points should approach an equally spaced set. Every rotation is also optimal, so expect a
  flat direction, not a unique minimizer.
- *Coherent vs. incoherent aliasing.* At fixed $n$, compare equally spaced, jittered (one point
  per cell), random and optimized points. Equal spacing puts aliasing into a few exact entries of
  $A$; irregular sampling spreads it across all entries, like broadband noise. First check
  ($K = 3$, $n = 9$, prior $1/k^2$): Frobenius 0.19 / 0.60 / 0.72 and spectral 0.035 / 0.33 / 0.35
  for equally spaced / jittered / random.
- *Ties for the norm survey.* $\cos kt$ and $\sin kt$ alias in matched pairs, so singular values
  of $X$ come in exactly tied pairs (the $K = 3$, $n = 9$ check has $\sigma_1 = \sigma_2$ from
  the $\cos 3t$ and $\sin 3t$ rows). This is the natural place to see the spectral norm's kinks.
- *Where it stops being trivial.* Predict only on part of the circle (e.g. $t \in [0, \pi/2]$),
  or use a prior that is not rotation-invariant. Then uniform is no longer optimal, and the
  optimized design is a real result.

Code: a `fourier_basis(kmax)` helper (closures, like `legendre_basis`). Nothing else is new.

**Equation of state (continuous, physical).** Which volumes should DFT be run at to fit an
energy–volume curve? The third-order Birch–Murnaghan equation of state is a cubic polynomial in
$x = V^{-2/3}$, so it is linear in the basis $\phi_k(V) = V^{-2k/3}$: modeled $k = 0,\dots,3$
($p = 4$), unmodeled $k = 4, 5, \dots$ (higher-order terms), with prior variance decaying in $k$.
The domain is a continuous volume range, e.g. $V \in [0.85\,V_0,\ 1.15\,V_0]$, which is also the
prediction region. The basis is not orthogonal, so the continuum problem is non-trivial and all of
steps 1–4 run unchanged. Truth: a smooth binding curve outside the model's span (a Morse or Rose
universal binding curve for a real metal, or a high-order fit to existing DFT $E$–$V$ data).
Compare optimized volumes with the usual evenly spaced ones, on the cost and on the derived
quantities $V_0$, $B_0$, $B_0'$.

**Cluster expansion.** Candidate pool: the 2,346 binary-alloy configurations (up to 10 sites).
Modeled clusters: the first $p$ in the natural ordering; $\Sigma_{\mathcal U}$ shrinks with
cluster size and diameter. First check which structures the DFT data cover, then split them
once, before anything else: a *calibration* part sets the synthetic truth (effective cluster
interactions fit to it, plus noise) and $\Sigma_{\mathcal U}$; a *validation* part is used only
for the final held-out risk on real DFT energies. If the validation energies also shaped the
prior, their labels would leak into the design through it. Run the capped Fisher–Rao flow over
the pool, round to $n$ structures, polish by exchange (see *Discrete domains*), and compare against
random training sets: $\lVert A \rVert$, held-out risk, and risk vs. $p$ (as in Fig. 6 of the
GAD paper).

## Course analyses

- **Identifiability.** The null space of $M_{\mathcal{TM}}^\top M_{\mathcal{TM}}$ is what $I-B$
  projects onto. With unmodeled terms the data identify only $\theta_{\mathcal M} + A\theta_{\mathcal U}$.
  Report the Fisher-information spectrum and $\lVert A\rVert$, random vs. optimized.
- **Uncertainty quantification.** For $n \ge p$, zero-mean noise independent of $\theta_{\mathcal U}$:

  $$
  \mathbb E\big[(\hat y(x) - y(x))^2\big] = e_{\mathcal U}(x)^\top \Sigma_{\mathcal U}\, e_{\mathcal U}(x)
  + \sigma^2 \phi_{\mathcal M}(x)^\top \big(M_{\mathcal{TM}}^\top M_{\mathcal{TM}}\big)^{-1} \phi_{\mathcal M}(x) \tag{30}
  $$

  This formula needs only the mean and covariance. Check calibration with the mean squared
  standardized error on held-out structures, which should be about 1 under any distribution;
  the stronger check "about 68% of errors within $\pm1\sigma$" holds only if the prediction
  errors are Gaussian. Also check sensitivity to scaling $\Sigma_{\mathcal U}$ by 0.5× and 2×.
- **Dynamics.** Flow fixed points are the KKT designs. Compare linearized rates with measured ones
  (toy: $2\sigma^2\operatorname{Var}_\rho(t)$ vs. $2\sigma^2$). The finite-$n$ cost is non-convex:
  count distinct optima over random starts.
- **Abstraction.** How well the continuum optimum predicts the finite-$n$ optimum as $n$ grows, and
  where it breaks ($n \approx p$); model hierarchy toy → 1D → cluster expansion.

## Package map

All functions below are implemented and tested (245 tests; `julia run_tests.jl`). Experiment
scripts are in [`experiments/`](../experiments/README.md).

| Step | Function | Status |
| ---- | -------- | ------ |
| — | `DesignProblem`, `design_matrix`, `design_matrices`; bases `legendre_basis`, `monomial_basis`, `gaussian_bumps`, `fourier_basis`, `eos_basis` | done |
| — | `aliasing_operator`, `data_insufficiency_operator`, `error_operator` (finite and weighted) | done |
| — | `cost(c, prob, points[, weights])`, `cost_op`, with `Aliasing(norm)`, `Risk(norm)`; noise term `WithNoise(c, τ)` (9a) | done |
| 1 | `first_variation(c, prob, points, weights)`, including (9b) | done (all norms) |
| 2 | `optimize_weights` (Fisher–Rao), with `cap = 1/n` for pools (28); `kkt_multiplier`, `kkt_residual` | done |
| 3 | `sample_points` (quantiles, (25)); `select_points` (pool rounding) | done |
| 4 | `cost_gradient` (exact finite-$n$, including $I - B$); `optimize_points` (particle flow, (26)); `exchange` (pool) | done |


**Implementation notes.**

- *Prediction matrix, by reduced QR when it is tall.* Every risk quantity (the cost, its first
  variation and gradient, the noise term) is a norm of $M_{\mathcal P}Y$ or an inner product
  $M_{\mathcal P}^\top M_{\mathcal P}$. With the reduced QR factorization
  $M_{\mathcal P} = QR_{\mathcal P}$, $Q$ has orthonormal columns, so
  $\lVert M_{\mathcal P}Y\rVert = \lVert R_{\mathcal P}Y\rVert$ in both the Frobenius and the
  operator norm, and $M_{\mathcal P}^\top M_{\mathcal P} = R_{\mathcal P}^\top R_{\mathcal P}$:
  $R_{\mathcal P}$ replaces $M_{\mathcal P}$ exactly. It is applied only when it helps, when
  $M_{\mathcal P}$ is tall ($n_{\mathcal P} > p + q$): $R_{\mathcal P}$ then has $p + q$ rows instead
  of $n_{\mathcal P}$ (2,346 → 300 for the cluster-expansion pool with 300 clusters, 50 → 7 for the 1D
  benchmark). When $n_{\mathcal P} \le p + q$, $M_{\mathcal P}$ is used as is. `DesignProblem`
  factorizes once, keeps only $R_{\mathcal P}$ (never $Q$) for the whole basis, and splits it into
  modeled and unmodeled columns by views, so `with_modeled(prob, p)` changes $p$ for free. Divide by
  $n_{\mathcal P}$, not by the row count of $R_{\mathcal P}$.
- *Weighted fits by QR.* $A[\rho]$ comes from a QR factorization of $\sqrt W M_{\mathcal{TM}}$
  (`weighted_fit`), never from forming $G_{\mathcal{MM}}$, which would square the condition number.
  `optimize_weights` builds the design matrices once per run and reuses each accepted step's fit
  for the next first variation; `WithNoise` shares one fit between its two terms.
- *Risk operator by blocks.* $M_{\mathcal P}E_\theta$ is computed as
  $[M_{\mathcal{PM}}(I - B)\ \ M_{\mathcal{PU}} - M_{\mathcal{PM}}A]$, without forming the
  $(p+q)\times(p+q)$ matrix $E_\theta$.

## Timeline

Weeks count from 2026-10-27; the class deadline is not yet set.

| Week | Goal                                                                                                                                    |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | Operators and costs; first variation and finite-$n$ gradient pass finite-difference tests                                             |
| 2    | Step 2: Fisher–Rao flow; optimal $\rho$ for monomial and Gaussian-bump bases; KKT check                                              |
| 3    | Steps 3–4: quantiles and particle flow; compare with Gauss points; behavior at $n \approx p$                                         |
| 4    | Equation-of-state experiment; Wasserstein vs. Fisher–Rao comparison; course analyses on 1D problems (**minimum viable project**) |
| 5    | Cluster expansion: capped pmf flow, rounding, exchange; risk curves vs. random; DFT validation                                          |
| 6    | Write-up, figures, buffer                                                                                                               |

If time runs short, cut week 4's comparison to one example rather than dropping the cluster expansion.

## Risks

| Risk                                                | Fallback                                                                            |
| --------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Pseudoinverse gradients blow up near rank changes   | Small ridge term; cap step size                                                     |
| Operator-norm cost non-smooth                       | Subgradient or smoothing; Frobenius version as fallback                             |
| Optimal design depends strongly on $\Sigma$       | Report designs for several priors as a result                                       |
| Discretizing weights loses the gain                 | Compare top-$n$, sampling, weighted least squares                                 |
| CE correlation matrix slow to build                 | Use the group's existing code or a smaller lattice                                  |
| Design specializes to the prediction set            | Representative $M_{\mathcal P}$; evaluate on a second, independent prediction set |
| Validation DFT labels leak in through the prior     | Split DFT data once: calibration (prior, synthetic truth) vs. validation only       |
| Capped pmf flow ($w_t \le 1/n$) converges poorly  | Drop the cap and rely on rounding + exchange                                        |
| Equation-of-state truth too close to the model span | Widen the volume range or use a stiffer binding curve                               |
| Noiseless cost degenerate (collapse; trivial risk)  | Noise term $S_\tau$ (9a) with a range of $\tau$; check $\operatorname{cond}(G_{\mathcal{MM}})$ of every reported design |
| Many local optima (worst at small $\tau$)           | Several starts per run; report the spread of final costs                            |
| Optimal atoms fall between grid points (slow, split weights) | Refine the grid near the support, or hand the points to the step-4 particle flow |


## Relation to the GAD paper

*Pending a full check: Sections II.2–II.3.5 of the paper (arXiv:2408.08294v4) were read verbatim;
Sections III–IV and the bibliography only in summary. Read those, and discuss with Gus, before
claiming novelty in the write-up.*

**What the paper does.** It measures aliasing by the spectral norm $\lVert A\rVert_2$ (Section II.3:
"the induced norm is often called the spectral norm") and relates it to error through
deterministic bounds ($R_\theta \le \lVert M\rVert^2\lVert E_\theta\rVert^2\lVert\theta\rVert^2$;
$\lVert A\rVert \le \lVert M^+_{\mathcal{TM}}\rVert\,\lVert M_{\mathcal{TU}}\rVert$). It deliberately avoids θ: aliasing
"can be analyzed without knowing θ or the labels y". Its only expectation over θ (Eqs. 29–30) is for
the insufficiency terms, under an isotropic prior (θ i.i.d., variance $\sigma^2$); A is never averaged
over θ and never weighted by a prior.

**What this project adds.** The expected aliasing (9′), $\mathbb E\lVert A\theta_{\mathcal U}\rVert^2 =
\lVert A\Sigma_{\mathcal U}^{1/2}\rVert_F^2$: the exact expectation of the paper's parameter aliasing under a
prior on the unmodeled coefficients. With the paper's own isotropic assumption it is
$\sigma^2\lVert A\rVert_F^2$; a non-isotropic $\Sigma_{\mathcal U}$ lets physical knowledge in (larger clusters have
smaller ECIs). Unlike $\lVert A\rVert_2$, which is a worst-case gain and needs $\lVert\theta_{\mathcal U}\rVert$ before
it bounds anything, it predicts the error itself, with a known spread
($\operatorname{Var}\lVert A\theta_{\mathcal U}\rVert^2 = 2\lVert A\Sigma_{\mathcal U}A^\top\rVert_F^2$ for Gaussian $\theta_{\mathcal U}$)
and probabilistic bounds (Markov; Hanson–Wright). It is an expectation, not a bound.

**Prior work to cite.** The paper's A is the classical *alias matrix* of experimental design (Box &
Draper, 1959, "A basis for the selection of a response surface design", *J. Amer. Statist. Assoc.*),
and prior-weighted bias from omitted terms appears in Bayesian design with "potential terms"
(DuMouchel & Jones, 1994, *Technometrics*). The defensible claim is bringing prior-weighted
expected aliasing, and label-free optimal design, into the GAD framework, not the idea itself.
Citations from memory; check details before citing.

**Optimal experimental design without labels.** The design step uses only the candidates' design
matrix (structures, not energies), the modeling choice p, the prior Σ (coefficient *sizes*, not
values), the prediction inputs and the noise level. It never uses labels or true parameters, so rows
are chosen before any DFT is run. Truth enters only in evaluating designs (held-out risk and
parameter aliasing on synthetic data). The prior is the one assumption: from geometry alone
(cluster expansion), from a different system (equation of state: Al for Cu), or from a calibration
split of existing data (kept apart from validation).

**Whether it matters is empirical.** The cluster-expansion experiments test it: across training
sets, does expected aliasing track the measured parameter aliasing and held-out risk better than
$\lVert A\rVert_2$? The decisive case is a truth the prior did not anticipate (`TRUTH=unit`).

## Future improvements

- **Solving the right problem with the right algorithm.** The *problem* is fixed by the GAD: the
  estimator is the minimum-norm least-squares fit $\hat\theta = M_{\mathcal{TM}}^+ y$, and $B$,
  $I - B$ and the pseudoinverse derivative in `cost_gradient` are defined from it. The *algorithm*
  is a separate choice: how to compute that same $\hat\theta$ accurately. Noted 2026-11-19, after
  the cluster-expansion run on the JuCE pool, where at $n = p$ the evaluation's fit with `\` gave a
  held-out RMSE of $2\times10^{15}$ while the costs stayed finite. Checked in Julia 1.13:

  | Shape of $M$ | What `\` uses | Result |
  |---|---|---|
  | wide ($n < p$) | pivoted QR + rank estimate | $M^+ y$ (agrees with `pinv` to $10^{-16}$) |
  | tall, even rank-deficient | pivoted QR + rank estimate | $M^+ y$ (agrees to $10^{-16}$) |
  | **square** ($n = p$) | **LU, no rank estimate** | blows up near singularity ($\lVert\hat\theta\rVert = 2\times10^{12}$ vs. 0.42) |

  So `\` solves the GAD's problem except for square systems. Pivoted QR,
  `qr(M, ColumnNorm()) \ y`, gives $M^+ y$ for every shape (0.42 in the square case) without a full
  SVD. The evaluation now uses `pinv` to match the costs. Possible changes to the package:
  - Replace the explicit `pinv(M_TM)` in `aliasing_operator`, `data_insufficiency_operator`,
    `error_operator`, the risk `cost_op`, `noise_variance` and `cost_gradient` by one pivoted-QR
    factorization, reused for $A$, $B$ and the solve. Same quantities, better accuracy, and a
    rank estimate that can be reported.
  - Never use plain `\` on a square $M_{\mathcal{TM}}$ (it switches to LU).
  - Near-singular designs remain ill-conditioned whatever the algorithm: the rank tolerance of
    `pinv` or pivoted QR is an implicit regularization that the costs do not model. A principled
    treatment is explicit regularization (ridge, or a truncation threshold tied to the noise level),
    which changes the estimator and so has to enter the costs too.
  - Report $\operatorname{cond}(M_{\mathcal{TM}})$ and the estimated rank for every design.

## Reading

Citations from memory; check details before citing.

- **GAD:** Transtrum, Hart, Jarvis, Whitehead, *Aliasing and Label-Independent Decomposition of Risk*, [arXiv:2408.08294](https://arxiv.org/abs/2408.08294)
- **Optimal design:** Kiefer & Wolfowitz (1960), "The equivalence of two extremum problems," *Canadian J. Math.*; Pukelsheim, *Optimal Design of Experiments*
- **Rounding relaxed designs:** Allen-Zhu, Li, Singh & Wang, "Near-optimal discrete optimization for experimental design: a regret minimization approach," *Math. Program.* (arXiv 2017); guarantees for convex criteria
- **Limits of designs:** Niederreiter, *Random Number Generation and Quasi-Monte Carlo Methods* (Koksma–Hlawka, discrepancy); Braides, *Γ-convergence for Beginners* (convergence of minimizers)
- **Calculus of variations:** Gelfand & Fomin, *Calculus of Variations*, ch. 1–2
- **Wasserstein flows:** Santambrogio, *Optimal Transport for Applied Mathematicians*, ch. 8, or "{Euclidean, metric, and Wasserstein} gradient flows: an overview" (2017); Chizat & Bach (2018), NeurIPS
- **Fisher–Rao:** Amari, *Information Geometry and Its Applications*
- **Model-manifold geometry:** Transtrum, Machta & Sethna (2010 PRL; 2011 PRE)

## Survey of norms

The cost is $S = \lVert X \rVert^2$ for an operator $X$ ($X_{\text{alias}}$ or $X_{\text{risk}}$,
eq. 8), and the plan so far uses two norms: Frobenius (smooth) and spectral (not). This
section compares a wider family. The goal is to show *where* a norm makes the optimization
hard, why, and which smooth substitutes fix it, not to rank the norms: they answer different
questions (average vs. worst case), so each optimum is also scored under every other norm.

### One formula for every norm

Steps 1–2 need only $K = \tfrac12\,\partial S/\partial X$. The first variation (15) then holds
unchanged with that $K$. Write the SVD $X = U\,\operatorname{diag}(\sigma)\,V^\top$ with
$\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r$. For a unitarily invariant cost,
$S = f(\sigma)$, and wherever $f$ is differentiable

$$
K = U\,\operatorname{diag}\!\Big(\tfrac12\,\frac{\partial f}{\partial \sigma}\Big)\,V^\top \tag{31}
$$

(Lewis 1995). So one SVD (or a truncated one) plus a formula per norm covers the whole family.
Where $f$ has a kink (tied singular values, or a singular value at zero) there is no single
$K$, only a set of them: that is the source of the problems below.

### Candidate norms

| #   | Cost $S$                                                               | Meaning                                                                                        | $K = \tfrac12\,\partial S/\partial X$                                                           | Smooth?                                             |
| --- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| N1  | Frobenius $\lVert X\rVert_F^2 = \sum_k \sigma_k^2$                     | Expected squared error under the prior (A-optimality)                                          | $X$                                                                                             | Yes                                                 |
| N2  | Spectral $\sigma_1^2$                                                  | Worst case over the prior ellipsoid (E-optimality)                                             | $\sigma_1 u_1 v_1^\top$                                                                         | No: where $\sigma_1 = \sigma_2$                   |
| N3  | Schatten-$p$, $\big(\sum_k \sigma_k^p\big)^{2/p}$                    | Interpolates N1 ($p = 2$) and N2 ($p \to \infty$); Kiefer's $\phi_p$ family              | $U \operatorname{diag}\!\big(\lVert\sigma\rVert_p^{2-p}\sigma^{p-1}\big) V^\top$                | Yes for $2 \le p < \infty$; stiff for large $p$ |
| N4  | Soft maximum $\tfrac1\beta \log \sum_k e^{\beta\sigma_k^2}$            | Smoothed N2: $\sigma_1^2 \le S_\beta \le \sigma_1^2 + \tfrac{\log r}{\beta}$                 | $U \operatorname{diag}(\pi_k \sigma_k) V^\top$, $\pi = \operatorname{softmax}(\beta\sigma^2)$ | Yes                                                 |
| N5  | Ky Fan $k$: $\sum_{i \le k} \sigma_i^2$                              | Total error in the $k$ worst directions                                                      | $U_k \operatorname{diag}(\sigma_{1:k}) V_k^\top$                                                | No: where $\sigma_k = \sigma_{k+1}$               |
| N6  | Nuclear $\big(\sum_k \sigma_k\big)^2$                                  | Rewards low-rank error; convex surrogate for rank                                              | $\lVert X\rVert_*\, U_r V_r^\top$                                                               | No: where a $\sigma_k$ reaches 0                  |
| N7  | Log-det $\log\det(I + X X^\top)$ (not a norm)                          | Volume of the error ellipsoid; saturates for large $\sigma$ (D-optimality flavor)            | $(I + X X^\top)^{-1} X$                                                                         | Yes                                                 |
| N8  | Max row $\max_j \lVert X_{j,:}\rVert^2$                                | Risk: worst prediction point (G-optimality). Aliasing: worst-contaminated coefficient          | $e_j X_{j,:}$ for the maximizing row $j$                                                      | No: where rows tie                                  |
| N9  | Soft max row $\tfrac1\beta\log\sum_j e^{\beta \lVert X_{j,:}\rVert^2}$ | Smoothed N8                                                                                    | $\operatorname{diag}(\pi)X$, $\pi = \operatorname{softmax}$ over rows                         | Yes                                                 |
| N10 | Induced $\infty$-norm $\max_j \sum_l \lvert X_{jl}\rvert$, squared   | Worst case over a *box* of coefficients ($\lvert z_l\rvert \le 1$) instead of an ellipsoid | $\lVert X\rVert_\infty\, e_j \operatorname{sign}(X_{j,:})$ for the maximizing row $j$         | No: polyhedral, kinks everywhere                    |

N1–N7 depend only on singular values, so they use (31). N8–N10 depend on coordinates: they are
meaningful when rows have a meaning (prediction points for risk, coefficients for aliasing), and
their kinks are ties between rows instead of between singular values.

**Control case.** When $X$ has rank one ($p = 1$ or $q = 1$: the toy problem, or
`legendre_basis(p + 1)`) every unitarily invariant norm gives the same $S$ and the same
optimum. Use it as a sanity check, and use problems with $\min(p, q) \ge 2$ for everything else.

### Experiment steps

1. **Landscape slice.** Pick a one-parameter family of designs (e.g. symmetric weights moving
   outward). Plot $\sigma_1^2, \sigma_2^2, \dots$ and every cost along it. Expect kinks in N2,
   N5, N8, N10 where the relevant quantities cross; N1, N3, N4, N7, N9 smooth.
2. **Same flow, every norm.** Run the Fisher–Rao flow from the same start for each norm. Record
   per iteration: the cost, the relative gap $(\sigma_1 - \sigma_2)/\sigma_1$ (or the row gap
   for N8–N10), and the cosine between successive first variations. Zigzag shows up as the gap
   going to 0 while the cosine turns negative and the cost plateaus.
3. **Optimality certificate.** At each final $\rho$, plot $\delta S/\delta\rho$ over the domain
   against the KKT condition (16). For a nonsmooth norm at a tie, (16) holds only for some convex
   combination of the tied directions' first variations; report whether one exists (a small
   nonnegative least-squares problem).
4. **Gradient reliability.** At random designs, measure the finite-difference error of
   `first_variation` against the gap. Expect the error to grow like $1/\text{gap}$ for the
   nonsmooth norms, and to stay flat for the smooth ones.
5. **Smoothing and continuation.** Rerun step 2 with N4 (increasing $\beta$), N3 (increasing $p$)
   and N9, each warm-started from the Frobenius optimum. Measure how close each gets to the
   N2 / N8 cost, and in how many iterations.
6. **Cross-evaluation.** Score every norm's optimum under every norm: an $N \times N$ table.
   Shows the trade-offs (e.g. how much average error the worst-case optimum gives up).
7. **Structure of the optimum.** For each norm, count the support of the optimal $\rho$ (number
   of atoms) and compare with the Gauss nodes and with the Frobenius optimum. Kiefer–Wolfowitz
   suggests few atoms; nonsmooth norms may land on ties with more.

**Problems.** (a) the toy problem as the rank-one control; (b) `monomial_basis(7)`, $p = 4$,
prior $1/k^2$ ($q = 3$; already in the tests); (c) Legendre with $q \ge 2$, e.g.
`legendre_basis(p + 3)` with a decaying prior; (d) the equation-of-state basis; (e) the Fourier
fit with equally spaced points, where $\cos$/$\sin$ pairs give exactly tied singular values.
Run each for
both $X_{\text{alias}}$ and $X_{\text{risk}}$.

**Hypotheses** (to test, not results). The nonsmooth norms (N2, N5, N8, N10) stall at ties under
plain gradient flow, fail (16) as stated, and give unreliable finite differences near ties.
Frobenius and the smooth substitutes do not. Continuation in $\beta$ or $p$ recovers nearly the
worst-case optimum without the stall. A clean outcome either way is a result: if the spectral
flow converges without trouble on some problem, its optimum is not at a tie, and the experiment
says when the issue matters.

### What the code needs

- `get_K(norm, X)` methods for each norm, using (31) for N1–N7. Only N1, N2 exist now.
- A helper for the top $k$ singular triplets (full SVD when small, `tsvd(X, k)` when large),
  shared by N2–N5 and the gap diagnostics.
- `cost` via `sqnorm(X, norm)` for each norm.
- Diagnostics: relative gap, cosine between successive first variations, KKT residual.
- The Fisher–Rao loop from step 2 (planned), with a step-size option: fixed, and diminishing
  ($\propto 1/\sqrt{k}$) for the nonsmooth norms.

### Reading for this section

Citations from memory; check details before citing.

- Lewis (1995), "The convex analysis of unitarily invariant matrix functions," *J. Convex Anal.*: gradients and subgradients of spectral functions, eq. (31)
- Overton (1988), "On minimizing the maximum eigenvalue of a symmetric matrix," *SIAM J. Matrix Anal. Appl.*; Overton & Womersley (1993): why max-eigenvalue optima sit at ties
- Nesterov (2005), "Smooth minimization of non-smooth functions," *Math. Program.*: soft-max smoothing with a known accuracy bound
- Pukelsheim, *Optimal Design of Experiments*, ch. 6: Kiefer's $\phi_p$ criteria (A, D, E as special cases)
