# Calculus of Variations Primer

A step-by-step companion to the calculus-of-variations parts of the
[project plan](project_plan.md): the first variation (12)–(15), the optimality condition (16),
and the Fisher–Rao gradient and descent rate (20), (22). Equation numbers like "plan (15)" refer
to the plan. Practice problems with worked solutions are at the end.

The whole subject reduces to one idea: **a functional derivative is an ordinary gradient whose
index is a point $t$ instead of an integer $i$.** Everything else is the chain rule, plus a few
matrix identities that say how to differentiate through a trace, an inverse and a norm.

---

## 1. From gradients to functional derivatives

### 1.1 Finite dimensions first

Let $S(w)$ depend on weights $w = (w_1, \dots, w_N)$. Nudge the weights in a direction $\eta$
and look at the first-order change:

$$
\frac{d}{d\varepsilon} S(w + \varepsilon\eta)\Big|_{\varepsilon=0} = \sum_{i=1}^{N} \frac{\partial S}{\partial w_i}\,\eta_i \tag{1.1}
$$

The gradient is *whatever multiplies $\eta_i$* in this sum. You can find it without computing
partial derivatives one at a time: expand $S(w + \varepsilon\eta)$, keep the term linear in
$\varepsilon$, and rearrange it into the form $\sum_i (\text{something})_i\,\eta_i$.

### 1.2 Infinitely many weights

Now the "weights" are the values of a density $\rho(t)$, one for every point $t$. The sum over
$i$ becomes an integral over $t$, and the same recipe defines the functional derivative
(plan (12)):

$$
\frac{d}{d\varepsilon} S[\rho + \varepsilon\eta]\Big|_{\varepsilon=0} = \int \frac{\delta S}{\delta\rho}(t)\,\eta(t)\,dt \tag{1.2}
$$

$\frac{\delta S}{\delta\rho}(t)$ plays the role of $\frac{\partial S}{\partial w_i}$: it is the
sensitivity of $S$ to adding a little weight at the point $t$.

**On a grid they are the same number.** If $w_i = \rho(t_i)\,\Delta t$ and $\eta_i = \eta(t_i)\,\Delta t$,
then (1.1) is a Riemann sum for (1.2), and $\frac{\delta S}{\delta\rho}(t_i) = \frac{\partial S}{\partial w_i}$.
This is why `first_variation` returns one number per grid point.

### 1.3 Notation used below

Write $\delta F$ for the directional derivative of any quantity $F$ (scalar, vector or matrix):

$$
\delta F = \frac{d}{d\varepsilon} F[\rho + \varepsilon\eta]\Big|_{\varepsilon=0}
$$

So (1.2) reads $\delta S = \int \frac{\delta S}{\delta\rho}(t)\,\eta(t)\,dt$. (The plan writes the
same thing with an explicit factor $\varepsilon$, as $\delta A = \varepsilon\int\cdots$ in
plan (13). Same content; here the $\varepsilon$ is already divided out.)

$\delta$ obeys the usual rules of differentiation: it is linear, it satisfies the product rule,
and it satisfies the chain rule.

### 1.4 The recipe

1. **Name the averages.** Write $S$ in terms of the quantities that depend on $\rho$, giving each
   integral over $\rho$ its own name, e.g. $m_f = \int f\rho\,dt$. Each such integral is a
   *number*, not a function of $t$. Never combine two of them into one integral (see the warning
   below).
2. **Perturb each one.** Replace $\rho$ by $\rho + \varepsilon\eta$ inside each integral:
   $m_f \to m_f + \varepsilon\int f\eta\,dt$. For a matrix of averages, do this entry by entry
   ($G \to G + \varepsilon\,\delta G$).
3. **Keep the first order.** Expand $S$ and keep the terms linear in $\varepsilon$, i.e. apply the
   product and chain rules to the named averages. Divide by $\varepsilon$; the result is $\delta S$.
4. **Collect under one integral.** Every term of $\delta S$ contains exactly one $\int(\cdot)\,\eta\,dt$
   with numbers in front. Move the numbers inside and combine into
   $\int(\text{something})(t)\,\eta(t)\,dt$.
5. **Read off:** the something is $\frac{\delta S}{\delta\rho}(t)$. It never contains $\eta$, and it
   contains $\rho$ only through named averages, never as a bare $\rho(t)$ (unless $S$ itself
   had a bare $\rho(t)$ inside an integral, as in Problem 3).

Step 4 is where the work is. In the matrix problems below, step 4 is exactly the question
"where does the trace go".

> **Warning: a product of integrals is not the integral of a product.**
> $\big(\int f\rho\,dt\big)\big(\int h\rho\,dt\big) \ne \int f h\,\rho^2\,dt$.
> On the left, the two integrals run over *separate* copies of the variable. Written out with
> separate dummy variables, it is the double integral $\iint f(s)\,h(u)\,\rho(s)\,\rho(u)\,ds\,du$.
> Using one letter $t$ for both dummies and merging them into one integral changes the
> functional, and gives the wrong answer $2fh\rho$ for Problem 1.
> Naming the averages in step 1 prevents this: $m_f\,m_h$ is clearly a product of two numbers.

**Worked example (Problem 1, $S = m_f\,m_h$).**

1. Averages: $m_f = \int f\rho\,dt$ and $m_h = \int h\rho\,dt$, two numbers.
2. Perturb: $m_f \to m_f + \varepsilon\int f\eta\,dt$, $m_h \to m_h + \varepsilon\int h\eta\,dt$.
3. Expand: $S \to m_f m_h + \varepsilon\big(m_h\int f\eta\,dt + m_f\int h\eta\,dt\big) + \varepsilon^2(\cdots)$,
   so $\delta S = m_h\int f\eta\,dt + m_f\int h\eta\,dt$.
4. Collect: $\delta S = \int\big(m_h f(t) + m_f h(t)\big)\,\eta(t)\,dt$.
5. Read off: $\frac{\delta S}{\delta\rho}(t) = m_h f(t) + m_f h(t)$.

Check with the double-integral form: varying $\rho(s)\rho(u)$ gives $\eta(s)\rho(u) + \rho(s)\eta(u)$.
The first term integrates over $u$ to $m_h\int f\eta$, and the second over $s$ to $m_f\int h\eta$:
the same answer.

### 1.5 Three warm-ups

**Linear functional.** $S[\rho] = \int f(t)\,\rho(t)\,dt$ for a fixed function $f$.
Then $S[\rho + \varepsilon\eta] = S[\rho] + \varepsilon\int f\,\eta\,dt$, already in the form of
step 4, so

$$
\frac{\delta S}{\delta\rho}(t) = f(t)
$$

*Compare with $S(w) = \sum_i f_i w_i$, whose gradient is $f$.*

**Square of a linear functional.** $S = \big(\int f\rho\,dt\big)^2$. Let $m = \int f\rho\,dt$.
By the chain rule $\delta S = 2m\,\delta m = 2m\int f\eta\,dt = \int 2m f(t)\,\eta(t)\,dt$, so

$$
\frac{\delta S}{\delta\rho}(t) = 2\,m\,f(t) = 2\Big(\int f\rho\,ds\Big) f(t)
$$

Note that $m$ is a number (it already integrates over $\rho$), so it rides along as a constant.

**The toy aliasing cost.** Fit a constant ($\phi_{\mathcal M} = 1$) when the truth has a slope
($\phi_{\mathcal U} = t$, prior variance $\sigma^2$). Then $A = \bar t = \int t\,\rho\,dt$ and
$S = \sigma^2\bar t^{\,2}$. This is the previous example with $f(t) = t$:

$$
\frac{\delta S}{\delta\rho}(t) = 2\sigma^2\,\bar t\,t
$$

The plan's (29) has $2\sigma^2\bar t\,(t - \bar t)$ instead. The two differ by the constant
$-2\sigma^2\bar t^{\,2}$, and §2 shows why constants do not matter.

---

## 2. Constraints: densities, additive constants, and the KKT condition

### 2.1 Why the first variation is only defined up to a constant

$\rho$ must stay a density: $\int\rho\,dt = 1$. So the only allowed directions have
$\int\eta\,dt = 0$ (you can move weight around, not create it). If you add a constant $c$ to
$\frac{\delta S}{\delta\rho}$,

$$
\int\Big(\frac{\delta S}{\delta\rho}(t) + c\Big)\eta(t)\,dt = \int \frac{\delta S}{\delta\rho}(t)\,\eta(t)\,dt + c\underbrace{\int\eta\,dt}_{=\,0}
$$

so no allowed direction can tell the difference. That is why the docstring of `first_variation`
says "defined up to an additive constant", and why the tests compare $\delta$ after removing a
constant.

### 2.2 Deriving the KKT condition (plan (16)) without Lagrange multipliers

Write $g(t) = \frac{\delta S}{\delta\rho}(t)$. Suppose $\rho$ is optimal. Take a little weight
$\varepsilon$ from a point $a$ where $\rho(a) > 0$ and put it at any point $b$. This is the
direction $\eta = \delta_b - \delta_a$ (two spikes), which integrates to zero, so it is allowed
for small $\varepsilon > 0$. The change in cost is

$$
\delta S = \int g\,\eta\,dt = g(b) - g(a)
$$

At an optimum, no allowed move can lower the cost, so $g(b) \ge g(a)$ for every $b$.

- If $b$ is also in the support, the move $a \to b$ and the move $b \to a$ are both allowed, so
  $g(b) \ge g(a)$ and $g(a) \ge g(b)$: **$g$ is constant on the support.** Call the constant $\lambda$.
- If $\rho(b) = 0$, only the move into $b$ is allowed, so **$g(b) \ge \lambda$ off the support.**

That is plan (16). In words: at the optimum, every point you sample is equally useful, and every
point you do not sample would make things worse. The Lagrange multiplier $\lambda$ is just the
common value of $g$ on the support.

---

## 3. Matrix calculus toolkit

This section answers **where the trace goes.** Short answer: the trace is a sum, so it is
linear, so $\delta$ passes straight through it, exactly as $\delta$ passes through $\int$ and
$\sum$. The trace does not get differentiated; it just carries the variation along until a
rank-one identity (§3.5) turns it into a plain number.

### 3.1 The trace is the dot product of matrices

For matrices of the same shape,

$$
\langle P, Q\rangle = \operatorname{tr}(P^\top Q) = \sum_{i,j} P_{ij}\,Q_{ij} \tag{3.1}
$$

This is the ordinary dot product of the two matrices flattened into long vectors. Check the
middle equality: $(P^\top Q)_{jj} = \sum_i P_{ij}Q_{ij}$, and the trace sums over $j$.

So whenever you see $\operatorname{tr}(P^\top Q)$, read it as "dot product of $P$ and $Q$".
The squared Frobenius norm is $\lVert X\rVert_F^2 = \operatorname{tr}(X^\top X) = \sum_{ij} X_{ij}^2$.

### 3.2 $\delta$ passes through the trace (and through integrals)

The trace is a sum of diagonal entries, and the variation of a sum is the sum of the variations:

$$
\delta\operatorname{tr}(F) = \delta\sum_j F_{jj} = \sum_j \delta F_{jj} = \operatorname{tr}(\delta F) \tag{3.2}
$$

The same reason (linearity) lets you swap a trace and an integral:

$$
\operatorname{tr}\Big(\int M(t)\,\eta(t)\,dt\Big) = \int \operatorname{tr}\big(M(t)\big)\,\eta(t)\,dt \tag{3.3}
$$

since integrating a matrix means integrating each entry, and $\eta(t)$ is a scalar.

### 3.3 Product rule (order matters)

$$
\delta(PQ) = (\delta P)\,Q + P\,(\delta Q) \tag{3.4}
$$

Same as the scalar product rule, except matrices do not commute, so keep each factor in place.
Check in indices: $\delta\sum_k P_{ik}Q_{kj} = \sum_k (\delta P_{ik}\,Q_{kj} + P_{ik}\,\delta Q_{kj})$.

### 3.4 Variation of an inverse

Vary both sides of $G\,G^{-1} = I$ with the product rule:
$(\delta G)\,G^{-1} + G\,\delta(G^{-1}) = 0$. Multiply on the left by $G^{-1}$:

$$
\delta(G^{-1}) = -G^{-1}\,(\delta G)\,G^{-1} \tag{3.5}
$$

*Scalar check:* $\delta(1/g) = -\delta g/g^2$. The matrix version puts one $G^{-1}$ on each side
because the factors do not commute.

### 3.5 Trace tricks

1. **Transpose:** $\operatorname{tr}(M^\top) = \operatorname{tr}(M)$ (same diagonal).
2. **Cyclic:** $\operatorname{tr}(PQR) = \operatorname{tr}(QRP) = \operatorname{tr}(RPQ)$.
   You may rotate factors, not reorder them arbitrarily.
3. **Rank one, the step where the trace disappears.** For vectors $a$, $b$ and a matrix $K$:

$$
\operatorname{tr}\big(K^\top a\, b^\top\big) = a^\top K\, b \tag{3.6}
$$

*Proof in indices:* $\operatorname{tr}(K^\top a b^\top) = \langle K, a b^\top\rangle = \sum_{ij} K_{ij}\,a_i b_j = a^\top K b$.
*Proof by cycling:* $\operatorname{tr}(K^\top a b^\top) = \operatorname{tr}(b^\top K^\top a) = b^\top K^\top a$
(the trace of a $1\times1$ matrix is the number itself) $= a^\top K b$.

The right side is a single number, a "sandwich" of $K$ between two vectors. Whenever the
variation of a matrix is an outer product $a b^\top$, the trace collapses to this sandwich.
That is what happens in every first variation in the plan, because each sample point adds a
rank-one term $\phi(t)\phi(t)^\top$ to the Gram matrix.

### 3.6 Variation of the Frobenius norm

$$
\delta\operatorname{tr}(X^\top X) = \operatorname{tr}(\delta X^\top X) + \operatorname{tr}(X^\top\delta X) = 2\operatorname{tr}(X^\top \delta X) \tag{3.7}
$$

using (3.2), the product rule, and the transpose rule on the first term
($\operatorname{tr}(\delta X^\top X) = \operatorname{tr}((X^\top\delta X)^\top) = \operatorname{tr}(X^\top\delta X)$).
In indices: $\delta\sum_{ij}X_{ij}^2 = 2\sum_{ij} X_{ij}\,\delta X_{ij}$, the familiar
$d(x^2) = 2x\,dx$ applied entrywise.

### 3.7 Variation of the top singular value

Let $\sigma_1$ be the largest singular value of $X$, with unit singular vectors $u_1$, $v_1$:
$Xv_1 = \sigma_1 u_1$, $X^\top u_1 = \sigma_1 v_1$, so $\sigma_1 = u_1^\top X v_1$. Vary all three
factors:

$$
\delta\sigma_1 = (\delta u_1)^\top X v_1 + u_1^\top (\delta X)\, v_1 + u_1^\top X\,\delta v_1
$$

The first term is $\sigma_1 (\delta u_1)^\top u_1$, and $u_1^\top\delta u_1 = 0$ because varying
$u_1^\top u_1 = 1$ gives $2u_1^\top\delta u_1 = 0$. The third term vanishes the same way. So

$$
\delta\sigma_1 = u_1^\top(\delta X)\,v_1 = \operatorname{tr}\big((u_1 v_1^\top)^\top \delta X\big), \qquad
\delta(\sigma_1^2) = 2\operatorname{tr}\big((\sigma_1 u_1 v_1^\top)^\top \delta X\big) \tag{3.8}
$$

The middle form is (3.6) read backwards. **This needs $\sigma_1 > \sigma_2$**: if the top singular
value is tied, $u_1$ and $v_1$ are not uniquely defined and the derivation breaks (the
nonsmoothness discussed in the plan's norm survey).

### 3.8 The $K$ convention

(3.7) and (3.8) have the same shape: $\delta S = 2\operatorname{tr}(K^\top\delta X)$ with

| Norm | $S$ | $K$ |
|---|---|---|
| Frobenius | $\operatorname{tr}(X^\top X)$ | $X$ |
| Spectral | $\sigma_1^2$ | $\sigma_1 u_1 v_1^\top$ |

So $K$ is half the gradient of $S$ with respect to the matrix $X$. Computing $K$ is the only
norm-specific step; everything after it is shared. This is `get_K` in [src/costs.jl](../src/costs.jl).

---

## 4. Deriving plan (13)–(15), one line at a time

Setup (plan (10), (11), (8)):

$$
G_{\mathcal{MM}} = \int \phi_{\mathcal M}\phi_{\mathcal M}^\top \rho\,dt, \qquad
G_{\mathcal{MU}} = \int \phi_{\mathcal M}\phi_{\mathcal U}^\top \rho\,dt, \qquad
A = G_{\mathcal{MM}}^{-1}G_{\mathcal{MU}}, \qquad
X = A\,\Sigma_{\mathcal U}^{1/2}, \qquad S = \lVert X\rVert^2
$$

The dependence runs $\rho \to G \to A \to X \to S$. Apply the chain rule one link at a time.

### Step A: vary the Gram matrices

They are linear in $\rho$, so (first warm-up of §1.5, entry by entry):

$$
\delta G_{\mathcal{MM}} = \int \phi_{\mathcal M}(t)\,\phi_{\mathcal M}(t)^\top\,\eta(t)\,dt, \qquad
\delta G_{\mathcal{MU}} = \int \phi_{\mathcal M}(t)\,\phi_{\mathcal U}(t)^\top\,\eta(t)\,dt \tag{4.1}
$$

Each is an integral of rank-one matrices weighted by $\eta(t)$. Keep that in mind; it is what
makes the trace collapse in step E.

### Step B: vary $A$

Product rule (3.4), then the inverse rule (3.5):

$$
\delta A = \delta(G_{\mathcal{MM}}^{-1})\,G_{\mathcal{MU}} + G_{\mathcal{MM}}^{-1}\,\delta G_{\mathcal{MU}}
= -G_{\mathcal{MM}}^{-1}\,\delta G_{\mathcal{MM}}\,\underbrace{G_{\mathcal{MM}}^{-1}G_{\mathcal{MU}}}_{A} + G_{\mathcal{MM}}^{-1}\,\delta G_{\mathcal{MU}}
= G_{\mathcal{MM}}^{-1}\big(\delta G_{\mathcal{MU}} - \delta G_{\mathcal{MM}}\,A\big)
$$

Substitute (4.1). Both terms share the integral, so combine them inside it:

$$
\delta A = \int G_{\mathcal{MM}}^{-1}\phi_{\mathcal M}(t)\,\big(\phi_{\mathcal U}(t)^\top - \phi_{\mathcal M}(t)^\top A\big)\,\eta(t)\,dt
= \int G_{\mathcal{MM}}^{-1}\phi_{\mathcal M}(t)\;e_{\mathcal U}(t)^\top\,\eta(t)\,dt \tag{4.2}
$$

with $e_{\mathcal U}(t)^\top = \phi_{\mathcal U}(t)^\top - \phi_{\mathcal M}(t)^\top A$ (plan (14)).
This is plan (13). The integrand is a column vector times a row vector, a rank-one
$p\times q$ matrix for each $t$.

*Meaning of $e_{\mathcal U}$:* $\phi_{\mathcal M}(t)^\top A$ is the modeled fit's best imitation of
the unmodeled features at $t$; $e_{\mathcal U}(t)$ is what it misses. Adding weight at $t$ changes
$A$ in proportion to that miss.

### Step C: vary $X$

$\Sigma_{\mathcal U}^{1/2}$ does not depend on $\rho$, so

$$
\delta X = (\delta A)\,\Sigma_{\mathcal U}^{1/2} \tag{4.3}
$$

### Step D: vary $S$, and carry the trace through

From §3.8, $\delta S = 2\operatorname{tr}(K^\top\delta X)$. Substitute (4.3), then (4.2):

$$
\delta S = 2\operatorname{tr}\Big(K^\top\,\delta A\,\Sigma_{\mathcal U}^{1/2}\Big)
= 2\operatorname{tr}\Big(K^\top \Big[\int G_{\mathcal{MM}}^{-1}\phi_{\mathcal M}\,e_{\mathcal U}^\top\,\eta\,dt\Big]\Sigma_{\mathcal U}^{1/2}\Big)
$$

$K^\top$ and $\Sigma_{\mathcal U}^{1/2}$ do not depend on $t$, so they move inside the integral,
and then the trace moves inside too, by linearity (3.3):

$$
\delta S = \int 2\operatorname{tr}\Big(K^\top\,G_{\mathcal{MM}}^{-1}\phi_{\mathcal M}(t)\;e_{\mathcal U}(t)^\top\,\Sigma_{\mathcal U}^{1/2}\Big)\,\eta(t)\,dt \tag{4.4}
$$

**This is the answer to "where does the trace go":** it has not gone anywhere yet. It went
*inside the integral*, around the matrix that multiplies $\eta(t)$ at each $t$.

### Step E: collapse the trace with the rank-one identity

Inside the trace, group the factors as $K^\top a\,b^\top$ with

$$
a = G_{\mathcal{MM}}^{-1}\phi_{\mathcal M}(t) \in \mathbb R^p, \qquad
b^\top = e_{\mathcal U}(t)^\top\Sigma_{\mathcal U}^{1/2}, \text{ i.e. } b = \Sigma_{\mathcal U}^{1/2}e_{\mathcal U}(t) \in \mathbb R^q
$$

($\Sigma_{\mathcal U}^{1/2}$ is symmetric.) By (3.6), $\operatorname{tr}(K^\top a b^\top) = a^\top K b$,
so

$$
\delta S = \int 2\,\big(G_{\mathcal{MM}}^{-1}\phi_{\mathcal M}(t)\big)^\top K\,\Sigma_{\mathcal U}^{1/2}e_{\mathcal U}(t)\;\eta(t)\,dt
$$

**Now the trace is gone:** it became the sandwich $a^\top K b$, a single number for each $t$.

### Step F: read off

The expression has the form $\int(\text{something})(t)\,\eta(t)\,dt$, so by the recipe (§1.4):

$$
\boxed{\frac{\delta S_{\text{alias}}}{\delta\rho}(t) = 2\,\big(G_{\mathcal{MM}}^{-1}\phi_{\mathcal M}(t)\big)^\top K\,\Sigma_{\mathcal U}^{1/2}\,e_{\mathcal U}(t)}
$$

This is the first half of plan (15).

**Frobenius special case.** $K = X = A\Sigma_{\mathcal U}^{1/2}$, and $G_{\mathcal{MM}}$ is symmetric, so
$a^\top K b = \phi_{\mathcal M}^\top G_{\mathcal{MM}}^{-1} A\,\Sigma_{\mathcal U}^{1/2}\Sigma_{\mathcal U}^{1/2} e_{\mathcal U}
= \phi_{\mathcal M}^\top G_{\mathcal{MM}}^{-1} A\,\Sigma_{\mathcal U}\,e_{\mathcal U}$, the line under plan (15).

**Sanity check on the toy problem.** $\phi_{\mathcal M} = 1$, $\phi_{\mathcal U} = t$,
$G_{\mathcal{MM}} = 1$, $A = \bar t$, $\Sigma_{\mathcal U}^{1/2} = \sigma$, $K = X = \bar t\sigma$,
$e_{\mathcal U}(t) = t - \bar t$:

$$
\frac{\delta S}{\delta\rho}(t) = 2\cdot 1\cdot \bar t\sigma\cdot\sigma\,(t - \bar t) = 2\sigma^2\bar t\,(t - \bar t)
$$

which is plan (29), and agrees with the warm-up in §1.5 up to a constant.

### The risk version

Only the unmodeled block of the risk operator depends on $\rho$ (plan, end of *The cost*):

$$
R = n_{\mathcal P}^{-1/2}\big(M_{\mathcal{PU}} - M_{\mathcal{PM}}A\big)\Sigma_{\mathcal U}^{1/2}
\quad\Longrightarrow\quad
\delta R = -\,n_{\mathcal P}^{-1/2}\,M_{\mathcal{PM}}\,(\delta A)\,\Sigma_{\mathcal U}^{1/2}
$$

($M_{\mathcal{PU}}$ and $M_{\mathcal{PM}}$ are evaluated at the fixed prediction points, so they
do not vary.) Steps D–F are identical, with $K^\top$ replaced by $-n_{\mathcal P}^{-1/2}K^\top M_{\mathcal{PM}}$
in front of $\delta A$. The rank-one identity now uses
$a = M_{\mathcal{PM}}G_{\mathcal{MM}}^{-1}\phi_{\mathcal M}(t)$, giving the second half of plan (15):

$$
\frac{\delta S_{\text{risk}}}{\delta\rho}(t) = -\frac{2}{\sqrt{n_{\mathcal P}}}\,\big(M_{\mathcal{PM}}G_{\mathcal{MM}}^{-1}\phi_{\mathcal M}(t)\big)^\top K\,\Sigma_{\mathcal U}^{1/2}\,e_{\mathcal U}(t)
$$

### How the code does it

In `first_variation(c::Aliasing, ...)` in [src/costs.jl](../src/costs.jl), each grid point
$t_i$ is one row:

| Code | Math |
|---|---|
| `Z = (M_TM / R) / R'` | row $i$ is $a^\top = (G_{\mathcal{MM}}^{-1}\phi_{\mathcal M}(t_i))^\top$ |
| `Res = M_TU - M_TM * A` | row $i$ is $e_{\mathcal U}(t_i)^\top$ |
| `D` | $\Sigma_{\mathcal U}^{1/2}$ |
| `sum((Z * K) .* (Res * D); dims = 2)` | $a^\top K b$ for every $i$ at once |

The last line is the sandwich (3.6) computed row by row: `(Z * K)[i, :]` is $a^\top K$,
`(Res * D)[i, :]` is $b^\top$, and the elementwise product summed along the row is their dot
product.

---

## 5. The Fisher–Rao gradient and the descent rate

The first variation is a *differential*: it says how $S$ responds to a direction $\eta$. A
*gradient* also needs a way to measure the size of $\eta$, a metric. The plan derives the
Fisher–Rao gradient (20) in function space; here is the same thing on a grid, where it is
ordinary linear algebra.

**Gradient in a metric.** In $\mathbb R^N$ with metric matrix $F$ (length$^2$ of a step
$\eta$ is $\eta^\top F\eta$), the steepest-descent direction is $F^{-1}\nabla S$, not $\nabla S$.

**The Fisher–Rao metric on weights** is $\lVert\eta\rVert^2 = \sum_i \eta_i^2/w_i$, so
$F = \operatorname{diag}(1/w_i)$ and $F^{-1} = \operatorname{diag}(w_i)$. With $g_i = \partial S/\partial w_i$:

$$
(F^{-1}g)_i = w_i\,g_i
$$

**Add the constraint** $\sum_i\eta_i = 0$. Projecting onto allowed directions in this metric
subtracts a constant from $g$ before multiplying by $w$; the constant that makes
$\sum_i w_i(g_i - c) = 0$ is $c = \sum_j w_j g_j$, the $\rho$-average of $g$:

$$
\operatorname{grad} S_i = w_i\big(g_i - \textstyle\sum_j w_j g_j\big)
$$

which is plan (20) on a grid. Two things to notice:

- Points with zero weight get zero gradient, so weights never become negative (and never come
  back from zero).
- The additive constant in $g$ (§2.1) is removed automatically by the subtraction.

**Descent rate (plan (22)).** Along the flow $\dot w_i = -\operatorname{grad}S_i$, the chain
rule gives

$$
\frac{dS}{d\tau} = \sum_i g_i\,\dot w_i = -\sum_i w_i\,g_i\,(g_i - \bar g) = -\sum_i w_i\,(g_i - \bar g)^2 = -\operatorname{Var}_w(g) \le 0
$$

where $\bar g = \sum_j w_j g_j$ and the third equality subtracts $\bar g\sum_i w_i(g_i - \bar g) = 0$.
The cost stops decreasing exactly when $g$ is constant wherever $w_i > 0$: the equality part of
the KKT condition from §2.2.

---

## 6. Checking any first variation numerically

Every formula above can be checked in a few lines, which is how the tests in
[test/test_costs.jl](../test/test_costs.jl) work. Pick an allowed direction
($\sum_i\eta_i = 0$) and compare the formula with a centered finite difference:

```julia
using AliasingDesign, LinearAlgebra, Random

monomial_basis(nb) = [t -> t^k for k in 0:nb-1]
prob = DesignProblem(monomial_basis(7); p = 4, prior = k -> 1 / k^2)
pts = collect(range(-1, 1; length = 15))
w = normalize(rand(15) .+ 0.1, 1)

c = Aliasing()                         # try Aliasing(L2Norm()) too
g = first_variation(c, prob, pts, w)

η = randn(15); η .-= sum(η) / 15       # an allowed direction
h = 1e-6
fd = (cost(c, prob, pts, w + h * η) - cost(c, prob, pts, w - h * η)) / 2h
@show dot(g, η) fd                     # should agree to ~1e-8 relative
```

If they disagree, the usual culprits are a missing factor of 2, a transpose in the wrong place,
or (for the spectral norm) a near-tie between the top two singular values.

---

## 7. Practice problems

Worked solutions are in §8. Try each one with the recipe of §1.4 first; for
§3-style problems, write out where the trace goes at each step.

**Problem 1 (product of two averages).** $S = \Big(\int f\rho\,dt\Big)\Big(\int h\rho\,dt\Big)$.
Find $\frac{\delta S}{\delta\rho}$. (Worked in §1.4 as the example; redo it without looking.)

**Problem 2 (variance of the design).** $S = \operatorname{Var}_\rho(t) = \int t^2\rho\,dt - \Big(\int t\rho\,dt\Big)^2$.
Show that $\frac{\delta S}{\delta\rho}(t) = (t - \bar t)^2$ up to an additive constant.

**Problem 3 (entropy, a pointwise nonlinearity).** $S = \int \rho\log\rho\,dt$. Find
$\frac{\delta S}{\delta\rho}$.

**Problem 4 (A-optimality: a trace of an inverse).** With $G = \int\phi\phi^\top\rho\,dt$, let
$S = \operatorname{tr}(G^{-1})$, the classical A-optimal design criterion. Show that
$\frac{\delta S}{\delta\rho}(t) = -\phi(t)^\top G^{-2}\phi(t)$. Say where the trace goes at each
step.

**Problem 5 (D-optimality).** Using the identity $\delta\log\det G = \operatorname{tr}(G^{-1}\delta G)$,
show that for $S = -\log\det G$, $\frac{\delta S}{\delta\rho}(t) = -\phi(t)^\top G^{-1}\phi(t)$.
Then use §2.2 to state the optimality condition. (This is the Kiefer–Wolfowitz equivalence
theorem: at the D-optimal design, $\phi^\top G^{-1}\phi \le p$ everywhere, with equality on the
support.)

**Problem 6 (risk for the toy problem).** Toy problem with prediction points $x_1,\dots,x_{n_{\mathcal P}}$:
$M_{\mathcal{PM}}$ is a column of ones, and Frobenius risk. Use the risk formula of §4 to show

$$
\frac{\delta S_{\text{risk}}}{\delta\rho}(t) = -2\sigma^2\,\overline{(x - \bar t)}\;(t - \bar t), \qquad
\overline{(x - \bar t)} = \frac{1}{n_{\mathcal P}}\sum_k (x_k - \bar t)
$$

This is what the toy test in [test/test_costs.jl](../test/test_costs.jl) checks (up to a constant).

**Problem 7 (orthonormal modeled basis).** Suppose at the current $\rho$ the modeled functions
are orthonormal: $G_{\mathcal{MM}} = I$. Simplify the Frobenius aliasing first variation, and
interpret each factor.

**Problem 8 (why the spectral norm has no gradient at a tie).** Let $X = I_2$ (both singular
values equal 1) and $S = \sigma_1(X)^2$. Compute the change in $S$ in the direction
$\delta X = \operatorname{diag}(1, -1)$ and in the opposite direction. Explain why no matrix $K$
can satisfy $\delta S = 2\operatorname{tr}(K^\top\delta X)$ for both.

**Problem 9 (code).** Implement the first variation from Problem 4 for a weighted design on a
grid and verify it with the finite-difference check of §6.

---

## 8. Solutions

**Solution to Problem 1.**

Let $m_f = \int f\rho$, $m_h = \int h\rho$. Product rule: $\delta S = m_h\,\delta m_f + m_f\,\delta m_h = \int (m_h f + m_f h)\,\eta\,dt$, so
$\frac{\delta S}{\delta\rho}(t) = m_h\,f(t) + m_f\,h(t)$.

**Solution to Problem 2.**

The first term is linear, giving $t^2$. The second is a square of a linear functional (§1.5),
giving $2\bar t\,t$. So $\frac{\delta S}{\delta\rho} = t^2 - 2\bar t\,t = (t - \bar t)^2 - \bar t^{\,2}$,
and $-\bar t^{\,2}$ is a constant. Interpretation: adding weight far from the mean raises the
variance the most.

**Solution to Problem 3.**

Here $S$ is not a function of a few averages; the integrand is a nonlinear function of
$\rho(t)$ at each point. Differentiate the integrand pointwise:
$\delta(\rho\log\rho) = (\log\rho + 1)\,\eta$, so $\frac{\delta S}{\delta\rho}(t) = \log\rho(t) + 1$.
This is the rule for any $S = \int L(\rho(t))\,dt$: $\frac{\delta S}{\delta\rho} = L'(\rho(t))$.

**Solution to Problem 4.**

1. Trace passes through $\delta$ (3.2): $\delta S = \operatorname{tr}(\delta(G^{-1}))$.
2. Inverse rule (3.5): $= -\operatorname{tr}(G^{-1}\,\delta G\,G^{-1})$.
3. Substitute $\delta G = \int\phi\phi^\top\eta\,dt$ and move the trace inside the integral (3.3):
   $= -\int\operatorname{tr}(G^{-1}\phi\phi^\top G^{-1})\,\eta\,dt$.
4. Collapse with cycling: $\operatorname{tr}(G^{-1}\phi\,\phi^\top G^{-1}) = \phi^\top G^{-1}G^{-1}\phi = \phi^\top G^{-2}\phi$.

So $\frac{\delta S}{\delta\rho}(t) = -\phi(t)^\top G^{-2}\phi(t)$. Negative everywhere: adding
weight anywhere lowers $\operatorname{tr}(G^{-1})$, as expected, since more data always helps.

**Solution to Problem 5.**

$\delta S = -\operatorname{tr}(G^{-1}\delta G) = -\int\operatorname{tr}(G^{-1}\phi\phi^\top)\,\eta\,dt = -\int\phi^\top G^{-1}\phi\;\eta\,dt$
(cycling the trace). KKT from §2.2: $-\phi^\top G^{-1}\phi = \lambda$ on the support and
$\ge\lambda$ off it, i.e. $\phi^\top G^{-1}\phi$ is maximal and constant on the support. Its
$\rho$-average is $\int\phi^\top G^{-1}\phi\,\rho\,dt = \operatorname{tr}(G^{-1}G) = p$, so the
constant is $p$.

**Solution to Problem 6.**

$G_{\mathcal{MM}} = 1$, so $a = M_{\mathcal{PM}}\cdot 1\cdot 1 = \mathbf 1$ (all ones, length $n_{\mathcal P}$).
$R = n_{\mathcal P}^{-1/2}(x - \bar t)\,\sigma$ is an $n_{\mathcal P}\times1$ column, and $K = R$.
$a^\top K = \mathbf 1^\top R = n_{\mathcal P}^{-1/2}\,\sigma\sum_k(x_k - \bar t)$, and
$b = \sigma(t - \bar t)$. So

$$
\frac{\delta S}{\delta\rho}(t) = -\frac{2}{\sqrt{n_{\mathcal P}}}\cdot\frac{\sigma\sum_k(x_k - \bar t)}{\sqrt{n_{\mathcal P}}}\cdot\sigma(t - \bar t)
= -2\sigma^2\,\overline{(x - \bar t)}\,(t - \bar t)
$$

**Solution to Problem 7.**

$A = G_{\mathcal{MU}}$ and $\frac{\delta S}{\delta\rho}(t) = 2\,\phi_{\mathcal M}(t)^\top G_{\mathcal{MU}}\,\Sigma_{\mathcal U}\,e_{\mathcal U}(t)$.
$G_{\mathcal{MU}}\Sigma_{\mathcal U}e_{\mathcal U}(t)$ is the current overlap, weighted by the
prior, in the direction of what the fit misses at $t$; $\phi_{\mathcal M}(t)^\top$ projects it
onto the modeled functions at $t$. Adding weight at $t$ raises aliasing when the point's modeled
features line up with the existing leakage.

**Solution to Problem 8.**

$\sigma_1(I + \varepsilon\operatorname{diag}(1,-1)) = 1 + \lvert\varepsilon\rvert$, so
$S = (1 + \lvert\varepsilon\rvert)^2$ and the one-sided rate is $+2$ in *both* directions. A linear
formula $2\operatorname{tr}(K^\top\delta X)$ must flip sign when $\delta X$ does, so no $K$ works:
$S$ has a kink, not a gradient. The derivation (3.8) failed because $u_1$, $v_1$ were not unique.

**Solution sketch to Problem 9.**

```julia
using LinearAlgebra
Φ = [t^k for t in pts, k in 0:3]                    # n × p, rows φ(tᵢ)ᵀ
G(w) = Φ' * (w .* Φ)
S(w) = tr(inv(G(w)))
g(w) = (Gi = inv(G(w)); -vec(sum((Φ * Gi * Gi) .* Φ; dims = 2)))   # −φᵀ G⁻² φ per row
```

Then compare `dot(g(w), η)` with the centered difference of `S` along a zero-sum `η`.
