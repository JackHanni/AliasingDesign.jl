"Matrix norms a cost can use. Both are squared."
abstract type DesignNorm end

"Squared Frobenius norm: the expected squared error under the prior. Smooth."
struct Frobenius <: DesignNorm end

sqnorm(X, ::Frobenius) = sum(abs2, X)

"Squared operator norm: the worst-case squared error over prior-weighted θ."
struct L2Norm <: DesignNorm end

sqnorm(X, ::L2Norm) = opnorm(X, 2)^2
abstract type DesignCost end

"""
    Aliasing(norm = Frobenius())

Size of the prior-weighted aliasing operator, `‖A Σ_𝒰^{1/2}‖²`. With `Frobenius()`
this is `S = tr(A Σ_𝒰 Aᵀ)`.
"""
struct Aliasing{N<:DesignNorm} <: DesignCost
    norm::N
end
Aliasing() = Aliasing(Frobenius())

"""
    Risk(norm = Frobenius())

Size of the prior-weighted prediction-error operator,
`‖M_𝒫 E_θ Σ^{1/2}‖² / n_𝒫`, averaged over the problem's prediction points.
"""
struct Risk{N<:DesignNorm} <: DesignCost
    norm::N
end
Risk() = Risk(Frobenius())

"""
    cost(c, prob, points)
    cost(c, prob, points, weights)

The cost at a finite set of training points, or at the sampling density given
by `weights` on `points` (the many-samples limit, which takes `B = I`).
"""
function cost(c::DesignCost, prob::DesignProblem, points)
    err = cost_op(c, prob, points)
    return sqnorm(err, c.norm)
end

cost(c::DesignCost, prob::DesignProblem, points, weights) =
    fit_cost(c, prob, weighted_fit(prob, points, weights))

# The weighted cost, first variation and noise term from a precomputed `weighted_fit`, so that
# callers (`optimize_weights`, `WithNoise`) can share one QR factorization among them.
fit_cost(c::Union{Aliasing,Risk}, prob::DesignProblem, fit) = sqnorm(fit_cost_op(c, prob, fit), c.norm)

"""
    cost_op(c, prob, points)
    cost_op(c, prob, points, weights)

The operator whose squared norm is the cost `c`: one method per cost type.
These are the "X_..." operators in the writeup.
"""
cost_op(c::Aliasing, prob::DesignProblem, points) =
    aliasing_operator(prob, points) * prior_sqrt_U(prob)
cost_op(c::Union{Aliasing,Risk}, prob::DesignProblem, points, weights) =
    fit_cost_op(c, prob, weighted_fit(prob, points, weights))

fit_cost_op(c::Aliasing, prob::DesignProblem, fit) = fit.A * prior_sqrt_U(prob)
fit_cost_op(c::Risk, prob::DesignProblem, fit) = risk_operator(prob, nothing, fit.A)  # B = I

# The risk cost without building its whole operator: see `risk_blocks`.
function cost(c::Risk, prob::DesignProblem, points)
    (M_TM, M_TU) = design_matrices(prob, points)
    P = pinv(M_TM)
    return risk_sqnorm(c.norm, prob, I - P * M_TM, P * M_TU)
end
fit_cost(c::Risk, prob::DesignProblem, fit) = risk_sqnorm(c.norm, prob, nothing, fit.A)

risk_sqnorm(::Frobenius, prob::DesignProblem, I_minus_B, A) =
    sum(abs2, risk_blocks(prob, I_minus_B, A)) + risk_fixed_sqnorm(prob)
risk_sqnorm(norm::L2Norm, prob::DesignProblem, I_minus_B, A) =
    sqnorm(risk_operator(prob, I_minus_B, A), norm)

# The top rows of K = get_K(norm, X_risk), the only ones the first variation and the gradient use
# (they meet K through R_PMᵀ, which is zero below row r; see `risk_blocks`).
risk_K(::Frobenius, prob::DesignProblem, I_minus_B, A) = risk_blocks(prob, I_minus_B, A)
risk_K(norm::L2Norm, prob::DesignProblem, I_minus_B, A) =
    get_K(norm, risk_operator(prob, I_minus_B, A))[1:size(first(prediction_matrices(prob)), 1), :]

# X_risk = M_𝒫 E_θ Σ^{1/2} / √n_𝒫, computed by blocks: M_𝒫 E_θ = [M_PM (I − B)   M_PU − M_PM A].
# Without forming the (p+q)×(p+q) E_θ, it costs O(n p q) instead of O(n (p+q)²). With M_𝒫 replaced by
# R_𝒫 (M_𝒫 = Q R_𝒫, see `DesignProblem`) the rows drop from n_𝒫 to at most p + q: the returned
# matrix is Qᵀ X_risk, with the same singular values, so every cost is unchanged.
function cost_op(c::Risk, prob::DesignProblem, points)
    (M_TM, M_TU) = design_matrices(prob, points)
    P = pinv(M_TM)
    return risk_operator(prob, I - P * M_TM, P * M_TU)
end

# `I_minus_B = nothing` stands for B = I (the weighted limit): a zero modeled block.
risk_operator(prob::DesignProblem, I_minus_B, A) =
    [risk_blocks(prob, I_minus_B, A); risk_fixed_rows(prob)]

# R_𝒫 is upper triangular, so its modeled columns R_PM vanish below row r = min(p, rows of R_𝒫):
# only the top r rows of X_risk depend on the design. They are returned here, at a cost of
# O(r p (p + q)) instead of O(rows · p (p + q)); the rest of X_risk, [0  R_PU,low Σ_𝒰^{1/2}] / √n_𝒫,
# is fixed for a given p (`risk_fixed_rows`, `risk_fixed_sqnorm`).
function risk_blocks(prob::DesignProblem, I_minus_B, A)
    (R_PM, R_PU) = prediction_matrices(prob)
    r, p = size(R_PM)
    d = sqrt.(prob.prior) ./ sqrt(prob.n_P)
    X_U = (view(R_PU, 1:r, :) .- R_PM * A) .* view(d, p+1:length(d))'
    X_M = I_minus_B === nothing ? zeros(r, p) : (R_PM * I_minus_B) .* view(d, 1:p)'
    return [X_M X_U]
end

function risk_fixed_rows(prob::DesignProblem)
    (R_PM, R_PU) = prediction_matrices(prob)
    r, p = size(R_PM)
    d_U = sqrt.(prob.prior[p+1:end]) ./ sqrt(prob.n_P)
    low = view(R_PU, r+1:size(R_PU, 1), :)
    return [zeros(size(low, 1), p)  low .* d_U']
end

function risk_fixed_sqnorm(prob::DesignProblem)
    (R_PM, R_PU) = prediction_matrices(prob)
    r, p = size(R_PM)
    total = 0.0
    for (j, k) in enumerate(p+1:length(prob.prior))
        total += prob.prior[k] * sum(abs2, view(R_PU, r+1:size(R_PU, 1), j))
    end
    return total / prob.n_P
end

"Σ^{1/2}, the square root of the prior covariance of all `p + q` coefficients."
prior_sqrt(prob::DesignProblem) = Diagonal(sqrt.(prob.prior))

"Σ_𝒰^{1/2}, the square root of the prior covariance of the unmodeled coefficients."
prior_sqrt_U(prob::DesignProblem) = Diagonal(sqrt.(prob.prior[prob.p+1:end]))

"""
    prediction_matrices(prob) -> (R_PM, R_PU)

The modeled and unmodeled blocks of the compressed prediction matrix `R_𝒫` (`M_𝒫 = Q R_𝒫`; see
`DesignProblem`). Any norm of `M_𝒫 Y` equals that of `R_𝒫 Y`, and `M_PMᵀ M_P· = R_PMᵀ R_P·`;
divide by `prob.n_P`, not by the row count. `R_𝒫` is upper triangular, so `R_PM` is zero below row
`r = min(p, rows)` and is returned as its top `r` rows only (`r × p`; the same norms and inner
products); `R_PU` keeps every row. Errors if `predict` was not given.
"""
function prediction_matrices(prob::DesignProblem)
    prob.R_P === nothing &&
        throw(ArgumentError("Risk needs prediction points: pass `predict` to DesignProblem"))
    R = prob.R_P
    r = min(prob.p, size(R, 1))
    return (view(R, 1:r, 1:prob.p), view(R, :, prob.p+1:size(R, 2)))
end

"""
    WithNoise(base, τ)

The cost `base` (an `Aliasing` or `Risk`) plus measurement noise: `S_τ = S + τ V`, eq. (9a).
`τ = σ_ϵ²/n` is the noise variance per sample. The variance term is

- `Aliasing`: `V = tr(G_MM⁻¹)`, the A-optimality criterion;
- `Risk`: `V = tr(G_MM⁻¹ G_𝒫)`, the I-optimality criterion,

with `G_MM = M_TMᵀM_TM / n` at `n` finite points (`n ‖M_TM⁺‖²_F` in general, which stays finite
for `n < p`: the minimum-norm fit's variance) and `G_MM[ρ]` at a density. `V` is a trace, so it
does not depend on `base`'s norm. `τ = 0` is the pure-aliasing cost; `τ → ∞` the classical design.
"""
struct WithNoise{C<:Union{Aliasing,Risk}} <: DesignCost
    base::C
    τ::Float64
    function WithNoise(base::C, τ::Real) where {C<:Union{Aliasing,Risk}}
        τ >= 0 || throw(ArgumentError("noise level τ = $τ must be nonnegative"))
        return new{C}(base, τ)
    end
end

cost(c::WithNoise, prob::DesignProblem, points) =
    cost(c.base, prob, points) + c.τ * noise_variance(c.base, prob, points)
fit_cost(c::WithNoise, prob::DesignProblem, fit) =
    fit_cost(c.base, prob, fit) + c.τ * fit_noise_variance(c.base, prob, fit)

"""
    noise_variance(c, prob, points[, weights])

The variance term `V` of eq. (9a) for the cost type of `c` (see `WithNoise`).
"""
function noise_variance(c::Aliasing, prob::DesignProblem, points)
    (M_TM, _) = design_matrices(prob, points)
    return length(points) * sum(abs2, pinv(M_TM))
end

function noise_variance(c::Risk, prob::DesignProblem, points)
    (M_TM, _) = design_matrices(prob, points)
    (R_PM, _) = prediction_matrices(prob)
    return length(points) / prob.n_P * sum(abs2, R_PM * pinv(M_TM))
end

noise_variance(c::Union{Aliasing,Risk}, prob::DesignProblem, points, weights) =
    fit_noise_variance(c, prob, weighted_fit(prob, points, weights))

# G_MM = RᵀR, so tr(G_MM⁻¹) = ‖R⁻¹‖²_F and tr(G_MM⁻¹G_𝒫) = ‖M_PM R⁻¹‖²_F / n_𝒫 = ‖R_PM R⁻¹‖²_F / n_𝒫.
fit_noise_variance(c::Aliasing, prob::DesignProblem, fit) = sum(abs2, inv(fit.R))
fit_noise_variance(c::Risk, prob::DesignProblem, fit) =
    sum(abs2, first(prediction_matrices(prob)) / fit.R) / prob.n_P

"""
    first_variation(c, prob, points, weights)

The first variation `δS/δρ` evaluated at each point, for the density given by
`weights` on `points`. Defined up to an additive constant. Drives the Fisher–Rao flow.
"""
first_variation(c::DesignCost, prob::DesignProblem, points, weights) =
    fit_first_variation(c, prob, weighted_fit(prob, points, weights))

function fit_first_variation(c::Aliasing, prob::DesignProblem, fit)
    (; M_TM, M_TU, R, A) = fit
    D = Diagonal(sqrt.(prob.prior[prob.p+1:end]))
    K = get_K(c.norm, A * D)
    Z = (M_TM / R) / R'          # row i: (G_MM⁻¹ m(tᵢ))ᵀ
    Res = M_TU - M_TM * A        # row i: (u(tᵢ) − Aᵀ m(tᵢ))ᵀ
    return 2 .* vec(sum((Z * K) .* (Res * D); dims = 2))
end

# Risk: only the unmodeled block of X_risk depends on ρ (B = I in the weighted limit),
#     X_U = n_P^{-1/2} (M_PU − M_PM A) Σ_𝒰^{1/2},   δX_U = −n_P^{-1/2} M_PM δA Σ_𝒰^{1/2},
# so with K = get_K(norm, X_U) and the same δA as for Aliasing,
#     δS/δρ(t) = −(2/√n_P) (M_PM G_MM⁻¹ m(t))ᵀ K Σ_𝒰^{1/2} (u(t) − Aᵀ m(t)).
# X_U is built from the A already computed, so the QR factorization runs once.
# Dropping the zero modeled-block columns of X_risk changes neither norm.
function fit_first_variation(c::Risk, prob::DesignProblem, fit)
    (; M_TM, M_TU, R, A) = fit
    (M_PM, _) = prediction_matrices(prob)       # compressed: the top r rows of R_PM
    p = prob.p
    D = Diagonal(sqrt.(prob.prior[p+1:end]))
    s = 1 / sqrt(prob.n_P)
    K = risk_K(c.norm, prob, nothing, A)[:, p+1:end]   # top rows of K_U (X_U already has s and D)
    Z = (M_TM / R) / R'          # row i: (G_MM⁻¹ m(tᵢ))ᵀ
    Res = M_TU - M_TM * A        # row i: (u(tᵢ) − Aᵀ m(tᵢ))ᵀ
    # Z * (M_PMᵀ K), not (Z M_PMᵀ) K: the left grouping builds an N × n_𝒫 matrix first, which
    # dominates the cost when the pool doubles as the prediction set.
    return -2s .* vec(sum((Z * (M_PM' * K)) .* (Res * D); dims = 2))
end

# Noise term, eq. (9b): δV_alias/δρ(t) = −‖G_MM⁻¹ m(t)‖², δV_risk/δρ(t) = −‖M_PM G_MM⁻¹ m(t)‖² / n_𝒫.
fit_first_variation(c::WithNoise, prob::DesignProblem, fit) =
    fit_first_variation(c.base, prob, fit) .+ c.τ .* fit_noise_first_variation(c.base, prob, fit)

function fit_noise_first_variation(c::Aliasing, prob::DesignProblem, fit)
    (; M_TM, R) = fit
    Z = (M_TM / R) / R'          # row i: (G_MM⁻¹ m(tᵢ))ᵀ
    return -vec(sum(abs2, Z; dims = 2))
end

function fit_noise_first_variation(c::Risk, prob::DesignProblem, fit)
    (; M_TM, R) = fit
    (R_PM, _) = prediction_matrices(prob)
    Z = (M_TM / R) / R'
    # ‖M_PM zᵢ‖² = zᵢᵀ (M_PMᵀM_PM) zᵢ = zᵢᵀ (R_PMᵀR_PM) zᵢ, through the small p × p Gram matrix.
    return -vec(sum((Z * (R_PM' * R_PM)) .* Z; dims = 2)) ./ prob.n_P
end

get_K(c::DesignCost, prob::DesignProblem, points, weights) =
    get_K(c.norm, cost_op(c, prob, points, weights))

# Half the derivative of the squared norm with respect to the operator X:
# d‖X‖²_F = 2X, and d(σ₁²) = 2σ₁u₁v₁ᵀ.
get_K(::Frobenius, X) = X
get_K(::L2Norm, X) = top_rank_one(X)

# Below this smaller dimension a full SVD is cheap (cost ~ m·n·min(m, n)), and
# `tsvd` is unreliable there: it throws on 1×1 matrices and hangs on single columns.
const DENSE_SVD_MAXDIM = 32

"""
    top_rank_one(K)

The best rank-one approximation `σ₁ u₁ v₁ᵀ` of `K`. Uses a full SVD when
`min(size(K)) ≤ DENSE_SVD_MAXDIM` and Lanczos (`tsvd`) otherwise.
"""
function top_rank_one(K::AbstractMatrix)
    if min(size(K)...) <= DENSE_SVD_MAXDIM
        F = svd(K)
        return F.S[1] * F.U[:, 1] * F.V[:, 1]'
    end
    u, s, v = tsvd(K, 1; maxiter = 100)
    return s[1] * u[:, 1] * v[:, 1]'
end

"""
    optimize_weights(c, prob, points, weights; maxiter = 10_000, rtol = 1e-8, η = 1.0, cap = nothing,
                     step = :bb, grow = 1.5)
        -> (; weights, cost, λ, residual, iterations, converged)

Minimize the cost `c` over weights on `points` with the Fisher–Rao flow (exponentiated
gradient), starting from `weights` (all positive).

With `cap`, each weight is also held at most `cap` (use `cap = 1/n` on a pool, so that each
structure counts once; eq. (28)): after every update, weights over the cap are clipped and the
rest renormalized (`project_capped`).

Step size: `h = η / max|g − λ|` from the *first* gradient. It is not recomputed from later
gradients, so the steps shrink as the gradient does. (Recomputing it every step would make the step
length constant, and the weights would oscillate around the optimum without converging.) If a step
would raise the cost (or would make `G_MM` singular, giving a NaN cost), `h` is halved and the step
retried; if 50 halvings in a row (a step ~1e-15 times the last one that worked) do not lower the
cost, the cost is at a rounding-level plateau and the run stops there. After an accepted
step, `h` changes by the rule `step`:

- `:grow`: `h *= grow` after every accepted step, so a step size that started (or was halved) too
  small recovers; `grow = 1` keeps it fixed.
- `:clean`: `h *= grow` only after a step accepted at the first try (none refused this iteration).
- `:ratio` (trust-region style): compare the decrease with the first-order prediction
  `⟨g, w − w_trial⟩`; `h *= 2` when their ratio is above 3/4, `h /= 2` when below 1/4.
- `:bb` (Barzilai–Borwein, the default): `h = ⟨s, W⁻¹s⟩ / ⟨s, y⟩` from the last step `s = Δw` and the change in
  the first variation `y = Δg`, in the Fisher–Rao metric `W⁻¹ = diag(1/w)`; kept within a factor
  of 10 of the last step; `h *= grow` when `⟨s, y⟩ ≤ 0`. Still monotone (refused steps halve).
  Fastest by far in our benchmarks (`experiments/benchmark_optimizer.jl`): 17× fewer seconds on
  the cluster-expansion pool at p = 100, 3–9× on 1D Legendre problems, same costs.

Returns a named tuple:
- `weights`: the final weights.
- `cost`: the cost at the final weights.
- `λ`: the Lagrange multiplier (see `kkt_multiplier`); without a cap, the `weights`-average
  of the first variation `g` over the support.
- `residual`: the largest violation of the KKT condition at the final weights (see
  `kkt_residual`).
- `iterations`: the number of weight updates taken.
- `converged`: whether `residual ≤ rtol * max(maximum(abs, g), cost)` was reached within `maxiter`.
"""
function optimize_weights(c::DesignCost, prob::DesignProblem, points, weights;
                          maxiter = 10_000, rtol = 1e-8, η = 1.0, cap = nothing, step = :bb,
                          grow = 1.5)
    step in (:grow, :clean, :ratio, :bb) ||
        throw(ArgumentError("step must be :grow, :clean, :ratio or :bb, got $(repr(step))"))
    weights = cap === nothing ? weights ./ sum(weights) : project_capped(weights, cap)
    length(weights) == length(points) ||
        throw(ArgumentError("got $(length(points)) points and $(length(weights)) weights"))
    # The points are fixed, so build the design matrices once. Each accepted step's fit (one QR)
    # then serves both its cost and the next first variation.
    (M_TM, M_TU) = design_matrices(prob, points)
    fit = weighted_fit(M_TM, M_TU, weights)
    S = fit_cost(c, prob, fit)
    λ, residual, iterations, converged = NaN, Inf, 0, false
    step_size = NaN
    last_w, last_g = nothing, nothing             # for :bb
    for iter in 1:maxiter
        δS_δρ = fit_first_variation(c, prob, fit)
        λ = kkt_multiplier(δS_δρ, weights; cap)
        residual = kkt_residual(δS_δρ, weights, λ; cap)
        # g has the units of the cost (weights are dimensionless). At an exact optimum g can be
        # pure rounding noise, so scale by the cost too, not only by max|g|.
        if residual <= rtol * max(maximum(abs, δS_δρ), S)
            converged = true
            break
        end
        isnan(step_size) && (step_size = η / maximum(abs, δS_δρ .- λ))   # set once
        if step === :bb && last_w !== nothing
            step_size = bb_step(weights .- last_w, δS_δρ .- last_g, weights, step_size, grow)
        end
        stalled = false
        halvings = 0
        while true
            # Zero weights stay zero under the multiplicative update, so update only the positive
            # ones (at a zero weight, exp of a large exponent is Inf, and 0 · Inf = NaN).
            # Subtracting their largest exponent leaves the normalized update unchanged and keeps
            # exp from overflowing.
            pos = weights .> 0
            z = -step_size .* (δS_δρ[pos] .- λ)
            trial = zeros(length(weights))
            trial[pos] = weights[pos] .* exp.(z .- maximum(z))
            trial = cap === nothing ? trial ./ sum(trial) : project_capped(trial, cap)
            S_trial, fit_trial = trial_cost(c, prob, M_TM, M_TU, trial)
            if S_trial <= S                  # false for NaN: a step to a singular G_MM is refused
                predicted = sum(δS_δρ .* (weights .- trial))       # first-order decrease (λ cancels)
                step === :bb && ((last_w, last_g) = (weights, δS_δρ))
                weights, S_old, S, fit = trial, S, S_trial, fit_trial
                if step === :grow || (step === :clean && halvings == 0)
                    step_size *= grow
                elseif step === :ratio && predicted > 0
                    ρ = (S_old - S) / predicted
                    step_size *= ρ > 3/4 ? 2.0 : ρ < 1/4 ? 0.5 : 1.0
                end
                break
            end
            step_size /= 2
            halvings += 1
            # No step lowers the cost: stop. 50 halvings (2⁻⁵⁰ ≈ 1e-15 of the last step that
            # worked) only fail at a rounding-level plateau, i.e. at the optimum; halving further
            # would cost a refit each time for nothing.
            if halvings >= MAX_HALVINGS || step_size < 1e-300
                stalled = true
                break
            end
        end
        stalled && break
        iterations = iter
    end
    if !converged   # the loop's λ and residual belong to the weights before the last update
        δS_δρ = fit_first_variation(c, prob, fit)
        λ = kkt_multiplier(δS_δρ, weights; cap)
        residual = kkt_residual(δS_δρ, weights, λ; cap)
    end
    return (; weights, cost = S, λ, residual, iterations, converged)
end

const MAX_HALVINGS = 50

# Barzilai–Borwein step for the Fisher–Rao (mirror) update, in the metric diag(1/w): the first
# BB formula ⟨s, W⁻¹s⟩/⟨s, y⟩, within a factor of 10 of the previous step; grow it when the
# curvature estimate ⟨s, y⟩ is not positive.
function bb_step(s, y, w, h, grow)
    pos = w .> 0
    sy = sum(s[pos] .* y[pos])
    sy > 0 || return h * grow
    h_bb = sum(abs2.(s[pos]) ./ w[pos]) / sy
    return clamp(h_bb, h / 10, 10h)
end

# The cost and fit at trial weights, or (Inf, nothing) if they leave G_MM exactly singular (fewer
# than p support points), so the optimizer refuses that step like any other that raises the cost.
function trial_cost(c, prob, M_TM, M_TU, weights)
    try
        fit = weighted_fit(M_TM, M_TU, weights)
        return fit_cost(c, prob, fit), fit
    catch err
        err isa SingularException && return Inf, nothing
        rethrow()
    end
end

"""
    project_capped(w, cap)

Rescale the nonnegative weights `w` to sum to 1 with every weight at most `cap`: weights that
would exceed `cap` are set to `cap`, and the rest are scaled up proportionally to fill the
remaining mass, repeated until none exceeds `cap`. This keeps the multiplicative update's
proportions among the uncapped weights. Needs `cap * length(w) ≥ 1`.
"""
function project_capped(w, cap)
    cap * length(w) >= 1 - 1e-12 ||
        throw(ArgumentError("cap = $cap is too small for $(length(w)) weights to sum to 1"))
    w = collect(float.(w))
    capped = falses(length(w))
    while true
        free_mass = 1 - cap * count(capped)
        free_sum = sum(w[.!capped])
        scale = free_sum > 0 ? free_mass / free_sum : 0.0
        over = .!capped .& (w .* scale .> cap)
        if !any(over)
            w[.!capped] .*= scale
            w[capped] .= cap
            return w
        end
        capped .|= over
    end
end

# A weight counts as zero below `1e-6/N`, and as capped within a relative 1e-9 of the cap.
is_zero_weight(wi, N) = wi < 1e-6 / N
is_capped_weight(wi, cap) = cap !== nothing && wi >= cap * (1 - 1e-9)

"""
    kkt_multiplier(g, w; cap = nothing)

The Lagrange multiplier `λ` that goes with the first variation `g` at weights `w`: the
`w`-average of `g` over the free weights (in the support and, with a `cap`, below it). With no
free weight (every weight 0 or at the cap), the midpoint of the interval of valid `λ`.
"""
function kkt_multiplier(g, w; cap = nothing)
    N = length(w)
    free = [!is_zero_weight(wi, N) && !is_capped_weight(wi, cap) for wi in w]
    any(free) && return sum(w[free] .* g[free]) / sum(w[free])
    at_cap = [is_capped_weight(wi, cap) for wi in w]
    lo = any(at_cap) ? maximum(g[at_cap]) : -Inf       # λ ≥ g at the cap
    hi = any(.!at_cap) ? minimum(g[.!at_cap]) : Inf    # λ ≤ g at zero
    return isfinite(lo) && isfinite(hi) ? (lo + hi) / 2 : (isfinite(lo) ? lo : hi)
end

"""
    kkt_residual(g, w, λ; cap = nothing)

Largest violation of the KKT condition: eq. (16) without a cap, eq. (28) with one.
`|gᵢ − λ|` where `wᵢ` is free, `max(0, λ − gᵢ)` where `wᵢ` is essentially zero (below
`1e-6/length(w)`), and `max(0, gᵢ − λ)` where `wᵢ` is at the cap. Zero at an optimum.
"""
function kkt_residual(g, w, λ; cap = nothing)
    N = length(w)
    residual = 0.0
    for (gi, wi) in zip(g, w)
        violation = is_zero_weight(wi, N) ? max(0.0, λ - gi) :
                    is_capped_weight(wi, cap) ? max(0.0, gi - λ) : abs(gi - λ)
        residual = max(residual, violation)
    end
    return residual
end

check_KKT(g, w, λ, tol; cap = nothing) = kkt_residual(g, w, λ; cap) <= tol

# Compact printing, e.g. `Aliasing(L2Norm)` and `WithNoise(Risk(Frobenius), τ = 0.001)`.
Base.show(io::IO, c::Union{Aliasing,Risk}) = print(io, nameof(typeof(c)), "(", nameof(typeof(c.norm)), ")")
Base.show(io::IO, c::WithNoise) = print(io, "WithNoise(", c.base, ", τ = ", c.τ, ")")
