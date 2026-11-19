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

function cost(c::DesignCost, prob::DesignProblem, points, weights)
    err = cost_op(c, prob, points, weights)
    return sqnorm(err, c.norm)
end

"""
    cost_op(c, prob, points)
    cost_op(c, prob, points, weights)

The operator whose squared norm is the cost `c`: one method per cost type.
These are the "X_..." operators in the writeup.
"""
cost_op(c::Aliasing, prob::DesignProblem, points) =
aliasing_operator(prob, points)*Diagonal(sqrt.(prob.prior[prob.p+1:end]))
cost_op(c::Aliasing, prob::DesignProblem, points, weights) =
aliasing_operator(prob, points, weights)*Diagonal(sqrt.(prob.prior[prob.p+1:end]))

cost_op(c::Risk, prob::DesignProblem, points) =
design_matrix(prob.basis, prob.predict)*error_operator(prob, points)*Diagonal(sqrt.(prob.prior))/sqrt(length(prob.predict))
cost_op(c::Risk, prob::DesignProblem, points, weights) =
design_matrix(prob.basis, prob.predict)*error_operator(prob, points, weights)*Diagonal(sqrt.(prob.prior))/sqrt(length(prob.predict))

"""
    cost_gradient(c, prob, points)

Gradient of `cost(c, prob, points)` with respect to each point `tᵢ`.
Drives the Wasserstein particle flow. Points must be real numbers.
"""
function cost_gradient(c::DesignCost, prob::DesignProblem, points)
    error("cost_gradient is not implemented for $(typeof(c))")
end

"""
    first_variation(c, prob, points, weights)

The first variation `δS/δρ` evaluated at each point, for the density given by
`weights` on `points`. Defined up to an additive constant. Drives the Fisher–Rao flow.
"""
function first_variation(c::Aliasing, prob::DesignProblem, points, weights)
    (; M_TM, M_TU, R, A) = weighted_fit(prob, points, weights)
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
function first_variation(c::Risk, prob::DesignProblem, points, weights)
    (; M_TM, M_TU, R, A) = weighted_fit(prob, points, weights)
    (M_PM, M_PU) = design_matrices(prob, prob.predict)
    D = Diagonal(sqrt.(prob.prior[prob.p+1:end]))
    s = 1 / sqrt(length(prob.predict))
    K = get_K(c.norm, s * (M_PU - M_PM * A) * D)
    Z = (M_TM / R) / R'          # row i: (G_MM⁻¹ m(tᵢ))ᵀ
    Res = M_TU - M_TM * A        # row i: (u(tᵢ) − Aᵀ m(tᵢ))ᵀ
    return -2s .* vec(sum((Z * M_PM' * K) .* (Res * D); dims = 2))
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
    optimize_weights(c, prob, points, weights; maxiter = 10_000, rtol = 1e-8, η = 1.0)
        -> (; weights, cost, λ, residual, iterations, converged)

Minimize the cost `c` over weights on `points` with the Fisher–Rao flow (exponentiated
gradient), starting from `weights` (all positive).

Step size: `h = η / max|g − λ|` from the *first* gradient, then held fixed, so the steps
shrink as the gradient does. (Recomputing it every step would make the step length constant,
and the weights would oscillate around the optimum without converging.) If a step would raise
the cost, `h` is halved and the step retried.

Returns a named tuple:
- `weights`: the final weights.
- `cost`: the cost at the final weights.
- `λ`: the Lagrange multiplier, the `weights`-average of the first variation `g`.
- `residual`: the largest violation of the KKT condition at the final weights (see
  `kkt_residual`).
- `iterations`: the number of weight updates taken.
- `converged`: whether `residual ≤ rtol * max(maximum(abs, g), cost)` was reached within `maxiter`.
"""
function optimize_weights(c::DesignCost, prob::DesignProblem, points, weights;
                          maxiter = 10_000, rtol = 1e-8, η = 1.0)
    λ, residual, iterations, converged = NaN, Inf, 0, false
    S = cost(c, prob, points, weights)
    step_size = NaN
    for iter in 1:maxiter
        δS_δρ = first_variation(c, prob, points, weights)
        λ = weights' * δS_δρ
        residual = kkt_residual(δS_δρ, weights, λ)
        # g has the units of the cost (weights are dimensionless). At an exact optimum g can be
        # pure rounding noise, so scale by the cost too, not only by max|g|.
        if residual <= rtol * max(maximum(abs, δS_δρ), S)
            converged = true
            break
        end
        isnan(step_size) && (step_size = η / maximum(abs, δS_δρ .- λ))   # set once
        while true
            trial = weights .* exp.(-step_size .* (δS_δρ .- λ))   # −λ only guards against overflow
            trial ./= sum(trial)
            S_trial = cost(c, prob, points, trial)
            if S_trial <= S || step_size < 1e-300
                weights, S = trial, S_trial
                break
            end
            step_size /= 2
        end
        iterations = iter
    end
    if !converged   # the loop's λ and residual belong to the weights before the last update
        δS_δρ = first_variation(c, prob, points, weights)
        λ = weights' * δS_δρ
        residual = kkt_residual(δS_δρ, weights, λ)
    end
    return (; weights, cost = cost(c, prob, points, weights), λ, residual, iterations, converged)
end

"""
    kkt_residual(g, w, λ)

Largest violation of the KKT condition (16): `|gᵢ − λ|` where `wᵢ` is in the support, and
`max(0, λ − gᵢ)` where `wᵢ` is essentially zero (below `1e-6/length(w)`). Zero at an optimum.
"""
function kkt_residual(g, w, λ)
    almost_zero_weight = 1e-6/length(w)
    residual = 0.0
    for (gi, wi) in zip(g, w)
        violation = wi < almost_zero_weight ? max(0.0, λ - gi) : abs(gi - λ)
        residual = max(residual, violation)
    end
    return residual
end

check_KKT(g, w, λ, tol) = kkt_residual(g, w, λ) <= tol
