"""
    DesignProblem(basis; p, prior = ones(length(basis)), predict = nothing)

A sampling-design problem. `basis` is the full list of basis functions of the true
model; the first `p` are modeled (kept in the fit) and the rest are unmodeled.

- `prior`: prior variance of each coefficient θₖ, as a vector or as a function of the
  basis index `k` (1-based). Its unmodeled block is Σ_𝒰.
- `predict`: the points where predictions are made (rows of M_𝒫). Needed for `Risk`.

The prediction points are processed once, here, since every `Risk` evaluation needs them:

- `n_P`: the number of prediction points.
- `R_P`: `R_𝒫` from the reduced QR factorization `M_𝒫 = Q R_𝒫` (only `R_𝒫` is kept). `Q` has orthonormal columns, so `‖M_𝒫 Y‖ = ‖R_𝒫 Y‖` for every `Y`, in the
  Frobenius and the operator norm alike: the risk costs, their first variations and gradients can
  all use `R_𝒫`, with `min(n_P, p + q)` rows, instead of the `n_P`-row `M_𝒫`, exactly. `R_𝒫` is
  upper triangular, so its first `p` columns vanish below row `p`: the part of the risk that
  depends on the design needs only the top `p` rows (see `prediction_matrices`). The QR is taken
  for short `M_𝒫` too, for that structure.
- `M_P`: `M_𝒫` itself, reused by `design_matrices` when the training points are the prediction
  points themselves (a pool).

Both are stored for the whole basis and split into modeled and unmodeled columns by views, so
`with_modeled` can change `p` without refactoring.
"""
struct DesignProblem{F,P}
    basis::Vector{F}
    p::Int
    prior::Vector{Float64}
    predict::P
    n_P::Int
    R_P::Union{Nothing,Matrix{Float64}}
    M_P::Union{Nothing,Matrix{Float64}}
end

function DesignProblem(basis; p, prior = ones(length(basis)), predict = nothing)
    nbasis = length(basis)
    1 <= p <= nbasis || throw(ArgumentError("p = $p must be between 1 and length(basis) = $nbasis"))
    σ² = prior isa Function ? [float(prior(k)) for k in 1:nbasis] : collect(Float64, prior)
    length(σ²) == nbasis || throw(ArgumentError("prior has length $(length(σ²)), expected $nbasis"))
    all(>=(0), σ²) || throw(ArgumentError("prior variances must be nonnegative"))
    basis = collect(basis)
    predict === nothing && return DesignProblem(basis, p, σ², predict, 0, nothing, nothing)
    M_P = design_matrix(basis, predict)
    n_P = size(M_P, 1)
    R_P = Matrix(qr(M_P).R)        # min(n_P, p + q) rows, upper triangular (trapezoidal)
    return DesignProblem(basis, p, σ², predict, n_P, R_P, M_P)
end

"""
    with_modeled(prob, p)

The same problem with the first `p` basis functions modeled instead of `prob.p`. Shares the
stored prediction matrices, so sweeping `p` (risk-vs-`p` curves) costs nothing extra.
"""
function with_modeled(prob::DesignProblem, p::Integer)
    1 <= p <= length(prob.basis) ||
        throw(ArgumentError("p = $p must be between 1 and length(basis) = $(length(prob.basis))"))
    return DesignProblem(prob.basis, Int(p), prob.prior, prob.predict, prob.n_P, prob.R_P, prob.M_P)
end

"Number of modeled basis functions, p."
nmodeled(prob::DesignProblem) = prob.p

"Number of unmodeled basis functions."
nunmodeled(prob::DesignProblem) = length(prob.basis) - prob.p
