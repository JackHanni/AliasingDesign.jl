"""
    DesignProblem(basis; p, prior = ones(length(basis)), predict = nothing)

A sampling-design problem. `basis` is the full list of basis functions of the true
model; the first `p` are modeled (kept in the fit) and the rest are unmodeled.

- `prior`: prior variance of each coefficient θₖ, as a vector or as a function of the
  basis index `k` (1-based). Its unmodeled block is Σ_𝒰.
- `predict`: the points where predictions are made (rows of M_𝒫). Needed for `Risk`.

The prediction matrix `M_𝒫` is built once here and stored, since every `Risk` evaluation needs
it and the prediction points never change.
"""
struct DesignProblem{F,P}
    basis::Vector{F}
    p::Int
    prior::Vector{Float64}
    predict::P
    M_P::Union{Nothing,Matrix{Float64}}
end

function DesignProblem(basis; p, prior = ones(length(basis)), predict = nothing)
    nbasis = length(basis)
    1 <= p <= nbasis || throw(ArgumentError("p = $p must be between 1 and length(basis) = $nbasis"))
    σ² = prior isa Function ? [float(prior(k)) for k in 1:nbasis] : collect(Float64, prior)
    length(σ²) == nbasis || throw(ArgumentError("prior has length $(length(σ²)), expected $nbasis"))
    all(>=(0), σ²) || throw(ArgumentError("prior variances must be nonnegative"))
    basis = collect(basis)
    M_P = predict === nothing ? nothing : design_matrix(basis, predict)
    return DesignProblem(basis, p, σ², predict, M_P)
end

"Number of modeled basis functions, p."
nmodeled(prob::DesignProblem) = prob.p

"Number of unmodeled basis functions."
nunmodeled(prob::DesignProblem) = length(prob.basis) - prob.p
