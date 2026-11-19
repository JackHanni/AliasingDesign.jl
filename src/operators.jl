"""
    design_matrix(basis, points)

Evaluate each basis function at each point: `M[i, j] = basis[j](points[i])`.
"""
design_matrix(basis, points) =
    isempty(basis) ? zeros(length(points), 0) : [float(f(t)) for t in points, f in basis]

"""
    design_matrices(prob, points) -> (M_TM, M_TU)

The modeled and unmodeled blocks of the design matrix at `points`.
"""
function design_matrices(prob::DesignProblem, points)
    # A pool whose candidates are also the prediction points: reuse the stored M_𝒫.
    # The returned blocks are shared with `prob`: callers must not modify them.
    if prob.M_P !== nothing && points === prob.predict
        return (view(prob.M_P, :, 1:prob.p), view(prob.M_P, :, prob.p+1:size(prob.M_P, 2)))
    end
    M_TM = design_matrix(prob.basis[1:prob.p], points)
    M_TU = design_matrix(prob.basis[prob.p+1:end], points)
    return (M_TM, M_TU)
end

"""
    aliasing_operator(M_TM, M_TU)
    aliasing_operator(prob, points)
    aliasing_operator(prob, points, weights)

The GAD aliasing operator `A = M_TM⁺ M_TU`: how unmodeled coefficients leak into
the fitted modeled coefficients.

With `weights` (nonnegative, summing to 1), the points are a discretized sampling
density ρ and `A[ρ] = G_MM[ρ]⁻¹ G_MU[ρ]`, the many-samples limit.
"""
aliasing_operator(M_TM, M_TU) = pinv(M_TM) * M_TU

function aliasing_operator(prob::DesignProblem, points)
    (M_TM, M_TU) = design_matrices(prob, points)
    return aliasing_operator(M_TM, M_TU)
end

aliasing_operator(prob::DesignProblem, points, weights) =
    weighted_fit(prob, points, weights).A

"""
    weighted_fit(prob, points, weights) -> (; M_TM, M_TU, R, A)

The weighted least-squares fit behind `A[ρ]`, without forming or inverting `G_MM`.
Takes a QR factorization `√W M_TM = Q R`, so `RᵀR = G_MM` and `A = R⁻¹ Qᵀ √W M_TU`.
Working with `R` keeps the conditioning of `√W M_TM` instead of squaring it as
`G_MM` does. Returns `R` so callers can apply `G_MM⁻¹ = R⁻¹R⁻ᵀ` with two triangular
solves.

Needs `G_MM` invertible: at least `p` points with positive weight in general position.
"""
function weighted_fit(prob::DesignProblem, points, weights)
    length(weights) == length(points) ||
        throw(ArgumentError("got $(length(points)) points and $(length(weights)) weights"))
    all(>=(0), weights) || throw(ArgumentError("weights must be nonnegative"))
    (M_TM, M_TU) = design_matrices(prob, points)
    return weighted_fit(M_TM, M_TU, weights)
end

# The same fit from design matrices built once, so an optimizer over weights on fixed points does
# not rebuild them at every step.
function weighted_fit(M_TM::AbstractMatrix, M_TU::AbstractMatrix, weights)
    sw = sqrt.(weights)
    F = qr(sw .* M_TM)
    R = UpperTriangular(F.R)
    A = R \ (F.Q' * (sw .* M_TU))[1:size(R, 1), :]
    return (; M_TM, M_TU, R, A)
end

"""
    data_insufficiency_operator(prob, points)

The data-insufficiency operator `I − B`, with `B = M_TM⁺ M_TM`: the modeled
directions the training points cannot pin down. Zero when there are at least `p`
points in general position.
"""
function data_insufficiency_operator(prob::DesignProblem, points)
    (M_TM, _) = design_matrices(prob, points)
    B = pinv(M_TM) * M_TM
    return I(size(B, 1)) - B
end

"""
    error_operator(prob, points)
    error_operator(prob, points, weights)

The GAD error operator `E_θ = [I−B  −A; 0  I_𝒰]`, so that the prediction error is
`y_𝒫 − ŷ_𝒫 = M_𝒫 E_θ θ`.

With `weights`, the many-samples limit: `B = I`, so `E_θ = [0  −A[ρ]; 0  I_𝒰]` with
`A[ρ]` the weighted aliasing operator.
"""
function error_operator(prob::DesignProblem, points)
    (M_TM, M_TU) = design_matrices(prob, points)
    P = pinv(M_TM)
    B = P * M_TM
    A = P * M_TU
    p, q = size(A)
    return [I(p) - B  -A; zeros(q, p)  I(q)]
end

function error_operator(prob::DesignProblem, points, weights)
    A = aliasing_operator(prob, points, weights)
    p, q = size(A)
    return [zeros(p, p)  -A; zeros(q, p)  I(q)]
end
