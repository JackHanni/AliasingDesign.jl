# Finite-n gradient ∂S/∂tᵢ for the particle flow (step 4).
#
# Every cost is S = ‖X‖² with X built from M = M_TM, N = M_TU and P = M⁺, so
#     dS = 2⟨K, dX⟩ = 2(⟨C_A, dA⟩ + ⟨C_B, dB⟩ + ⟨W_V, dP⟩),
# with K = get_K(norm, X) and cost-specific coefficient matrices C_A (p×q), C_B (p×p), W_V (p×n)
# (`gradient_coefficients`). A = PN and B = PM, so dA = dP N + P dN and dB = dP M + P dM.
# For dP we use the derivative of the pseudoinverse at constant rank (Golub & Pereyra, 1973):
#     dP = −P dM P + P Pᵀ dMᵀ (I − M P) + (I − P M) dMᵀ Pᵀ P.
# Collecting every ⟨·, dP⟩ into W = C_A Nᵀ + C_B Mᵀ + W_V and moving dM to one side gives the
# matrix gradients ∂S/∂M and ∂S/∂N (`pinv_adjoint`). Moving point tᵢ changes only row i of M and
# N, by the basis derivatives φ′(tᵢ), so ∂S/∂tᵢ is row i of each gradient dotted with them.
# The constant-rank assumption holds for points in general position (rank min(n, p)).

"""
    cost_gradient(c, prob, points)

Gradient of `cost(c, prob, points)` with respect to each point `tᵢ`: the exact finite-`n`
cost, including the data-insufficiency term `I − B` for `n < p`. Drives the particle flow
(step 4, eq. (26)). Points must be real numbers, and the basis functions must be
differentiable by ForwardDiff. Valid where `rank M_TM` is constant, i.e. away from designs
with repeated points or other rank drops.
"""
function cost_gradient(c::DesignCost, prob::DesignProblem, points)
    (M, N) = design_matrices(prob, points)
    P = pinv(M)
    C_A, C_B, W_V = gradient_coefficients(c, prob, points, M, N, P)
    ∂S_∂M, ∂S_∂N = pinv_adjoint(M, N, P, C_A, C_B, W_V)
    D_M = basis_derivatives(prob.basis[1:prob.p], points)
    D_U = basis_derivatives(prob.basis[prob.p+1:end], points)
    return 2 .* (vec(sum(∂S_∂M .* D_M; dims = 2)) .+ vec(sum(∂S_∂N .* D_U; dims = 2)))
end

"""
    basis_derivatives(basis, points)

`D[i, j] = basis[j]′(points[i])`, by forward-mode automatic differentiation.
"""
function basis_derivatives(basis, points)
    isempty(basis) && return zeros(length(points), 0)
    return [float(ForwardDiff.derivative(f, float(t))) for t in points, f in basis]
end

# Half the matrix gradients of S with respect to M = M_TM and N = M_TU, given the coefficients
# of dA, dB and dP (see the comment at the top of this file).
function pinv_adjoint(M, N, P, C_A, C_B, W_V)
    W = C_A * N' + C_B * M' + W_V
    ∂S_∂M = -P' * W * P' + (I - M * P) * W' * (P * P') + (P' * P) * W' * (I - P * M) + P' * C_B
    ∂S_∂N = P' * C_A
    return ∂S_∂M, ∂S_∂N
end

"""
    gradient_coefficients(c, prob, points, M, N, P) -> (C_A, C_B, W_V)

Coefficients of `dA`, `dB` and `dP` in `dS / 2`, for the cost `c` at finite points.
"""
function gradient_coefficients(c::Aliasing, prob::DesignProblem, points, M, N, P)
    D_U = prior_sqrt_U(prob)
    K = get_K(c.norm, P * N * D_U)
    return K * D_U, zeros(prob.p, prob.p), zeros(size(P))
end

# X = s M_P E_θ Σ^{1/2} with dE_θ = [−dB  −dA; 0  0], so
# ⟨K, dX⟩ = −s⟨M_PMᵀ K_M Σ_M^{1/2}, dB⟩ − s⟨M_PMᵀ K_U Σ_U^{1/2}, dA⟩.
function gradient_coefficients(c::Risk, prob::DesignProblem, points, M, N, P)
    (M_PM, _) = prediction_matrices(prob)    # compressed R_PM: same inner products as M_PM
    p = prob.p
    s = 1 / sqrt(prob.n_P)
    K = get_K(c.norm, cost_op(c, prob, points))
    D = prior_sqrt(prob)
    C_B = -s * M_PM' * K[:, 1:p] * D[1:p, 1:p]
    C_A = -s * M_PM' * K[:, p+1:end] * D[p+1:end, p+1:end]
    return C_A, C_B, zeros(size(P))
end

# The noise term enters through P only: V_alias = n‖P‖²_F and V_risk = (n/n_𝒫)‖M_PM P‖²_F, so
# τ dV = 2⟨τ n P, dP⟩ and 2⟨τ (n/n_𝒫) M_PMᵀ M_PM P, dP⟩.
function gradient_coefficients(c::WithNoise, prob::DesignProblem, points, M, N, P)
    C_A, C_B, W_V = gradient_coefficients(c.base, prob, points, M, N, P)
    return C_A, C_B, W_V .+ c.τ .* noise_dP_coefficient(c.base, prob, points, P)
end

noise_dP_coefficient(::Aliasing, prob::DesignProblem, points, P) = length(points) .* P

function noise_dP_coefficient(::Risk, prob::DesignProblem, points, P)
    (M_PM, _) = prediction_matrices(prob)
    return (length(points) / prob.n_P) .* (M_PM' * (M_PM * P))
end
