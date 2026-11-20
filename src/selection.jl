# Fast finite-n costs for selecting rows from a pool: greedy forward selection and a fast exchange.
#
# For a training set S with full column rank (n ≥ p, so B = I), every Frobenius cost is a function
# of three p × p matrices built from the pool's rows x_i (modeled features) and u_i (unmodeled):
#
#     H  = Σ_{i∈S} x_i x_iᵀ                    (= M_TMᵀ M_TM)
#     P₁ = Σ_{i∈S} x_i a_iᵀ,   a_i = W u_i,     W = R_PMᵀ R_PU Σ_𝒰          (Risk only)
#     P₂ = Σ_{i,j∈S} L_ij x_i x_jᵀ,  L = U Σ_𝒰 Uᵀ  (pool × pool)
#
# With A = H⁻¹ M_TMᵀ M_TU and G = R_PMᵀ R_PM:
#
#     Aliasing:  ‖A Σ_𝒰^{1/2}‖²_F = tr(H⁻¹ P₂ H⁻¹)                          V = n tr(H⁻¹)
#     Risk:      ‖R_P E_θ Σ^{1/2}‖²_F / n_𝒫 = (c₀ − 2 tr(H⁻¹P₁) + tr(H⁻¹GH⁻¹P₂)) / n_𝒫,
#                c₀ = ‖R_PU Σ_𝒰^{1/2}‖²_F                                     V = n tr(G H⁻¹) / n_𝒫
#
# Adding or swapping one row changes H, P₁, P₂ by rank ≤ 6 terms, so a trial costs O(p³) for the
# factorization instead of O(n p (p + q)) for a refit: with q ≫ p (2,000+ unmodeled clusters) that
# is a large saving. Forming H squares the condition number, so trials only rank swaps; every
# accepted step is checked with the exact `cost`.

struct PoolCache{C}
    c::C                       # the cost (Aliasing, Risk, or WithNoise of either; Frobenius)
    X::Matrix{Float64}         # N × p, modeled features of the pool
    Apool::Matrix{Float64}     # N × p, rows a_iᵀ (Risk), empty for Aliasing
    L::Matrix{Float64}         # N × N, U Σ_𝒰 Uᵀ
    G::Matrix{Float64}         # p × p, R_PMᵀ R_PM (Risk), empty for Aliasing
    c0::Float64                # ‖R_PU Σ_𝒰^{1/2}‖²_F (Risk)
    n_P::Int
end

base_cost(c::Union{Aliasing,Risk}) = c
base_cost(c::WithNoise) = c.base
noise_level(c::Union{Aliasing,Risk}) = 0.0
noise_level(c::WithNoise) = c.τ

"Whether the fast selection path supports cost `c` (Frobenius-norm costs only)."
fast_supported(c) = base_cost(c).norm isa Frobenius

function PoolCache(c::DesignCost, prob::DesignProblem, pool)
    fast_supported(c) || throw(ArgumentError("fast selection needs a Frobenius-norm cost, got $c"))
    (M, U) = design_matrices(prob, pool)
    p = prob.p
    σ²U = prob.prior[p+1:end]
    L = (U .* σ²U') * U'
    if base_cost(c) isa Risk
        (R_PM, R_PU) = prediction_matrices(prob)
        W = (R_PM' * view(R_PU, 1:size(R_PM, 1), :)) .* σ²U'      # p × q
        Apool = U * W'
        G = Matrix(R_PM' * R_PM)
        c0 = sum(abs2, R_PU .* sqrt.(σ²U'))
    else
        Apool, G, c0 = zeros(0, 0), zeros(0, 0), 0.0
    end
    return PoolCache(c, Matrix{Float64}(M), Apool, L, G, c0, prob.n_P)
end

# The cost from H, P₁, P₂ at n points; Inf if H is not positive definite (rank-deficient set).
function fast_cost(pc::PoolCache, H, P1, P2, n)
    F = cholesky(Symmetric(H); check = false)
    issuccess(F) || return Inf
    Hi = inv(F)
    τ = noise_level(pc.c)
    if base_cost(pc.c) isa Aliasing
        return sum((Hi * P2) .* Hi) + τ * n * tr(Hi)
    else
        HiG = Hi * pc.G
        S = (pc.c0 - 2 * sum(Hi .* P1') + sum((HiG * Hi) .* P2)) / pc.n_P
        return S + τ * n * tr(HiG) / pc.n_P
    end
end

# H, P₁, P₂ and B = L[:, S] X_S (row k: b_kᵀ = Σ_{i∈S} L_ki x_iᵀ) for the set S.
function set_matrices(pc::PoolCache, S)
    XS = pc.X[S, :]
    H = XS' * XS
    P1 = isempty(pc.Apool) ? zeros(0, 0) : XS' * pc.Apool[S, :]
    B = pc.L[:, S] * XS
    P2 = XS' * B[S, :]
    return H, P1, P2, B
end

# Rank-one and rank-two updates: add j to S; swap o ∈ S for j ∉ S.
function added(pc::PoolCache, H, P1, P2, B, j)
    x = view(pc.X, j, :)
    b = view(B, j, :)
    H′ = H + x * x'
    P1′ = isempty(P1) ? P1 : P1 + x * view(pc.Apool, j, :)'
    P2′ = P2 + x * b' + b * x' + pc.L[j, j] * (x * x')
    return H′, P1′, P2′
end

function swapped(pc::PoolCache, H, P1, P2, B, o, j)
    xo, xj = view(pc.X, o, :), view(pc.X, j, :)
    bo = view(B, o, :)
    c = view(B, j, :) - pc.L[j, o] * xo                      # Σ_{i∈S∖o} L_ji x_i
    H′ = H - xo * xo' + xj * xj'
    P1′ = isempty(P1) ? P1 : P1 - xo * view(pc.Apool, o, :)' + xj * view(pc.Apool, j, :)'
    P2′ = P2 - xo * bo' - bo * xo' + pc.L[o, o] * (xo * xo') + xj * c' + c * xj' + pc.L[j, j] * (xj * xj')
    return H′, P1′, P2′
end

"""
    greedy_select(c, prob, pool, n; start = nothing) -> (; selected, path)

Greedy forward selection of `n` of the `pool` points for the finite-`n` cost `c` (a Frobenius
`Aliasing` or `Risk`, optionally `WithNoise`). Starts from `start` (indices into `pool`) or, by
default, from `p` points chosen by column-pivoted QR of the modeled features (a well-conditioned
square set), then adds, one at a time, the point that lowers the cost most. `path` lists the
points in the order added, so `sort(path[1:m])` is the greedy set of every size `m` (from the
start's size up to `n`) and one call serves a whole budget sweep.
"""
function greedy_select(c::DesignCost, prob::DesignProblem, pool, n; start = nothing)
    N, p = length(pool), prob.p
    n <= N || throw(ArgumentError("cannot select $n of $N points"))
    pc = PoolCache(c, prob, pool)
    S = start === nothing ? qr(pc.X', ColumnNorm()).p[1:min(p, n)] : collect(start)
    allunique(S) || throw(ArgumentError("start indices must be distinct"))
    path = copy(S)
    length(S) >= n && return (; selected = sort(S[1:n]), path = path[1:n])
    length(S) >= p || throw(ArgumentError("greedy selection needs a start of at least p = $p points"))
    H, P1, P2, B = set_matrices(pc, S)
    inS = falses(N)
    inS[S] .= true
    while length(S) < n
        outside = findall(!, inS)
        trial = fill(Inf, length(outside))
        Threads.@threads for k in eachindex(outside)
            trial[k] = fast_cost(pc, added(pc, H, P1, P2, B, outside[k])..., length(S) + 1)
        end
        j = outside[argmin(trial)]
        H, P1, P2 = added(pc, H, P1, P2, B, j)
        B .+= view(pc.L, :, j) * view(pc.X, j, :)'
        push!(S, j)
        push!(path, j)
        inS[j] = true
    end
    return (; selected = sort(S), path)
end

# Fast exchange (called by `exchange(...; fast = true)`).
function fast_exchange(c::DesignCost, prob::DesignProblem, pool, selected; candidates, maxiter, verify = 3)
    N, n = length(pool), length(selected)
    n >= prob.p || throw(ArgumentError("fast exchange needs n ≥ p"))
    pc = PoolCache(c, prob, pool)
    S = sort(collect(selected))
    H, P1, P2, B = set_matrices(pc, S)
    S_exact = cost(c, prob, pool[S])
    swaps = 0
    for _ in 1:maxiter
        outside = setdiff(1:N, S)
        isempty(outside) && break
        w = zeros(N)
        w[S] .= 1 / n
        g = first_variation(c, prob, pool, w)
        tries = outside[partialsortperm(g[outside], 1:min(candidates, length(outside)))]
        pairs = [(k, j) for k in eachindex(S) for j in tries]
        trial = fill(Inf, length(pairs))
        Threads.@threads for t in eachindex(pairs)
            (k, j) = pairs[t]
            trial[t] = fast_cost(pc, swapped(pc, H, P1, P2, B, S[k], j)..., n)
        end
        accepted = false
        for t in partialsortperm(trial, 1:min(verify, length(trial)))
            isfinite(trial[t]) || break
            (k, j) = pairs[t]
            S′ = sort(replace(S, S[k] => j))
            S′_exact = cost(c, prob, pool[S′])
            if S′_exact < S_exact
                S, S_exact = S′, S′_exact
                H, P1, P2, B = set_matrices(pc, S)          # rebuilt, so no rounding drift
                swaps += 1
                accepted = true
                break
            end
        end
        accepted || break
    end
    return (; selected = S, cost = S_exact, swaps)
end
