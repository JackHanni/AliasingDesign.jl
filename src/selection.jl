# Fast finite-n costs for selecting rows from a pool: greedy forward selection and a fast exchange.
# Both regimes are covered, with the minimum-norm fit θ̂_M = M_TM⁺ y throughout (as in `cost`):
#
# n ≥ p (more rows than columns; full column rank, B = I). Every Frobenius cost is a function of
# three p × p matrices built from the pool's rows x_i (modeled features) and u_i (unmodeled):
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
# Adding or swapping one row changes H, P₁, P₂ by rank ≤ 6 terms, so a trial costs O(p³).
#
# n < p (more columns than rows; full row rank). M⁺ = Mᵀ K⁻¹ with K = M Mᵀ (n × n), B = Mᵀ K⁻¹ M ≠ I,
# and every cost is a trace of n × n blocks [·]_SS of fixed pool × pool matrices:
#
#     Aliasing:  tr(K⁻¹ L_SS)                                                   V = n tr(K⁻¹)
#     Risk:      (tr(G Σ_M) + c₀ − 2 tr(K⁻¹ T_SS) + tr(K⁻¹ Φ_SS K⁻¹ E_SS)) / n_𝒫
#                                                                              V = n tr(K⁻² E_SS) / n_𝒫
#
# with Φ = X Σ_M Xᵀ + U Σ_𝒰 Uᵀ (the prior kernel of all p + q functions), E = X G Xᵀ and
# T = sym(X Σ_M G Xᵀ + X Apoolᵀ), sym(Y) = (Y + Yᵀ)/2. The data-insufficiency term ‖R_PM (I − B) Σ_M^{1/2}‖²
# contributes tr(G Σ_M), the X Σ_M G Xᵀ part of T and the X Σ_M Xᵀ part of Φ.
# A trial costs O(n³) < O(p³).
#
# Either way the cost of a trial does not grow with q, the number of unmodeled functions, where a
# refit costs O(n p (p + q)). Forming H or K squares the condition number, so trials only rank
# candidates; every accepted exchange step is checked with the exact `cost`.

struct PoolCache{C}
    c::C                       # the cost (Aliasing, Risk, or WithNoise of either; Frobenius)
    p::Int
    X::Matrix{Float64}         # N × p, modeled features of the pool
    Apool::Matrix{Float64}     # N × p, rows a_iᵀ (Risk), empty for Aliasing
    L::Matrix{Float64}         # N × N, U Σ_𝒰 Uᵀ
    G::Matrix{Float64}         # p × p, R_PMᵀ R_PM (Risk), empty for Aliasing
    c0::Float64                # ‖R_PU Σ_𝒰^{1/2}‖²_F (Risk)
    n_P::Int
    # n < p (built on first use):
    XX::Matrix{Float64}        # N × N, X Xᵀ
    Φ::Matrix{Float64}         # N × N, X Σ_M Xᵀ + L (Risk)
    E::Matrix{Float64}         # N × N, X G Xᵀ (Risk)
    T::Matrix{Float64}         # N × N, sym(X Σ_M G Xᵀ + X Apoolᵀ) (Risk)
    trGΣ::Float64              # tr(G Σ_M) (Risk)
end

base_cost(c::Union{Aliasing,Risk}) = c
base_cost(c::WithNoise) = c.base
noise_level(c::Union{Aliasing,Risk}) = 0.0
noise_level(c::WithNoise) = c.τ

"Whether the fast selection path supports cost `c` (Frobenius-norm costs only)."
fast_supported(c) = base_cost(c).norm isa Frobenius

sym(A) = (A + A') / 2

function PoolCache(c::DesignCost, prob::DesignProblem, pool; underdetermined = true)
    fast_supported(c) || throw(ArgumentError("fast selection needs a Frobenius-norm cost, got $c"))
    (M, U) = design_matrices(prob, pool)
    X = Matrix{Float64}(M)
    p = prob.p
    σ²M, σ²U = prob.prior[1:p], prob.prior[p+1:end]
    L = (U .* σ²U') * U'
    e = zeros(0, 0)
    if base_cost(c) isa Risk
        (R_PM, R_PU) = prediction_matrices(prob)
        W = (R_PM' * view(R_PU, 1:size(R_PM, 1), :)) .* σ²U'      # p × q
        Apool = U * W'
        G = Matrix(R_PM' * R_PM)
        c0 = sum(abs2, R_PU .* sqrt.(σ²U'))
    else
        Apool, G, c0 = e, e, 0.0
    end
    XX, Φ, E, T, trGΣ = e, e, e, e, 0.0
    if underdetermined
        XX = X * X'
        if base_cost(c) isa Risk
            XΣ = X .* σ²M'
            Φ = XΣ * X' + L
            E = X * G * X'
            T = sym(XΣ * G * X' + X * Apool')
            trGΣ = sum(diag(G) .* σ²M)
        end
    end
    return PoolCache(c, p, X, Apool, L, G, c0, prob.n_P, XX, Φ, E, T, trGΣ)
end

# n ≥ p: the cost from H, P₁, P₂ at n points; Inf if H is not positive definite.
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

# n < p: the cost of the set S from n × n blocks; Inf if K = M Mᵀ is not positive definite.
function row_cost(pc::PoolCache, S)
    n = length(S)
    F = cholesky(Symmetric(pc.XX[S, S]); check = false)
    issuccess(F) || return Inf
    Ki = inv(F)
    τ = noise_level(pc.c)
    if base_cost(pc.c) isa Aliasing
        return sum(Ki .* view(pc.L, S, S)) + τ * n * tr(Ki)
    else
        ES = pc.E[S, S]
        S_risk = (pc.trGΣ + pc.c0 - 2 * sum(Ki .* view(pc.T, S, S)) + sum((Ki * pc.Φ[S, S] * Ki) .* ES)) / pc.n_P
        return S_risk + τ * n * sum((Ki * Ki) .* ES) / pc.n_P
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

# Threaded trial loops, as separate functions so that their closures capture only arguments
# (closures over reassigned locals are boxed, which made compilation very slow).
function trials_row_add(pc, S, outside)
    out = fill(Inf, length(outside))
    Threads.@threads for k in eachindex(outside)
        out[k] = row_cost(pc, [S; outside[k]])
    end
    return out
end

function trials_add(pc, H, P1, P2, B, outside, n)
    out = fill(Inf, length(outside))
    Threads.@threads for k in eachindex(outside)
        out[k] = fast_cost(pc, added(pc, H, P1, P2, B, outside[k])..., n)
    end
    return out
end

function trials_row_swap(pc, S, pairs)
    out = fill(Inf, length(pairs))
    Threads.@threads for t in eachindex(pairs)
        (k, j) = pairs[t]
        out[t] = row_cost(pc, replace(S, S[k] => j))
    end
    return out
end

function trials_swap(pc, H, P1, P2, B, S, pairs)
    out = fill(Inf, length(pairs))
    Threads.@threads for t in eachindex(pairs)
        (k, j) = pairs[t]
        out[t] = fast_cost(pc, swapped(pc, H, P1, P2, B, S[k], j)..., length(S))
    end
    return out
end

"""
    greedy_select(c, prob, pool, n; start = Int[]) -> (; selected, path)

Greedy forward selection of `n` of the `pool` points for the finite-`n` cost `c` (a Frobenius
`Aliasing` or `Risk`, optionally `WithNoise`), with the minimum-norm fit: from `start` (indices
into `pool`, empty by default), add, one point at a time, the point that lowers the cost most.
Works for any `n`, below and above `p`. `path` lists the points in the order added, so
`sort(path[1:m])` is the greedy set of every size `m` and one call serves a whole budget sweep.
"""
function greedy_select(c::DesignCost, prob::DesignProblem, pool, n; start = Int[])
    N, p = length(pool), prob.p
    n <= N || throw(ArgumentError("cannot select $n of $N points"))
    S = collect(Int, start)
    allunique(S) || throw(ArgumentError("start indices must be distinct"))
    length(S) >= n && return (; selected = sort(S[1:n]), path = S[1:n])
    pc = PoolCache(c, prob, pool; underdetermined = length(S) < p)
    path = copy(S)
    inS = falses(N)
    inS[S] .= true
    # Below p: each trial is the n × n block cost of S ∪ {j}.
    while length(S) < min(n, p)
        outside = findall(!, inS)
        trial = trials_row_add(pc, S, outside)
        j = outside[argmin(trial)]
        push!(S, j); push!(path, j); inS[j] = true
    end
    length(S) >= n && return (; selected = sort(S), path)
    # From p on: rank-one updates of the p × p matrices.
    H, P1, P2, B = set_matrices(pc, S)
    while length(S) < n
        outside = findall(!, inS)
        trial = trials_add(pc, H, P1, P2, B, outside, length(S) + 1)
        j = outside[argmin(trial)]
        H, P1, P2 = added(pc, H, P1, P2, B, j)
        B .+= view(pc.L, :, j) * view(pc.X, j, :)'
        push!(S, j); push!(path, j); inS[j] = true
    end
    return (; selected = sort(S), path)
end

# Fast exchange (called by `exchange(...; fast = true)`). Additions to try: for n ≥ p, the
# `candidates` lowest first variations at weight 1/n on the selection (as in the exact exchange);
# for n < p, where that weighted fit does not exist, the `candidates` best single additions.
function fast_exchange(c::DesignCost, prob::DesignProblem, pool, selected; candidates, maxiter, verify = 3)
    N, n, p = length(pool), length(selected), prob.p
    allunique(selected) || throw(ArgumentError("selected indices must be distinct"))
    under = n < p
    pc = PoolCache(c, prob, pool; underdetermined = under)
    S = sort(collect(selected))
    mats = under ? nothing : set_matrices(pc, S)
    S_exact = cost(c, prob, pool[S])
    swaps = 0
    for _ in 1:maxiter
        outside = setdiff(1:N, S)
        isempty(outside) && break
        rank_by = if under
            trials_row_add(pc, S, outside)
        else
            w = zeros(N)
            w[S] .= 1 / n
            first_variation(c, prob, pool, w)[outside]
        end
        tries = outside[partialsortperm(rank_by, 1:min(candidates, length(outside)))]
        pairs = [(k, j) for k in eachindex(S) for j in tries]
        trial = under ? trials_row_swap(pc, S, pairs) : trials_swap(pc, mats..., S, pairs)
        accepted = false
        for t in partialsortperm(trial, 1:min(verify, length(trial)))
            isfinite(trial[t]) || break
            (k, j) = pairs[t]
            S′ = sort(replace(S, S[k] => j))
            S′_exact = cost(c, prob, pool[S′])
            if S′_exact < S_exact
                S, S_exact = S′, S′_exact
                under || (mats = set_matrices(pc, S))     # rebuilt, so no rounding drift
                swaps += 1
                accepted = true
                break
            end
        end
        accepted || break
    end
    return (; selected = S, cost = S_exact, swaps)
end
