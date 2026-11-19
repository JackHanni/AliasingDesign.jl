# Steps 3 and 4: turn optimized weights into n points, then polish the points at finite n.

"""
    sample_points(points, weights, n; domain = (first(points), last(points)))

Step 3 for a 1D grid, eq. (25): `n` evenly spaced quantiles `F⁻¹((i − ½)/n)` of the
distribution with `weights` on the sorted grid `points`. Each weight is spread uniformly over
its grid cell (from the midpoints to its neighbors, and to the `domain` ends for the end
points), so the quantiles fall between grid points rather than on them.
"""
function sample_points(points, weights, n; domain = (first(points), last(points)))
    issorted(points) || throw(ArgumentError("points must be sorted"))
    length(points) == length(weights) ||
        throw(ArgumentError("got $(length(points)) points and $(length(weights)) weights"))
    edges = [domain[1]; (points[1:end-1] .+ points[2:end]) ./ 2; domain[2]]
    cdf = [0.0; cumsum(weights ./ sum(weights))]
    samples = zeros(n)
    for i in 1:n
        u = (i - 1/2) / n
        k = clamp(searchsortedlast(cdf, u), 1, length(weights))   # cell k holds cdf[k] .. cdf[k+1]
        mass = cdf[k+1] - cdf[k]
        frac = mass > 0 ? (u - cdf[k]) / mass : 1/2
        samples[i] = edges[k] + frac * (edges[k+1] - edges[k])
    end
    return samples
end

"""
    select_points(weights, n)

Step 3 for a pool: the indices of the `n` largest weights, in increasing order.
"""
function select_points(weights, n)
    n <= length(weights) || throw(ArgumentError("cannot select $n of $(length(weights)) points"))
    return sort(partialsortperm(weights, 1:n; rev = true))
end

"""
    optimize_points(c, prob, points; domain = nothing, maxiter = 10_000, rtol = 1e-8,
                    atol = 1e-14, η = 0.1)
        -> (; points, cost, gradient, iterations, converged)

Step 4, eq. (26): minimize the exact finite-`n` cost by gradient descent on the points (the
particle form of the Wasserstein flow), starting from `points`.

- `domain = (lo, hi)` keeps every point inside `[lo, hi]` (projected gradient); `nothing` lets
  points move freely.
- Step size: the first step moves the fastest point by `η` times the mean spacing; afterwards
  the step grows by 1.5× after an accepted step and halves when a step would raise the cost.
- Converged when the cost is below `atol`, or when the projected gradient is small relative to
  the cost: `max|gᵢ| · L ≤ rtol · S`, with `L` the mean spacing of the points.

`gradient` is the projected gradient at the final points: components that push a point at a
domain end further outward are zero, since those are optimal there.
"""
function optimize_points(c::DesignCost, prob::DesignProblem, points; domain = nothing,
                         maxiter = 10_000, rtol = 1e-8, atol = 1e-14, η = 0.1)
    points = float.(collect(points))
    domain === nothing || (points = clamp.(points, domain...))
    n = length(points)
    L = max(maximum(points) - minimum(points), eps()) / n          # mean spacing
    S = cost(c, prob, points)
    g = projected_gradient(cost_gradient(c, prob, points), points, domain)
    step = η * L / max(maximum(abs, g), eps())
    iterations, converged = 0, false
    for iter in 1:maxiter
        if S <= atol || maximum(abs, g) * L <= rtol * S
            converged = true
            break
        end
        while true
            trial = points .- step .* g
            domain === nothing || (trial = clamp.(trial, domain...))
            S_trial = cost(c, prob, trial)
            if S_trial <= S || step < 1e-15 * L / max(maximum(abs, g), eps())
                points, S = trial, S_trial
                step *= 1.5
                break
            end
            step /= 2
        end
        g = projected_gradient(cost_gradient(c, prob, points), points, domain)
        iterations = iter
    end
    converged = converged || S <= atol || maximum(abs, g) * L <= rtol * S
    return (; points, cost = S, gradient = g, iterations, converged)
end

function projected_gradient(g, points, domain)
    domain === nothing && return g
    lo, hi = domain
    return [(t <= lo && gi > 0) || (t >= hi && gi < 0) ? 0.0 : gi for (gi, t) in zip(g, points)]
end

"""
    exchange(c, prob, pool, selected; candidates = 20, maxiter = 1_000)
        -> (; selected, cost, swaps)

Step 4 for a pool (Fedorov exchange): starting from the indices `selected` into `pool`,
repeatedly make the single swap (one selected structure out, one unselected in) that lowers the
exact finite-`n` cost the most, until no swap helps or `maxiter` swaps are made.

Only the `candidates` unselected structures with the lowest first variation (computed at
weight `1/n` on the selection) are tried as additions; `candidates = length(pool)` tries every
swap, so the result is then a local optimum over all single swaps. Needs `n ≥ p` for the
ranking.
"""
function exchange(c::DesignCost, prob::DesignProblem, pool, selected; candidates = 20, maxiter = 1_000)
    selected = sort(collect(selected))
    n, N = length(selected), length(pool)
    allunique(selected) || throw(ArgumentError("selected indices must be distinct"))
    S = cost(c, prob, pool[selected])
    swaps = 0
    for _ in 1:maxiter
        outside = setdiff(1:N, selected)
        isempty(outside) && break
        w = zeros(N)
        w[selected] .= 1 / n
        g = first_variation(c, prob, pool, w)
        tries = outside[partialsortperm(g[outside], 1:min(candidates, length(outside)))]
        best = (S, 0, 0)
        for (k, i) in enumerate(selected), j in tries
            trial = copy(selected)
            trial[k] = j
            S_trial = cost(c, prob, pool[trial])
            S_trial < best[1] && (best = (S_trial, k, j))
        end
        best[2] == 0 && break
        S = best[1]
        selected[best[2]] = best[3]
        sort!(selected)
        swaps += 1
    end
    return (; selected, cost = S, swaps)
end
