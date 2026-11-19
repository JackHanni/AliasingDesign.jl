# Shared helpers for the experiment scripts. Each script does `include("common.jl")`.

using AliasingDesign
using DelimitedFiles
using LinearAlgebra
using Plots
using Printf
using Random
using Statistics

# FULL=1 runs the sizes used in the plan's findings (more starts, more iterations); the default
# is a quicker run with the same structure.
const FULL = get(ENV, "FULL", "0") == "1"
const STARTS = FULL ? 6 : 4
const MAXITER = FULL ? 20_000 : 5_000

"The output folder `experiments/output/<name>`, created if needed."
function outdir(name)
    dir = joinpath(@__DIR__, "output", name)
    mkpath(dir)
    return dir
end

# A CSV field: numbers to 6 significant digits; text quoted if it contains a comma or a quote.
csv_field(x::AbstractFloat) = @sprintf("%.6g", x)
function csv_field(x)
    s = string(x)
    return any(in(",\"\n"), s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : s
end

"Write `rows` (a vector of tuples or vectors) under `header` to a CSV file, and say where."
function write_table(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join((csv_field(x) for x in row), ","))
        end
    end
    println("  wrote ", relpath(path, @__DIR__))
end

function save_figure(fig, path)
    savefig(fig, path)
    println("  wrote ", relpath(path, @__DIR__))
end

"""
    multistart(c, prob, grid; starts = STARTS, rng, kwargs...)

Run the Fisher–Rao flow from uniform weights and `starts − 1` random ones. Returns the runs
sorted by cost and the number of distinct optima (costs differing by more than 1e-4 relative).
"""
function multistart(c, prob, grid; starts = STARTS, rng = Xoshiro(1), maxiter = MAXITER, η = 8, kwargs...)
    N = length(grid)
    inits = [[fill(1 / N, N)]; [normalize(rand(rng, N) .+ 0.05, 1) for _ in 2:starts]]
    runs = [optimize_weights(c, prob, grid, w0; maxiter, η, kwargs...) for w0 in inits]
    sort!(runs; by = r -> r.cost)
    costs = [r.cost for r in runs]
    distinct = 1 + count(i -> costs[i] - costs[i-1] > 1e-4 * max(abs(costs[i]), eps()), 2:length(costs))
    return runs, distinct
end

"Grid points carrying at least `frac` of the largest weight (the design's atoms)."
support(grid, w; frac = 1e-3) = grid[w .>= frac * maximum(w)]

"The Gram matrix `G_MM` of the modeled basis under weights `w` on `grid`."
function gram(prob, grid, w)
    (M, _) = design_matrices(prob, grid)
    return M' * Diagonal(w) * M
end

"Bias (`base` cost) and variance (`τ V`) parts of a noisy cost at a density."
function bias_variance(c::WithNoise, prob, grid, w)
    return cost(c.base, prob, grid, w), c.τ * noise_variance(c.base, prob, grid, w)
end

"""
    pipeline(c, prob, grid, n; domain, starts, rng)

Steps 2–4 for a continuous 1D domain: the best Fisher–Rao weights on `grid` (several starts),
their `n` midpoint quantiles, and those points polished by the particle flow. Returns a named
tuple with the relaxed weights, the quantile points, the polished points, and their costs.
"""
function pipeline(c, prob, grid, n; domain = (first(grid), last(grid)), starts = STARTS, rng = Xoshiro(1))
    runs, _ = multistart(c, prob, grid; starts, rng)
    w = runs[1].weights
    quantile_points = sample_points(grid, w, n; domain)
    polished = optimize_points(c, prob, quantile_points; domain, maxiter = 2_000)
    return (; weights = w, relaxed_cost = runs[1].cost,
              quantile_points, quantile_cost = cost(c, prob, quantile_points),
              points = sort(polished.points), cost = polished.cost)
end

"Best of several particle-flow runs from random starts in `domain`."
function best_particle_flow(c, prob, n, domain; starts = STARTS, rng = Xoshiro(2), maxiter = 2_000)
    lo, hi = domain
    runs = [optimize_points(c, prob, sort(lo .+ (hi - lo) .* rand(rng, n)); domain, maxiter)
            for _ in 1:starts]
    best = argmin(r -> r.cost, runs)
    return (; points = sort(best.points), cost = best.cost)
end

equally_spaced(n, (lo, hi)) = n == 1 ? [(lo + hi) / 2] : collect(range(lo, hi; length = n))
chebyshev(n, (lo, hi)) = sort((lo + hi) / 2 .+ (hi - lo) / 2 .* cos.((2(1:n) .- 1) .* π ./ 2n))
gauss_legendre(n) = eigvals(SymTridiagonal(zeros(n), [j / sqrt(4j^2 - 1) for j in 1:n-1]))
