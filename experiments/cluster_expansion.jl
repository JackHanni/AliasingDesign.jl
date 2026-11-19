# Cluster expansion (plan: Experiments → Cluster expansion; Discrete domains).
#
#   julia --project=experiments experiments/cluster_expansion.jl
#
# The pool pipeline: capped Fisher–Rao flow on the candidate structures (cap 1/n, so each
# structure counts once), top-n rounding, Fedorov exchange; compared with random training sets.
#
# Data. experiments/data/export_juce.jl writes the 2,346-structure FCC binary pool from JuCE
# (run it once in JuCE's environment; see experiments/README.md). Energies are synthetic, built
# like JuCE's examples/01_fcc_binary.jl (random true ECIs, energies M θ), but with the ECIs drawn
# from the prior, θ ~ (0, Σ), rather than with unit variance, plus 1 meV noise.
# In general, put in experiments/data/:
#   correlations.csv   one row per candidate structure, one column per cluster in the natural
#                      ordering (empty, point, pairs short → long, triplets, …), no header;
#   cluster_sizes.csv  one row per cluster: "size,diameter" (any length unit; for the prior).
# Without them, the script builds a demo pool: every distinct binary configuration on periodic
# rings of 1–10 sites, with the correlation functions of a 1D cluster expansion. The demo shows
# the pipeline end to end; the results that matter come from the real 2,346-structure pool.
#
# Before using real DFT energies, split them once into calibration and validation sets (plan:
# Experiments → Cluster expansion). This script uses only correlations and synthetic truths.

include("common.jl")

const OUT = outdir("cluster_expansion")
const DATA = joinpath(@__DIR__, "data")
# With real data, the "true" model uses the first K_TRUE clusters in the natural ordering (all
# pairs and triplets and most 4-body clusters for the JuCE pool); larger clusters have tiny prior
# variance and would only make every cost evaluation slower.
const K_TRUE = parse(Int, get(ENV, "K_TRUE", "300"))
# Each Fisher–Rao iteration on the 2,346-structure pool takes ~0.1 s, so the pool flow runs fewer
# iterations than the 1D problems; top-n rounding and exchange polish the result.
const POOL_MAXITER = FULL ? 5_000 : 500

# ---------------------------------------------------------------- the pool
# Demo clusters on a ring: (sites relative to 0). Natural ordering: by size, then diameter.
const DEMO_CLUSTERS = [Int[], [0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
                       [0, 1, 2], [0, 1, 3], [0, 2, 4], [0, 1, 2, 3]]

function demo_pool()
    corr(spins, cl) = isempty(cl) ? 1.0 :
        mean(prod(spins[mod1(i + d, length(spins))] for d in cl) for i in eachindex(spins))
    rows = Set{Vector{Float64}}()
    for L in 1:10, bits in 0:(2^L - 1)
        spins = [((bits >> i) & 1) == 1 ? 1.0 : -1.0 for i in 0:L-1]
        push!(rows, [round(corr(spins, cl); digits = 12) for cl in DEMO_CLUSTERS])
    end
    X = permutedims(reduce(hcat, sort(collect(rows))))
    sizes = [(length(cl), isempty(cl) ? 0 : maximum(cl) - minimum(cl)) for cl in DEMO_CLUSTERS]
    return X, sizes
end

function load_pool()
    if isfile(joinpath(DATA, "correlations.csv"))
        X = readdlm(joinpath(DATA, "correlations.csv"), ',', Float64)
        S = readdlm(joinpath(DATA, "cluster_sizes.csv"), ',', Float64)
        K = min(K_TRUE, size(X, 2))
        println("loaded ", size(X, 1), " structures × ", size(X, 2), " clusters from experiments/data;",
                " the true model uses the first ", K, " (K_TRUE)")
        return X[:, 1:K], [(Int(S[i, 1]), S[i, 2]) for i in 1:K], "juce"
    end
    X, sizes = demo_pool()
    println("no experiments/data/correlations.csv: using the demo pool (", size(X, 1),
            " distinct ring configurations × ", size(X, 2), " clusters)")
    return X, sizes, "demo"
end

const X, SIZES, SOURCE = load_pool()
const POOL = collect(1:size(X, 1))                  # a "point" is a structure's row index
const BASIS = [t -> X[t, j] for j in axes(X, 2)]

# Prior: shrinks with cluster size and diameter (the physical intuition behind the natural
# ordering). Diameters are measured relative to the smallest one (the nearest-neighbor pair), so
# the units of the data do not matter. σ_k = 1 for the empty and point clusters.
const D_NN = minimum(d for (size, d) in SIZES if size >= 2)
prior_cluster((size, diam)) = size <= 1 ? 1.0 : 1 / (size * diam / D_NN)^2
const PRIOR = prior_cluster.(SIZES)

problem(p) = DesignProblem(BASIS; p, prior = PRIOR, predict = POOL)

"Optimized n-structure design: capped Fisher–Rao, top-n rounding, exchange. Also the relaxed bound."
function optimized_design(c, prob, n)
    relaxed = optimize_weights(c, prob, POOL, fill(1 / length(POOL), length(POOL));
                               cap = 1 / n, η = 8, maxiter = POOL_MAXITER)
    rounded = select_points(relaxed.weights, n)
    ex = exchange(c, prob, POOL, rounded; candidates = 10)
    return (; relaxed_cost = relaxed.cost, rounded, rounded_cost = cost(c, prob, POOL[rounded]),
              selected = ex.selected, cost = ex.cost, swaps = ex.swaps)
end

"""Held-out RMSE of the fit θ̂ = M⁺y on `train` (the minimum-norm least-squares fit the GAD and the
costs assume), averaged over synthetic truths θ ~ (0, Σ). `pinv` rather than `\\`: for n = p a
nearly singular square system would otherwise give an arbitrarily large fit."""
function heldout_rmse(prob, train; truths = 200, noise = 1e-3, rng = Xoshiro(11))
    M = design_matrix(prob.basis, POOL)
    test = setdiff(POOL, train)
    errs = map(1:truths) do _
        θ = sqrt.(prob.prior) .* randn(rng, length(prob.prior))
        y = M * θ .+ noise .* randn(rng, length(POOL))
        θ̂ = pinv(M[train, 1:prob.p]) * y[train]
        sqrt(mean(abs2, M[test, 1:prob.p] * θ̂ .- M[test, :] * θ))
    end
    return mean(errs)
end

# ---------------------------------------------------------------- designs at fixed p, n
p, n = 6, 10
prob = problem(p)
c = WithNoise(Risk(), 1e-6 / n)
println("\nDesigns at p = $p modeled clusters, n = $n structures (cost: $c)")
opt = optimized_design(c, prob, n)
rng = Xoshiro(12)
randoms = [sort(randperm(rng, length(POOL))[1:n]) for _ in 1:50]
rand_costs = [cost(c, prob, POOL[r]) for r in randoms]
median_random = randoms[sortperm(rand_costs)[25]]
rows = [
    ("relaxed bound (capped, continuous weights)", opt.relaxed_cost, NaN, NaN, ""),
    ("rounded (top-n weights)", opt.rounded_cost, cost(Aliasing(), prob, POOL[opt.rounded]),
     heldout_rmse(prob, opt.rounded), join(opt.rounded, " ")),
    ("rounded + exchange ($(opt.swaps) swaps)", opt.cost, cost(Aliasing(), prob, POOL[opt.selected]),
     heldout_rmse(prob, opt.selected), join(opt.selected, " ")),
    ("random (median of 50)", median(rand_costs), cost(Aliasing(), prob, POOL[median_random]),
     heldout_rmse(prob, median_random), join(median_random, " ")),
]
for r in rows
    @printf("  %-44s cost=%.3e  aliasing=%.3e  held-out RMSE=%.3e\n", r[1:4]...)
end
@printf("  gap (rounded + exchange) − (relaxed bound) = %.3e  (bounds the distance to the best design\n", opt.cost - opt.relaxed_cost)
println("   if the relaxed minimum is global; S is not convex, so treat it as empirical)")
write_table(joinpath(OUT, "designs_$(SOURCE).csv"), ["design", "cost", "aliasing", "heldout_rmse", "structures"], rows)

# ---------------------------------------------------------------- risk vs. p (cf. GAD paper Fig. 6)
println("\nRisk vs. number of modeled clusters p, n = $n")
ps = 2:min(size(X, 2) - 1, n)
rows = []
for p in ps
    local prob = problem(p)
    local c = WithNoise(Risk(), 1e-6 / n)
    local opt = optimized_design(c, prob, n)
    r_rand = [heldout_rmse(prob, r; truths = 50) for r in randoms[1:20]]
    push!(rows, (p, heldout_rmse(prob, opt.selected), median(r_rand), quantile(r_rand, 0.9)))
    @printf("  p=%2d  optimized=%.3e  random median=%.3e  random 90%%=%.3e\n", rows[end]...)
end
write_table(joinpath(OUT, "risk_vs_p_$(SOURCE).csv"), ["p", "optimized", "random_median", "random_q90"], rows)
fig = plot(first.(rows), [r[2] for r in rows]; label = "optimized (capped FR + exchange)", marker = :circle,
           yscale = :log10, xlabel = "modeled clusters p", ylabel = "held-out RMSE",
           title = "Cluster expansion ($(SOURCE) pool), n = $n training structures", size = (800, 450))
plot!(fig, first.(rows), [r[3] for r in rows]; label = "random (median of 20)", marker = :square)
plot!(fig, first.(rows), [r[4] for r in rows]; label = "random (90th percentile)", linestyle = :dash)
save_figure(fig, joinpath(OUT, "risk_vs_p_$(SOURCE).png"))
