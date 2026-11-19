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
# With real data, the "true" model uses the first K_TRUE clusters in the natural ordering (all
# pairs and triplets and most 4-body clusters for the JuCE pool); larger clusters have tiny prior
# variance and would only make every cost evaluation slower.
const K_TRUE = parse(Int, get(ENV, "K_TRUE", "300"))
# Each Fisher–Rao iteration on the 2,346-structure pool takes ~0.1 s, so the pool flow runs fewer
# iterations than the 1D problems; top-n rounding and exchange polish the result.
const POOL_MAXITER = FULL ? 5_000 : 500

# ---------------------------------------------------------------- the pool
include("ce_pool.jl")

const X, SIZES, SOURCE = load_pool(K_TRUE)
const POOL = collect(1:size(X, 1))                  # a "point" is a structure's row index
const BASIS = pool_basis(X)
const PRIOR = cluster_prior(SIZES)

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
# Each p is independent: one thread per p.
rows = tmap(ps) do p
    prob = problem(p)
    c = WithNoise(Risk(), 1e-6 / n)
    opt = optimized_design(c, prob, n)
    r_rand = [heldout_rmse(prob, r; truths = 50) for r in randoms[1:20]]
    (p, heldout_rmse(prob, opt.selected), median(r_rand), quantile(r_rand, 0.9))
end
for r in rows
    @printf("  p=%2d  optimized=%.3e  random median=%.3e  random 90%%=%.3e\n", r...)
end
write_table(joinpath(OUT, "risk_vs_p_$(SOURCE).csv"), ["p", "optimized", "random_median", "random_q90"], rows)
fig = plot(first.(rows), [r[2] for r in rows]; label = "optimized (capped FR + exchange)", marker = :circle,
           yscale = :log10, xlabel = "modeled clusters p", ylabel = "held-out RMSE",
           title = "Cluster expansion ($(SOURCE) pool), n = $n training structures", size = (800, 450))
plot!(fig, first.(rows), [r[3] for r in rows]; label = "random (median of 20)", marker = :square)
plot!(fig, first.(rows), [r[4] for r in rows]; label = "random (90th percentile)", linestyle = :dash)
save_figure(fig, joinpath(OUT, "risk_vs_p_$(SOURCE).png"))
