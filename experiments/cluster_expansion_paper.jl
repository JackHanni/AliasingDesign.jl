# Cluster expansion at the GAD paper's scale (plan: Experiments → Cluster expansion; cf. Fig. 6 of
# the GAD paper).
#
#   julia -t auto --project=experiments experiments/cluster_expansion_paper.jl
#
# Fixed training sets of N_TRAIN = 600 structures from the 2,346-structure JuCE pool; the true model
# uses all of the pool's clusters (2,344). For each training set and each number of modeled clusters
# p (1 up to K − 1), two risks on the held-out structures (the rest of the pool):
#   expected risk  the GAD's prediction from the design alone (prior-averaged, no labels);
#   held-out risk  measured: fit synthetic energies on the training set, mean squared error on the
#                  held-out structures, averaged over N_TRUTHS synthetic truths. TRUTH=prior draws
#                  the truths from the design's prior; TRUTH=unit uses unit-variance ECIs (JuCE's
#                  synthetic setup), a truth the prior did not anticipate.
# and the parameter-space counterpart: expected aliasing (from the design) and parameter aliasing
# (‖θ̂_M − B θ_M‖² of the actual fits, averaged over the truths), and the paper's aliasing norm
# ‖A‖₂ for comparison with its figures. Training sets:
#   natural    the first N_TRAIN structures in the enumeration's natural ordering (smallest
#              supercells first) — the conventional choice;
#   random     N_RANDOM random sets of N_TRAIN structures (mean ± std over draws);
#   optimized  one set per target p in P_TARGETS: capped Fisher–Rao flow (cap 1/N_TRAIN, noise term
#              τ = 1e-6/N_TRAIN) on the pool at that p, then top-n rounding. No exchange step: at
#              n = 600 each exchange sweep would try ~6,000 swaps, each a full cost evaluation.
#
# Cost (rough, 8 threads): each risk evaluation is a pseudoinverse of a 600 × p matrix plus
# products of size ~K·p·(K − p), up to ~0.5 s on one core. The default p grid (~400 values, dense
# near p = N_TRAIN) × 13 training sets is ~5,000 evaluations, ~10–15 min; P_GRID=all (2,343 values)
# is ~30,000 evaluations, ~1–1.5 h. (Each evaluation computes both risks and the aliasing.) Each optimized design adds a few minutes (POOL_MAXITER
# Fisher–Rao iterations at ~0.5 s each).
#
# Settings (environment variables): N_TRAIN (600), N_RANDOM (10), P_TARGETS ("100,300"),
# POOL_MAXITER (300; 2000 with FULL=1), P_GRID ("default" or "all"), N_TRUTHS (50), TRUTH ("prior"
# or "unit"), NOISE (1e-3, the noise standard deviation per energy).

include("common.jl")
include("ce_pool.jl")

const OUT = outdir("cluster_expansion_paper")
const N_TRAIN = parse(Int, get(ENV, "N_TRAIN", "600"))
const N_RANDOM = parse(Int, get(ENV, "N_RANDOM", "10"))
const P_TARGETS = parse.(Int, split(get(ENV, "P_TARGETS", "100,300"), ","))
const POOL_MAXITER = parse(Int, get(ENV, "POOL_MAXITER", FULL ? "2000" : "300"))
const N_TRUTHS = parse(Int, get(ENV, "N_TRUTHS", "50"))
const TRUTH = Symbol(get(ENV, "TRUTH", "prior"))
const NOISE = parse(Float64, get(ENV, "NOISE", "1e-3"))

const X, SIZES, SOURCE = load_pool(typemax(Int))    # the true model: every cluster in the pool
const K = size(X, 2)
const POOL = collect(1:size(X, 1))
const PRIOR = cluster_prior(SIZES)
const BASE = DesignProblem(pool_basis(X); p = 1, prior = PRIOR, predict = POOL)  # prediction QR, once
const TRUTHS = synthetic_truths(X, PRIOR; n_truths = N_TRUTHS, noise = NOISE, kind = TRUTH)

N_TRAIN <= length(POOL) || error("N_TRAIN = $N_TRAIN exceeds the pool size $(length(POOL))")

# Dense around the interpolation threshold p = N_TRAIN, coarser elsewhere.
const P_GRID = if get(ENV, "P_GRID", "default") == "all"
    collect(1:K-1)
else
    n = N_TRAIN
    grid = [1:20; 25:25:n-50; n-50:2:n+50; n+75:75:K-1; K-1]
    sort(unique(clamp.(grid, 1, K - 1)))
end
println("p grid: ", length(P_GRID), " values from ", first(P_GRID), " to ", last(P_GRID))

# ---------------------------------------------------------------- training sets
rng = Xoshiro(21)
designs = Pair{String,Vector{Int}}["natural" => collect(1:N_TRAIN)]
for d in 1:N_RANDOM
    push!(designs, "random $d" => sort(randperm(rng, length(POOL))[1:N_TRAIN]))
end

println("optimizing training sets for p = ", join(P_TARGETS, ", "), " (", POOL_MAXITER, " iterations each)")
optimized = tmap(P_TARGETS) do pstar
    prob = with_modeled(BASE, pstar)
    c = WithNoise(Risk(), NOISE^2 / N_TRAIN)
    r = optimize_weights(c, prob, POOL, fill(1 / length(POOL), length(POOL));
                         cap = 1 / N_TRAIN, η = 8, maxiter = POOL_MAXITER)
    (pstar, select_points(r.weights, N_TRAIN), r.cost)
end
for (pstar, sel, relaxed) in optimized
    push!(designs, "optimized at p = $pstar" => sel)
    @printf("  p* = %d: relaxed cost %.4e, rounded cost %.4e\n", pstar, relaxed,
            cost(WithNoise(Risk(), NOISE^2 / N_TRAIN), with_modeled(BASE, pstar), POOL[sel]))
end

# ---------------------------------------------------------------- risk vs. p
tasks = [(name, train, p) for (name, train) in designs for p in P_GRID]
println("evaluating ", length(tasks), " (design, p) pairs on ", Threads.nthreads(), " threads")
results = tmap(tasks) do (name, train, p)
    r = risks(X, PRIOR, train, p, TRUTHS)
    (name, p, r.expected, r.heldout, r.expected_alias, r.param_alias, r.aliasing_norm)
end
write_table(joinpath(OUT, "risk_vs_p_truth_$(TRUTH).csv"),
            ["design", "p", "expected_risk", "heldout_risk", "expected_aliasing", "parameter_aliasing",
             "aliasing_norm_2"], results)

# ---------------------------------------------------------------- figure
# One panel per quantity; the expected and held-out risks share an axis range so they compare.
curve(name, col) = [r[col] for r in results if r[1] == name]
panels = ((3, "expected risk"), (4, "held-out risk (truth: $(TRUTH))"),
          (5, "expected aliasing"), (6, "parameter aliasing (truth: $(TRUTH))"),
          (7, "aliasing norm ‖A‖₂ (as in the paper)"))
fig = plot(layout = (5, 1), size = (900, 1700), legend = :topleft, xlabel = "modeled clusters p",
           yscale = :log10, link = :x)
for (k, (col, label)) in enumerate(panels)
    rand_curves = reduce(hcat, [curve("random $d", col) for d in 1:N_RANDOM])
    μ, σ = vec(mean(rand_curves; dims = 2)), vec(std(rand_curves; dims = 2))
    lower = max.(μ .- σ, minimum(μ) / 10)                       # a log axis cannot show μ − σ ≤ 0
    plot!(fig[k], P_GRID, μ; ribbon = (μ .- lower, σ), fillalpha = 0.25,
          label = "random (mean ± std, $N_RANDOM sets)", ylabel = label)
    plot!(fig[k], P_GRID, curve("natural", col); label = "natural ordering (first $N_TRAIN)")
    for (pstar, _, _) in optimized
        plot!(fig[k], P_GRID, curve("optimized at p = $pstar", col); label = "optimized at p = $pstar")
    end
    vline!(fig[k], [N_TRAIN]; label = "p = n", color = :gray, linestyle = :dash)
end
# Each expected/measured pair shares a y-range so the two compare by eye.
for (a, b) in ((1, 2), (3, 4))
    range_ab = extrema(vcat([r[panels[a][1]] for r in results], [r[panels[b][1]] for r in results]))
    plot!(fig[a]; ylims = range_ab)
    plot!(fig[b]; ylims = range_ab)
end
plot!(fig[1]; title = "Cluster expansion ($(SOURCE) pool), n = $N_TRAIN training structures, $K true clusters")
save_figure(fig, joinpath(OUT, "risk_vs_p_truth_$(TRUTH).png"))
