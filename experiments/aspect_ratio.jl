# Risk vs. the aspect ratio p/n of the training design matrix, on the cluster-expansion pool.
#
#   julia -t auto --project=experiments experiments/aspect_ratio.jl
#
# For each training-set size n in NS and each aspect ratio γ in RATIOS, model the first
# p = round(γ n) clusters and evaluate:
#   random   DRAWS random training sets of n structures;
#   natural  the first n structures in the enumeration's natural ordering.
# Each γ therefore has several points (one per n and draw); the figure shows all of them and their
# mean ± std per γ, for three quantities on the held-out structures (the rest of the pool):
#   expected risk  the GAD's prediction from the design alone (prior-averaged, no labels);
#   held-out risk  measured: fit synthetic energies on the training set, mean squared error on the
#                  held-out structures, averaged over N_TRUTHS truths (TRUTH=prior or unit);
#   expected aliasing  E‖θ̂_M − B θ_M‖² from the design alone;
#   parameter aliasing ‖θ̂_M − B θ_M‖² of the actual fits, averaged over the truths;
#   aliasing norm      ‖A‖₂, as in the GAD paper (unweighted, label-free).
# The true model uses all of the pool's clusters (2,344 for the JuCE pool). Points with different n at the
# same γ test whether the risk is a function of the aspect ratio alone, as random-matrix theory
# predicts for i.i.d. designs; the double-descent peak should sit at γ = 1.
#
# Cost (rough, 8 threads): one risk evaluation per (n, γ, set), each ≲ 1 s on one core;
# the defaults (4 n × 13 γ × 6 sets ≈ 300 evaluations) take a few minutes, mostly loading.
#
# Settings: NS ("100,200,400,600"), RATIOS (comma-separated; default below), DRAWS (5), N_TRUTHS (50),
# TRUTH ("prior" or "unit"), NOISE (1e-3).

include("common.jl")
include("ce_pool.jl")

const OUT = outdir("aspect_ratio")
const NS = parse.(Int, split(get(ENV, "NS", "100,200,400,600"), ","))
const RATIOS = parse.(Float64, split(get(ENV, "RATIOS", "0.1,0.25,0.5,0.75,0.9,0.95,1.0,1.05,1.1,1.25,1.5,2.0,3.0"), ","))
const DRAWS = parse(Int, get(ENV, "DRAWS", "5"))
const N_TRUTHS = parse(Int, get(ENV, "N_TRUTHS", "50"))
const TRUTH = Symbol(get(ENV, "TRUTH", "prior"))
const NOISE = parse(Float64, get(ENV, "NOISE", "1e-3"))

const X, SIZES, SOURCE = load_pool(typemax(Int))
const K = size(X, 2)
const POOL = collect(1:size(X, 1))
const PRIOR = cluster_prior(SIZES)
const TRUTHS = synthetic_truths(X, PRIOR; n_truths = N_TRUTHS, noise = NOISE, kind = TRUTH)

maximum(NS) <= length(POOL) || error("largest n exceeds the pool size $(length(POOL))")

# ---------------------------------------------------------------- tasks
rng = Xoshiro(31)
tasks = []
for n in NS, γ in RATIOS
    p = clamp(round(Int, γ * n), 1, K - 1)
    push!(tasks, ("natural", n, γ, p, collect(1:n)))
    for d in 1:DRAWS
        push!(tasks, ("random", n, γ, p, sort(randperm(rng, length(POOL))[1:n])))
    end
end
println(length(tasks), " evaluations on ", Threads.nthreads(), " threads")
results = tmap(tasks) do (kind, n, γ, p, train)
    r = risks(X, PRIOR, train, p, TRUTHS)
    (kind, n, γ, p, p / n, r.expected, r.heldout, r.expected_alias, r.param_alias, r.aliasing_norm)
end
write_table(joinpath(OUT, "aspect_ratio_truth_$(TRUTH).csv"),
            ["design", "n", "target_ratio", "p", "ratio", "expected_risk", "heldout_risk",
             "expected_aliasing", "parameter_aliasing", "aliasing_norm_2"], results)

# ---------------------------------------------------------------- summary and figure
const QUANTITIES = ((6, "expected risk"), (7, "held-out risk (truth: $(TRUTH))"),
                    (8, "expected aliasing"), (9, "parameter aliasing (truth: $(TRUTH))"),
                    (10, "aliasing norm ‖A‖₂ (as in the paper)"))
summary = []
for kind in ("random", "natural"), γ in RATIOS, (col, name) in QUANTITIES
    v = [r[col] for r in results if r[1] == kind && r[3] == γ]
    push!(summary, (kind, γ, name, length(v), mean(v), std(v), minimum(v), maximum(v)))
end
write_table(joinpath(OUT, "aspect_ratio_summary_truth_$(TRUTH).csv"),
            ["design", "ratio", "quantity", "points", "mean", "std", "min", "max"], summary)

# Legend inside the top panel (an outside legend would shrink that panel only and misalign the
# x axes); a left margin keeps the long y labels on the canvas.
fig = plot(layout = (5, 1), size = (900, 1700), xlabel = "aspect ratio p/n", xscale = :log10,
           yscale = :log10, legend = :topleft, link = :x, left_margin = 12Plots.mm)
colors = Dict(n => i for (i, n) in enumerate(NS))
for (k, (col, label)) in enumerate(QUANTITIES)
    for n in NS
        pts = [r for r in results if r[1] == "random" && r[2] == n]
        jitter = exp.(0.04 .* (rand(rng, length(pts)) .- 0.5))   # spread overlapping points
        scatter!(fig[k], [r[5] for r in pts] .* jitter, [r[col] for r in pts]; label = k == 1 ? "random, n = $n" : "",
                 color = colors[n], markersize = 3, markerstrokewidth = 0, alpha = 0.5, ylabel = label)
    end
    for (kind, style) in (("random", :circle), ("natural", :diamond))
        rows = [s for s in summary if s[1] == kind && s[3] == label]
        μ, σ = [s[5] for s in rows], [s[6] for s in rows]
        lower = min.(σ, 0.9 .* μ)                              # a log axis cannot show μ − σ ≤ 0
        plot!(fig[k], RATIOS, μ; yerror = (lower, σ), marker = style, color = kind == "random" ? :black : :red,
              label = k == 1 ? "$kind: mean ± std" : "", linewidth = 2)
    end
    vline!(fig[k], [1.0]; label = k == 1 ? "p = n" : "", color = :gray, linestyle = :dash)
end
# Each expected/measured pair shares a y-range so the two compare by eye.
for (a, b) in ((1, 2), (3, 4))
    range_ab = extrema(vcat([r[QUANTITIES[a][1]] for r in results], [r[QUANTITIES[b][1]] for r in results]))
    plot!(fig[a]; ylims = range_ab)
    plot!(fig[b]; ylims = range_ab)
end
plot!(fig[1]; title = "Risk vs. aspect ratio ($(SOURCE) pool, $K true clusters)")
save_figure(fig, joinpath(OUT, "aspect_ratio_truth_$(TRUTH).png"))
