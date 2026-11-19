# Fourier fit (plan: Experiments → Fourier fit).
#
#   julia --project=experiments experiments/fourier.jl
#
# (a) Textbook aliasing: A at n equally spaced points is a 0/±1 folding matrix (heatmap).
# (b) Nyquist sweep: cost vs. n for equally spaced points, for two smoothness priors.
# (c) Coherent vs. incoherent aliasing: equally spaced, jittered, random and optimized points at
#     fixed n, in the Frobenius and spectral norms; the spectral norm's tied singular values.
# (d) Where it stops being trivial: predict only on [0, π/2]; the optimized design vs. uniform.

include("common.jl")

const OUT = outdir("fourier")
const CIRCLE = (0.0, 2π)
const K = 3                                     # modeled frequencies 0..K, p = 2K + 1
const KMAX = 12                                 # unmodeled frequencies K+1..KMAX
const DENSE = collect(2π .* (0:199) ./ 200)     # prediction points on the whole circle

fourier_prior(s) = j -> (k = fourier_frequency(j); k == 0 ? 1.0 : float(k)^(-2s))
mode_labels(kmax) = ["1"; [f * "$k" for k in 1:kmax for f in ("c", "s")]]
equispaced_circle(n) = 2π .* (0:n-1) ./ n

prob(s; kmax = KMAX, predict = DENSE) = DesignProblem(fourier_basis(kmax); p = 2K + 1, prior = fourier_prior(s), predict)

# ---------------------------------------------------------------- (a) A as a folding matrix
println("(a) aliasing operator at n = 9 equally spaced points")
A = aliasing_operator(prob(1), equispaced_circle(9))
A[abs.(A) .< 1e-12] .= 0
labels = mode_labels(KMAX)
fig = heatmap(labels[2K+2:end], labels[1:2K+1], A; c = :balance, clims = (-1, 1), yflip = true,
              xlabel = "unmodeled mode", ylabel = "modeled mode", size = (800, 330),
              title = "A at n = 9 equally spaced points (K = $K): frequency m folds onto m mod 9")
save_figure(fig, joinpath(OUT, "A_heatmap_n9.png"))

# ---------------------------------------------------------------- (b) Nyquist sweep
println("(b) Nyquist sweep")
ns = 1:30
rows = []
fig = plot(; xlabel = "n (equally spaced)", ylabel = "cost", yscale = :log10, legend = :topright,
           title = "Nyquist sweep, K = $K (p = $(2K+1))", size = (800, 450))
vline!(fig, [2K + 1]; label = "n = p", color = :gray, linestyle = :dash)
for s in (1, 2), (cname, c) in ("aliasing" => Aliasing(), "risk" => Risk())
    ys = [cost(c, prob(s), equispaced_circle(n)) for n in ns]
    append!(rows, [(s, cname, n, y) for (n, y) in zip(ns, ys)])
    plot!(fig, ns, max.(ys, 1e-16); label = "$cname, prior k^-$(2s)", marker = :circle, markersize = 3)
end
save_figure(fig, joinpath(OUT, "nyquist_sweep.png"))
write_table(joinpath(OUT, "nyquist_sweep.csv"), ["s", "cost", "n", "value"], rows)

# ---------------------------------------------------------------- (c) coherent vs. incoherent
println("(c) coherent vs. incoherent aliasing at n = 9")
n = 9
pr = prob(1)
rng = Xoshiro(5)
jittered() = sort(2π .* ((0:n-1) .+ rand(rng, n)) ./ n)
random_pts() = sort(2π .* rand(rng, n))
rows = []
for (cname, c) in ("Frobenius" => Aliasing(Frobenius()), "spectral" => Aliasing(L2Norm()))
    flow = best_particle_flow(c, pr, n, CIRCLE)
    values = [
        "equally spaced" => cost(c, pr, equispaced_circle(n)),
        "jittered (median of 100)" => median(cost(c, pr, jittered()) for _ in 1:100),
        "random (median of 100)" => median(cost(c, pr, random_pts()) for _ in 1:100),
        "particle flow (best of $STARTS)" => flow.cost,
    ]
    for (dname, v) in values
        push!(rows, (cname, dname, v))
        @printf("  %-9s %-28s %.4f\n", cname, dname, v)
    end
    gaps = diff(sort(mod.(flow.points, 2π)))
    @printf("  %-9s particle-flow spacing: mean %.4f, std %.4f (equal spacing: %.4f)\n",
            cname, mean(gaps), std(gaps), 2π / n)
end
σ = svdvals(cost_op(Aliasing(), pr, equispaced_circle(n)))
@printf("  singular values of X at equal spacing: %s (σ₁ = σ₂: the cos/sin pair tie)\n",
        join(round.(σ[1:4]; digits = 4), ", "))
write_table(joinpath(OUT, "coherent_vs_incoherent.csv"), ["norm", "design", "aliasing_cost"], rows)

# ---------------------------------------------------------------- (d) partial prediction region
println("(d) predicting only on [0, π/2]")
region = collect(range(0, π / 2; length = 60))
pq = DesignProblem(fourier_basis(KMAX); p = 2K + 1, prior = fourier_prior(1), predict = region)
grid = collect(2π .* (0:127) ./ 128)
c = WithNoise(Risk(), 1e-3)
runs, distinct = multistart(c, pq, grid)
w = runs[1].weights
designs = [
    "equally spaced on the circle" => equispaced_circle(n),
    "equally spaced on [0, π/2]" => collect(range(0, π / 2; length = n)),
    "quantiles of the relaxed optimum" => sample_points(grid, w, n; domain = CIRCLE),
]
polished = optimize_points(c, pq, designs[end][2]; maxiter = 2_000)
push!(designs, "pipeline (quantiles + particle flow)" => sort(mod.(polished.points, 2π)))
rows = [(dname, cost(c, pq, pts), join(round.(pts; digits = 3), " ")) for (dname, pts) in designs]
for r in rows
    @printf("  %-38s %.4e\n", r[1], r[2])
end
@printf("  relaxed optimum: %.4e (%d distinct optima of %d starts)\n", runs[1].cost, distinct, STARTS)
write_table(joinpath(OUT, "partial_region.csv"), ["design", "cost", "points"], rows)

fig = plot(grid, w; linetype = :sticks, label = "relaxed optimum (weights)", xlabel = "t",
           ylabel = "weight", title = "Risk + noise (τ = 1e-3), predictions on [0, π/2]",
           size = (800, 400), xticks = (0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"]))
vspan!(fig, [0, π / 2]; alpha = 0.15, color = :green, label = "prediction region")
scatter!(fig, designs[end][2], fill(maximum(w) * 1.05, n); label = "pipeline points (n = $n)", marker = :diamond)
save_figure(fig, joinpath(OUT, "partial_region.png"))
