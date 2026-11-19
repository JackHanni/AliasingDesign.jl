# 1D benchmark (plan: Experiments → 1D benchmark; Findings from the first optimizer runs).
#
#   julia --project=experiments experiments/benchmark_1d.jl        (FULL=1 for the plan's sizes)
#
# Part A, the relaxed problem (step 2): the best Fisher–Rao density on a 101-point grid for each
#   basis × cost × noise level τ, from several starts. Reproduces the findings table: number of
#   distinct optima, support, cond(G_MM), and the bias / τV split.
# Part B, finite n (steps 3–4): cost vs. n for n below, at and above p, comparing the pipeline
#   (Fisher–Rao → quantiles → particle polish) with equally spaced, Chebyshev, Gauss–Legendre and
#   random points. Also reports λ_min(M_TMᵀM_TM), the identifiability of the modeled parameters.

include("common.jl")

const OUT = outdir("benchmark_1d")
const DOMAIN = (-1.0, 1.0)
const GRID = collect(range(DOMAIN...; length = 101))
const PREDICT = collect(range(DOMAIN...; length = 50))
const P = 4
const PRIOR = k -> 1 / k^2

problems = [
    "monomial" => DesignProblem(monomial_basis(7); p = P, prior = PRIOR, predict = PREDICT),
    # Modeled bumps on the left half (centers −1, −2/3, −1/3, 0), unmodeled on the right.
    "bumps" => DesignProblem(gaussian_bumps(range(-1, 1; length = 7); width = 0.3); p = P, prior = PRIOR, predict = PREDICT),
    "legendre" => DesignProblem(legendre_basis(7); p = P, prior = PRIOR, predict = PREDICT),
]
taus = [0.0, 1e-6, 1e-4, 1e-2]

# ---------------------------------------------------------------- Part A: relaxed optimum
println("Part A: relaxed optimum (", STARTS, " starts, ", MAXITER, " iterations each)")
rows = []
best_weights = Dict{Tuple{String,String,Float64},Vector{Float64}}()
for (name, prob) in problems, (cname, base) in ("aliasing" => Aliasing(), "risk" => Risk()), τ in taus
    c = WithNoise(base, τ)
    runs, distinct = multistart(c, prob, GRID)
    best = runs[1]
    best_weights[(name, cname, τ)] = best.weights
    bias, var = bias_variance(c, prob, GRID, best.weights)
    atoms = support(GRID, best.weights)
    push!(rows, (name, cname, τ, best.cost, bias, var, distinct, length(atoms),
                 @sprintf("%.2f..%.2f", extrema(atoms)...), cond(gram(prob, GRID, best.weights)),
                 best.converged, best.residual))
    @printf("  %-8s %-8s τ=%-6g cost=%.3e  distinct=%d  atoms=%3d  cond(G)=%.1e\n",
            name, cname, τ, best.cost, distinct, length(atoms), rows[end][10])
end
write_table(joinpath(OUT, "relaxed.csv"),
            ["problem", "cost", "tau", "best_cost", "bias", "tauV", "distinct_optima", "n_atoms",
             "support_range", "cond_G", "converged", "kkt_residual"], rows)

for (name, _) in problems
    fig = plot(layout = (2, 1), size = (800, 600), legend = :topright)
    for (k, cname) in enumerate(("aliasing", "risk")), τ in taus
        plot!(fig[k], GRID, best_weights[(name, cname, τ)]; label = "τ = $τ", title = "$name, $cname",
              xlabel = "t", ylabel = "weight", linetype = :sticks, alpha = 0.7)
    end
    save_figure(fig, joinpath(OUT, "relaxed_weights_$name.png"))
end

# ---------------------------------------------------------------- Part B: finite n
println("\nPart B: cost vs. n")
ns = [2, 3, 4, 5, 6, 8, 12]
cases = [("monomial", WithNoise(Aliasing(), 1e-4)), ("monomial", WithNoise(Risk(), 1e-2)),
         ("legendre", Aliasing()), ("bumps", WithNoise(Aliasing(), 1e-4))]
rows = []
for (name, c) in cases
    prob = Dict(problems)[name]
    runs, _ = multistart(c, prob, GRID)                     # the relaxed optimum does not depend on n
    w = runs[1].weights
    curves = Dict{String,Vector{Float64}}()
    rng = Xoshiro(3)
    for n in ns
        designs = [
            "equally spaced" => equally_spaced(n, DOMAIN),
            "Chebyshev" => chebyshev(n, DOMAIN),
            "Gauss–Legendre" => gauss_legendre(n),
            "quantiles" => sample_points(GRID, w, n; domain = DOMAIN),
        ]
        polished = optimize_points(c, prob, designs[end][2]; domain = DOMAIN, maxiter = 2_000)
        push!(designs, "pipeline" => sort(polished.points))
        push!(designs, "particle flow" => best_particle_flow(c, prob, n, DOMAIN).points)
        random_costs = [cost(c, prob, sort(2rand(rng, n) .- 1)) for _ in 1:50]
        for (dname, pts) in designs
            S = cost(c, prob, pts)
            (M, _) = design_matrices(prob, pts)
            λmin = n >= P ? eigmin(Symmetric(M' * M)) : 0.0
            push!(rows, (name, string(c), n, dname, S, λmin, join(round.(pts; digits = 4), " ")))
            push!(get!(curves, dname, Float64[]), S)
        end
        push!(rows, (name, string(c), n, "random (median of 50)", median(random_costs), NaN, ""))
        push!(get!(curves, "random (median)", Float64[]), median(random_costs))
    end
    fig = plot(; xlabel = "n", ylabel = "cost", yscale = :log10, title = "$name: $c",
               legend = :outerright, size = (900, 450))
    vline!(fig, [P]; label = "n = p", color = :gray, linestyle = :dash)
    for (dname, ys) in sort(collect(curves); by = first)
        plot!(fig, ns, max.(ys, 1e-16); label = dname, marker = :circle)
    end
    save_figure(fig, joinpath(OUT, "cost_vs_n_$(name)_$(nameof(typeof(c isa WithNoise ? c.base : c))).png"))
    println("  done: $name, $c")
end
write_table(joinpath(OUT, "finite_n.csv"), ["problem", "cost", "n", "design", "value", "lambda_min_MtM", "points"], rows)
