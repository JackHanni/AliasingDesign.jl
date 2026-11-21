# Cluster expansion on a budget: the user has chosen the p modeled clusters (the columns of the fit)
# and can afford n DFT calculations (the rows). Which n structures should they compute? Both
# regimes are swept, n < p (more columns than rows: the minimum-norm fit, with data insufficiency)
# and n > p (more rows than columns: aliasing and noise only), through the interpolation spike at
# n = p, in the style of the GAD paper's figures (one curve per training-set family, dense x axis).
#
#   julia -t auto --project=experiments experiments/ce_budget.jl
#
# For each p in P_LIST, budgets n from p/4 to 4p (N_GRID values, plus every n within 10% of p).
# Families of nested training sets, so each is one fixed ordering evaluated at every n (dense):
#   random     N_RANDOM random orderings (median and interquartile band);
#   natural    the enumeration's natural ordering (smallest supercells first);
#   greedy     greedy forward selection on the exact finite-n cost (`greedy_select`): a standard
#              baseline, not the method of the plan.
# The plan's method, at the budgets SPARSE (multiples of p; one optimization per budget):
#   Fisher–Rao + rounding  steps 1–3: capped Fisher–Rao flow on the pool (cap 1/n), n largest weights;
#   steps 1–4              that set refined by exchange (step 4) on the exact finite-n cost;
#   greedy + step 4        the greedy set refined by the same exchange.
# Exchange: `exchange(...; fast = true)`, EXCHANGE_SWEEPS swaps, EXCHANGE_CANDIDATES candidates.
#
# Each set is scored on the held-out structures (the rest of the pool), as in
# cluster_expansion_paper.jl: expected risk (from the design alone), held-out risk for both truth
# kinds (prior and unit-variance ECIs), expected aliasing, and the paper's ‖A‖₂.
#
# Figures:
#   budget_p<p>.png  per p: one panel per quantity against n, with n = p marked;
#   savings.png      random structures needed to match greedy + step 4, ÷ n (median random set);
#   method.png       per p: the design cost of the relaxed optimum and of each selection, vs n.
#
# Settings: P_LIST ("50,100,300"), N_GRID (200), N_RANDOM (20), N_TRUTHS (50), NOISE (1e-3),
# SPARSE ("0.25,0.5,0.75,0.9,1,1.1,1.25,1.5,2,3,4", multiples of p), POOL_MAXITER (300),
# EXCHANGE_SWEEPS (30), EXCHANGE_CANDIDATES (20).

include("common.jl")
include("ce_pool.jl")

const OUT = outdir("ce_budget")
envlist(T, key, default) = parse.(T, split(get(ENV, key, default), ","))
const P_LIST = envlist(Int, "P_LIST", "50,100,300")
const N_GRID = parse(Int, get(ENV, "N_GRID", "200"))
const N_RANDOM = parse(Int, get(ENV, "N_RANDOM", "20"))
const N_TRUTHS = parse(Int, get(ENV, "N_TRUTHS", "50"))
const NOISE = parse(Float64, get(ENV, "NOISE", "1e-3"))
const SPARSE = envlist(Float64, "SPARSE", "0.25,0.5,0.75,0.9,1,1.1,1.25,1.5,2,3,4")
const POOL_MAXITER = parse(Int, get(ENV, "POOL_MAXITER", "300"))
const EXCHANGE_SWEEPS = parse(Int, get(ENV, "EXCHANGE_SWEEPS", "30"))
const EXCHANGE_CANDIDATES = parse(Int, get(ENV, "EXCHANGE_CANDIDATES", "20"))
const TRUTH_KINDS = [:prior, :unit]

const X, SIZES, SOURCE = load_pool(typemax(Int))
const K = size(X, 2)
const NPOOL = size(X, 1)
const POOL = collect(1:NPOOL)
const PRIOR = cluster_prior(SIZES)
const BASE = DesignProblem(pool_basis(X); p = 1, prior = PRIOR, predict = POOL)
const TRUTHS = [synthetic_truths(X, PRIOR; n_truths = N_TRUTHS, noise = NOISE, kind) for kind in TRUTH_KINDS]

cost_for(n) = WithNoise(Risk(), NOISE^2 / n)
grid(p) = filter(n -> 1 <= n <= NPOOL - 1,
                 sort(unique([round.(Int, range(p / 4, 4p; length = N_GRID)); round(Int, 0.9p):round(Int, 1.1p)])))
sparse(p) = filter(n -> 1 <= n <= NPOOL - 1, unique(round.(Int, SPARSE .* p)))

const NAN_SCORE = (; expected = NaN, heldout = fill(NaN, length(TRUTH_KINDS)), expected_alias = NaN,
                   param_alias = fill(NaN, length(TRUTH_KINDS)), aliasing_norm = NaN)
score(train, p) = try
    risks(X, PRIOR, train, p, TRUTHS)
catch e
    println("  p = $p, n = $(length(train)) failed: ", sprint(showerror, e)); flush(stdout)
    NAN_SCORE
end
row(p, n, design, r) = (p, n, design, r.expected, r.heldout..., r.expected_alias, r.aliasing_norm)
const HEADER = ["p", "n", "design", "expected_risk", ("heldout_risk_truth_$k" for k in TRUTH_KINDS)...,
                "expected_aliasing", "aliasing_norm_2"]

dense_rows, sparse_rows, cost_rows = Any[], Any[], Any[]
rng = Xoshiro(21)
for p in P_LIST
    t0 = time()
    prob = with_modeled(BASE, p)
    ns, nsp = grid(p), sparse(p)
    nmax = maximum(ns)

    # ------------------------------------------------------------ nested families (dense in n)
    orders = Pair{String,Vector{Int}}["natural" => POOL]
    for d in 1:N_RANDOM
        push!(orders, "random $d" => randperm(rng, NPOOL))
    end
    greedy = greedy_select(cost_for(nmax), prob, POOL, nmax).path    # τ of the largest budget
    push!(orders, "greedy" => greedy)
    @printf("p = %d: greedy path to n = %d in %.1f min\n", p, nmax, (time() - t0) / 60); flush(stdout)
    tasks = [(name, n) for (name, _) in orders for n in ns]
    order_of = Dict(orders)
    scored = tmap(sort(tasks; by = last, rev = true)) do (name, n)
        (name, n, score(sort(order_of[name][1:n]), p))
    end
    append!(dense_rows, [row(p, n, name, r) for (name, n, r) in scored])

    # ------------------------------------------------------------ the plan's method (sparse in n)
    for n in nsp
        t1 = time()
        c = cost_for(n)
        rel = optimize_weights(c, prob, POOL, fill(1 / NPOOL, NPOOL); cap = 1 / n, η = 8, maxiter = POOL_MAXITER)
        rounded = select_points(rel.weights, n)
        ex = (start) -> exchange(c, prob, POOL, start; candidates = EXCHANGE_CANDIDATES,
                                 maxiter = EXCHANGE_SWEEPS, fast = true)
        full = ex(rounded)
        g = sort(greedy[1:n])
        gx = ex(g)
        sets = ["Fisher–Rao + rounding" => rounded, "steps 1–4" => full.selected,
                "greedy + step 4" => gx.selected]
        for (name, train) in sets
            push!(sparse_rows, row(p, n, name, score(train, p)))
        end
        S = [cost(c, prob, POOL[s]) for s in (rounded, full.selected, g, gx.selected)]
        push!(cost_rows, (p, n, rel.cost, S...))
        @printf("  n = %4d: relaxed %.4g | rounded %.4g → steps 1–4 %.4g (%d swaps) | greedy %.4g → +step 4 %.4g (%d swaps) | %.1f min\n",
                n, rel.cost, S[1], S[2], full.swaps, S[3], S[4], gx.swaps, (time() - t1) / 60); flush(stdout)
    end
    @printf("p = %d done in %.1f min\n", p, (time() - t0) / 60); flush(stdout)
end
write_table(joinpath(OUT, "budget_dense.csv"), HEADER, dense_rows)
write_table(joinpath(OUT, "budget_sparse.csv"), HEADER, sparse_rows)
write_table(joinpath(OUT, "method_costs.csv"),
            ["p", "n", "relaxed", "fisher_rao_rounding", "steps_1_4", "greedy", "greedy_step4"], cost_rows)

# ---------------------------------------------------------------- figures
family(name) = startswith(name, "random") ? "random" : name
pick(rows, p, fam, col) = begin
    sel = [r for r in rows if r[1] == p && family(r[3]) == fam]
    ns = sort(unique(r[2] for r in sel))
    ns, [[r[col] for r in sel if r[2] == n] for n in ns]
end
q(v, a) = (f = filter(isfinite, v); isempty(f) ? NaN : quantile(f, a))
const PANELS = [(4, "expected risk"), (5, "held-out risk (truth: prior)"), (6, "held-out risk (truth: unit)"),
                (7, "expected aliasing"), (8, "aliasing norm ‖A‖₂")]
const STYLE = Dict("natural" => (:darkorange, :solid), "greedy" => (:purple, :solid),
                   "Fisher–Rao + rounding" => (:green, :diamond), "steps 1–4" => (:red, :circle),
                   "greedy + step 4" => (:black, :star5))

for p in P_LIST
    fig = plot(layout = (length(PANELS), 1), size = (900, 300 * length(PANELS)), yscale = :log10,
               legend = :topright, left_margin = 12Plots.mm, link = :x, xlabel = "training structures n")
    for (k, (col, label)) in enumerate(PANELS)
        ns, v = pick(dense_rows, p, "random", col)
        med = [q(x, 0.5) for x in v]
        lo, hi = [q(x, 0.25) for x in v], [q(x, 0.75) for x in v]
        plot!(fig[k], ns, med; ribbon = (med .- lo, hi .- med), fillalpha = 0.25, color = :steelblue,
              label = "random (median, IQR, $N_RANDOM orderings)", ylabel = label)
        for fam in ("natural", "greedy")
            ns, v = pick(dense_rows, p, fam, col)
            plot!(fig[k], ns, first.(v); color = STYLE[fam][1], label = fam == "greedy" ? "greedy (baseline)" : "natural ordering")
        end
        for fam in ("Fisher–Rao + rounding", "steps 1–4", "greedy + step 4")
            ns, v = pick(sparse_rows, p, fam, col)
            plot!(fig[k], ns, first.(v); color = STYLE[fam][1], marker = STYLE[fam][2], ms = 4, lw = 1, label = fam)
        end
        vline!(fig[k], [p]; color = :gray, linestyle = :dash, label = "n = p")
    end
    plot!(fig[1]; title = "Cluster expansion ($SOURCE pool): p = $p modeled clusters, $K true clusters")
    save_figure(fig, joinpath(OUT, "budget_p$(p).png"))
end

# Savings: the smallest budget past which the median random set has reached the greedy + step 4
# risk (held-out, prior truth), on the dense grid; NaN if it never does within 4p. Shown for n ≥ p.
savings = plot(size = (800, 450), xlabel = "budget n / p", ylabel = "random structures needed ÷ n",
               title = "Random structures needed to match greedy + step 4 (held-out risk, prior truth)",
               titlefontsize = 10, legend = :topleft)
for p in P_LIST
    ns, v = pick(dense_rows, p, "random", 5)
    med = [q(x, 0.5) for x in v]
    nb, vb = pick(sparse_rows, p, "greedy + step 4", 5)
    ratio = map(zip(nb, first.(vb))) do (n, target)
        j = findfirst(i -> ns[i] >= n && med[i] <= target, eachindex(ns))
        j === nothing ? NaN : ns[j] / n
    end
    keep = nb .>= p
    plot!(savings, nb[keep] ./ p, ratio[keep]; marker = :circle, label = "p = $p")
end
save_figure(savings, joinpath(OUT, "savings.png"))

fig = plot(layout = (length(P_LIST), 1), size = (900, 320 * length(P_LIST)), yscale = :log10,
           legend = :topright, left_margin = 12Plots.mm)
for (i, p) in enumerate(P_LIST)
    rs = sort([r for r in cost_rows if r[1] == p]; by = r -> r[2])
    ns = [r[2] for r in rs]
    for (j, label, color, m) in ((3, "relaxed optimum (steps 1–2)", :gray, :none),
                                 (4, "Fisher–Rao + rounding", :green, :diamond),
                                 (5, "steps 1–4", :red, :circle), (6, "greedy (baseline)", :purple, :utriangle),
                                 (7, "greedy + step 4", :black, :star5))
        plot!(fig[i], ns, [r[j] for r in rs]; label, color, marker = m, ms = 4)
    end
    vline!(fig[i], [p]; color = :gray, linestyle = :dash, label = "n = p")
    plot!(fig[i]; title = "p = $p: design cost (expected risk on the pool)", xlabel = "training structures n",
          ylabel = "design cost")
end
save_figure(fig, joinpath(OUT, "method.png"))
