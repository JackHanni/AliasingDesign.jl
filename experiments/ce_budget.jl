# Cluster expansion on a tight budget: the user has chosen the modeled clusters (p fixed) and can
# afford n DFT calculations, n not much larger than p. Which n structures should they compute?
#
#   julia -t auto --project=experiments experiments/ce_budget.jl
#
# For each p in P_LIST and each budget n = ratio · p (ratio in RATIOS), training sets of n
# structures from the JuCE pool:
#   natural    the first n structures in the enumeration's natural ordering;
#   random     N_RANDOM random sets (median and 10–90% band);
#   rounded    capped Fisher–Rao flow on the pool (cap 1/n), then the n largest weights. Kept to
#              show that the relaxation says nothing here: the uniform weights on the whole pool
#              already reach the floor, so the weights barely move and the n largest are arbitrary;
#   greedy     greedy forward selection on the exact finite-n cost (`greedy_select`), from p
#              structures chosen by pivoted QR; one greedy path per p serves every budget;
#   exchanged  `greedy` improved by fast Fedorov exchange (EXCHANGE_SWEEPS swaps,
#              EXCHANGE_CANDIDATES candidates per swap), for n ≤ EXCHANGE_MAX_N.
# Each set is scored like cluster_expansion_paper.jl: expected risk (from the design alone) and
# held-out risk (fit synthetic energies, error on the rest of the pool) for both truth kinds.
#
# The floor is the risk of the fit on the whole pool: the error of the unmodeled clusters that no
# training set of these p clusters can remove. Distances to it ("excess risk") show how much of the
# avoidable error each design removes.
#
# Figures, per truth kind:
#   budget_truth_*.png   held-out risk vs n, one panel per p;
#   savings_truth_*.png  how many random structures match the exchanged set's risk (median random);
#   rounding.png         the design cost of the relaxed weights, the rounded, greedy and exchanged sets.
#
# Settings: P_LIST ("50,100,300"), RATIOS ("1,1.1,1.25,1.5,2,3,4"), N_RANDOM (50), N_TRUTHS (50),
# NOISE (1e-3), POOL_MAXITER (300), EXCHANGE_SWEEPS (30), EXCHANGE_CANDIDATES (20),
# EXCHANGE_MAX_N (1200).

include("common.jl")
include("ce_pool.jl")

const OUT = outdir("ce_budget")
envlist(T, key, default) = parse.(T, split(get(ENV, key, default), ","))
const P_LIST = envlist(Int, "P_LIST", "50,100,300")
const RATIOS = envlist(Float64, "RATIOS", "1,1.1,1.25,1.5,2,3,4")
const N_RANDOM = parse(Int, get(ENV, "N_RANDOM", "50"))
const N_TRUTHS = parse(Int, get(ENV, "N_TRUTHS", "50"))
const NOISE = parse(Float64, get(ENV, "NOISE", "1e-3"))
const POOL_MAXITER = parse(Int, get(ENV, "POOL_MAXITER", "300"))
const EXCHANGE_SWEEPS = parse(Int, get(ENV, "EXCHANGE_SWEEPS", "30"))
const EXCHANGE_CANDIDATES = parse(Int, get(ENV, "EXCHANGE_CANDIDATES", "20"))
const EXCHANGE_MAX_N = parse(Int, get(ENV, "EXCHANGE_MAX_N", "1200"))
const TRUTH_KINDS = [:prior, :unit]

const X, SIZES, SOURCE = load_pool(typemax(Int))
const K = size(X, 2)
const POOL = collect(1:size(X, 1))
const PRIOR = cluster_prior(SIZES)
const BASE = DesignProblem(pool_basis(X); p = 1, prior = PRIOR, predict = POOL)
const TRUTHS = [synthetic_truths(X, PRIOR; n_truths = N_TRUTHS, noise = NOISE, kind) for kind in TRUTH_KINDS]

"Prior-averaged risk of the fit on the whole pool with the first `p` clusters (the floor)."
function floor_risk(p)
    XU = X[:, p+1:end] .* sqrt.(PRIOR[p+1:end])'
    Q = Matrix(qr(X[:, 1:p]).Q)
    return sum(abs2, XU - Q * (Q' * XU)) / size(X, 1)
end
const FLOORS = Dict(p => floor_risk(p) for p in P_LIST)
println("floors: ", join(("p = $p: $(round(FLOORS[p]; sigdigits = 4))" for p in P_LIST), ", "))

const BUDGETS = [(p, n) for p in P_LIST for n in unique(round.(Int, p .* RATIOS)) if n <= length(POOL)]

# ---------------------------------------------------------------- optimized sets
# Greedy: one path per p up to the largest budget (the sets are nested). The noise term uses the
# largest budget's τ; at NOISE = 1e-3 it is negligible next to the aliasing either way. Greedy
# selection and the exchange spread their trials over all threads, so they run one at a time.
cost_for(n) = WithNoise(Risk(), NOISE^2 / n)
paths = Dict{Int,Vector{Int}}()
for p in P_LIST
    t0 = time()
    nmax = maximum(n for (pp, n) in BUDGETS if pp == p)
    paths[p] = greedy_select(cost_for(nmax), with_modeled(BASE, p), POOL, nmax).path
    @printf("  greedy path for p = %d up to n = %d: %.1f min\n", p, nmax, (time() - t0) / 60); flush(stdout)
end

println("optimizing ", length(BUDGETS), " budgets on ", Threads.nthreads(), " threads"); flush(stdout)
optimized = map(BUDGETS) do (p, n)
    t0 = time()
    prob = with_modeled(BASE, p)
    c = cost_for(n)
    r = optimize_weights(c, prob, POOL, fill(1 / length(POOL), length(POOL));
                         cap = 1 / n, η = 8, maxiter = POOL_MAXITER)
    rounded = select_points(r.weights, n)
    greedy = sort(paths[p][1:n])
    S_rounded, S_greedy = cost(c, prob, POOL[rounded]), cost(c, prob, POOL[greedy])
    exchanged, S_exchanged, swaps = if n <= EXCHANGE_MAX_N && EXCHANGE_SWEEPS > 0
        e = exchange(c, prob, POOL, greedy; candidates = EXCHANGE_CANDIDATES, maxiter = EXCHANGE_SWEEPS,
                     fast = true)
        (e.selected, e.cost, e.swaps)
    else
        (greedy, S_greedy, 0)
    end
    @printf("  p = %4d, n = %5d: relaxed %.4g, rounded %.4g, greedy %.4g, exchanged %.4g (%d swaps, %.1f min)\n",
            p, n, r.cost, S_rounded, S_greedy, S_exchanged, swaps, (time() - t0) / 60); flush(stdout)
    (; p, n, relaxed = r.cost, S_rounded, S_greedy, S_exchanged, rounded, greedy, exchanged)
end
write_table(joinpath(OUT, "rounding.csv"), ["p", "n", "relaxed", "rounded", "greedy", "exchanged", "floor"],
            [(o.p, o.n, o.relaxed, o.S_rounded, o.S_greedy, o.S_exchanged, FLOORS[o.p]) for o in optimized])

# ---------------------------------------------------------------- every training set, scored
rng = Xoshiro(21)
sets = Tuple{Int,Int,String,Vector{Int}}[]
for (p, n) in BUDGETS
    push!(sets, (p, n, "natural", collect(1:n)))
    for d in 1:N_RANDOM
        push!(sets, (p, n, "random $d", sort(randperm(rng, length(POOL))[1:n])))
    end
end
for o in optimized
    push!(sets, (o.p, o.n, "rounded", o.rounded))
    push!(sets, (o.p, o.n, "greedy", o.greedy))
    push!(sets, (o.p, o.n, "exchanged", o.exchanged))
end
write_table(joinpath(OUT, "training_sets.csv"), ["p", "n", "design", "structures"],
            [(p, n, name, join(train, " ")) for (p, n, name, train) in sets if !startswith(name, "random")])

println("scoring ", length(sets), " training sets"); flush(stdout)
scored = tmap(sets) do (p, n, name, train)
    r = try
        risks(X, PRIOR, train, p, TRUTHS)
    catch e
        println("  p = $p, n = $n, $name failed: ", sprint(showerror, e)); flush(stdout)
        (; expected = NaN, heldout = fill(NaN, length(TRUTHS)), expected_alias = NaN,
         param_alias = fill(NaN, length(TRUTHS)), aliasing_norm = NaN)
    end
    (p, n, name, r)
end
write_table(joinpath(OUT, "budget.csv"),
            ["p", "n", "design", "expected_risk", "expected_aliasing",
             ("heldout_risk_truth_$k" for k in TRUTH_KINDS)..., "floor"],
            [(p, n, name, r.expected, r.expected_alias, r.heldout..., FLOORS[p]) for (p, n, name, r) in scored])

# ---------------------------------------------------------------- figures
kind_name(name) = startswith(name, "random") ? "random" : name
heldout_of(p, n, kind, k) = [r.heldout[k] for (pp, nn, name, r) in scored if pp == p && nn == n && kind_name(name) == kind]
q(v, a) = isempty(v) ? NaN : quantile(filter(isfinite, v), a)

for (k, truth) in enumerate(TRUTH_KINDS)
    fig = plot(layout = (length(P_LIST), 1), size = (800, 330 * length(P_LIST)), left_margin = 10Plots.mm,
               bottom_margin = 5Plots.mm, yscale = :log10, legend = :topright)
    savings = plot(size = (800, 450), xlabel = "budget n / p", ylabel = "random structures needed ÷ n",
                   title = "Random structures needed to match the optimized set (truth: $truth)",
                   legend = :topright)
    for (i, p) in enumerate(P_LIST)
        ns = sort(unique(n for (pp, n) in BUDGETS if pp == p))
        med = [q(heldout_of(p, n, "random", k), 0.5) for n in ns]
        lo, hi = [q(heldout_of(p, n, "random", k), 0.1) for n in ns], [q(heldout_of(p, n, "random", k), 0.9) for n in ns]
        plot!(fig[i], ns, med; ribbon = (med .- lo, hi .- med), fillalpha = 0.25,
              label = "random (median, 10–90%)", marker = :circle, ms = 3)
        for kind in ("natural", "rounded", "greedy", "exchanged")
            plot!(fig[i], ns, [q(heldout_of(p, n, kind, k), 0.5) for n in ns]; label = kind, marker = :circle, ms = 3)
        end
        truth == :prior && hline!(fig[i], [FLOORS[p]]; label = "floor (fit on whole pool)", color = :gray, linestyle = :dash)
        plot!(fig[i]; title = "p = $p modeled clusters", xlabel = "training structures n",
              ylabel = "held-out risk (truth: $truth)")
        # Savings: the budget at which the median random set reaches the exchanged set's risk,
        # interpolated in log risk; budgets beyond the grid are not extrapolated.
        best = [q(heldout_of(p, n, "exchanged", k), 0.5) for n in ns]
        ratio = map(best) do target
            j = findfirst(<=(target), med)
            j === nothing ? NaN : j == 1 ? ns[1] :
                ns[j-1] + (ns[j] - ns[j-1]) * (log(med[j-1]) - log(target)) / (log(med[j-1]) - log(med[j]))
        end ./ ns
        plot!(savings, ns ./ p, ratio; label = "p = $p", marker = :circle)
    end
    save_figure(fig, joinpath(OUT, "budget_truth_$(truth).png"))
    save_figure(savings, joinpath(OUT, "savings_truth_$(truth).png"))
end

fig = plot(layout = (length(P_LIST), 1), size = (800, 300 * length(P_LIST)), left_margin = 10Plots.mm,
           yscale = :log10, legend = :topright)
for (i, p) in enumerate(P_LIST)
    os = sort([o for o in optimized if o.p == p]; by = o -> o.n)
    ns = [o.n for o in os]
    plot!(fig[i], ns, [o.relaxed for o in os]; label = "relaxed weights", marker = :circle, ms = 3)
    plot!(fig[i], ns, [o.S_rounded for o in os]; label = "rounded (n largest)", marker = :circle, ms = 3)
    plot!(fig[i], ns, [o.S_greedy for o in os]; label = "greedy", marker = :circle, ms = 3)
    plot!(fig[i], ns, [o.S_exchanged for o in os]; label = "exchanged", marker = :circle, ms = 3)
    hline!(fig[i], [FLOORS[p]]; label = "floor", color = :gray, linestyle = :dash)
    plot!(fig[i]; title = "p = $p: relaxed optimum vs. selected sets", xlabel = "training structures n", ylabel = "design cost")
end
save_figure(fig, joinpath(OUT, "rounding.png"))
