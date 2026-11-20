# A sloppy model: decaying exponentials with fixed rates, y(t) = Σⱼ θⱼ exp(−kⱼ t), linear in the
# amplitudes θ. With rates spread over decades the basis functions are nearly collinear, so the
# Fisher information's eigenvalues span many orders of magnitude (stiff and sloppy directions).
#
#   julia -t auto --project=experiments experiments/sloppy.jl
#
# The candidate pool is a dense uniform grid of times on [0, T_MAX]; the prediction set is the
# pool. A random training set is therefore uniform in time: most of its points fall where the fast
# modes have already decayed. Training sets of n times, for budgets n = ratio · p:
#   random     N_RANDOM random sets (median and 10–90% band);
#   even grid  n evenly spaced times (the conventional choice);
#   log grid   n times spaced roughly logarithmically (the usual expert choice for exponentials);
#   optimized  capped Fisher–Rao flow on the pool, rounded two ways (the n largest weights; quantile
#              rounding), then Fedorov exchange from the better of the two.
#
# Two settings (EXTRA_VAR): 0 is well specified (the modeled rates are the true ones); v > 0 adds
# K_MODEL − 1 unmodeled rates between the modeled ones, with prior variance v, so aliasing enters.
#
# Figures, per setting:
#   budget_extra_*.png      held-out risk vs n;
#   directions_extra_*.png  at n = DIRECTION_RATIO · p, the expected squared error of θ̂ along each
#                           eigendirection of the Fisher information (stiff on the left, sloppy on
#                           the right), and that direction's share of the prediction risk (λᵢ times
#                           it). This shows which parameter combinations each design pins down.
#
# Settings: K_MODEL (8), LOG10_RATES ("-1,1"), T_MAX (10), N_POOL (2000), EXTRA_VAR ("0,0.01"),
# RATIOS ("1,1.25,1.5,2,3,5,8"), N_RANDOM (200), N_TRUTHS (50), NOISE (1e-3), DIRECTION_RATIO (2),
# MAXITER (5000), EXCHANGE_SWEEPS (50).

include("common.jl")
include("ce_pool.jl")              # risks() and synthetic_truths() work for any pool matrix

const OUT = outdir("sloppy")
envlist(T, key, default) = parse.(T, split(get(ENV, key, default), ","))
const K_MODEL = parse(Int, get(ENV, "K_MODEL", "8"))
const LOG10_RATES = envlist(Float64, "LOG10_RATES", "-1,1")
const T_MAX = parse(Float64, get(ENV, "T_MAX", "10"))
const N_POOL = parse(Int, get(ENV, "N_POOL", "2000"))
const EXTRA_VARS = envlist(Float64, "EXTRA_VAR", "0,0.01")
const RATIOS = envlist(Float64, "RATIOS", "1,1.25,1.5,2,3,5,8")
const N_RANDOM = parse(Int, get(ENV, "N_RANDOM", "200"))
const N_TRUTHS = parse(Int, get(ENV, "N_TRUTHS", "50"))
const NOISE = parse(Float64, get(ENV, "NOISE", "1e-3"))
const DIRECTION_RATIO = parse(Float64, get(ENV, "DIRECTION_RATIO", "2"))
const MAXITER_FR = parse(Int, get(ENV, "MAXITER", "5000"))
const EXCHANGE_SWEEPS = parse(Int, get(ENV, "EXCHANGE_SWEEPS", "50"))

const P = K_MODEL
const TIMES = collect(range(0, T_MAX; length = N_POOL))
const POOL = collect(1:N_POOL)
const NS = unique(round.(Int, P .* RATIOS))

"Pool indices of `n` times, roughly log-spaced (strictly increasing, so all distinct)."
function log_grid(n; decades = 3)
    g = n == 1 ? [0.0] : [(exp(decades * log(10) * (j - 1) / (n - 1)) - 1) / (10.0^decades - 1) for j in 1:n]
    return [j + round(Int, (N_POOL - n) * g[j]) for j in 1:n]
end
even_grid(n) = round.(Int, range(1, N_POOL; length = n))

"""
Quantile rounding (the plan's quantile discretization on the time-ordered pool): the pool index at
each cumulative-weight quantile (j − ½)/n, nudged to the nearest unused index so all n are distinct.
"""
function quantile_round(w, n)
    cw = cumsum(w) ./ sum(w)
    idx = [min(searchsortedfirst(cw, (j - 0.5) / n), N_POOL) for j in 1:n]
    used = falses(N_POOL)
    for (j, i) in enumerate(idx)
        k = findmin(c -> used[c] ? typemax(Int) : abs(c - i), 1:N_POOL)[2]
        used[k] = true
        idx[j] = k
    end
    return sort(idx)
end

"Expected error covariance of θ̂_M − θ_M for training rows `train` (eq. 9′ plus the data-insufficiency part)."
function error_covariance(X, prior, train)
    M, MU = X[train, 1:P], X[train, P+1:end]
    Mp = pinv(M)
    B, A = Mp * M, Mp * MU
    IB = I - B
    return IB * Diagonal(prior[1:P]) * IB' + A * Diagonal(prior[P+1:end]) * A' + NOISE^2 * Mp * Mp'
end

for v in EXTRA_VARS
    tag = v == 0 ? "0" : string(v)
    rates = 10 .^ range(LOG10_RATES[1], LOG10_RATES[2]; length = K_MODEL)
    extra = v == 0 ? Float64[] : sqrt.(rates[1:end-1] .* rates[2:end])      # geometric midpoints
    allrates = [rates; extra]
    prior = [ones(K_MODEL); fill(v, length(extra))]
    X = exp.(-TIMES * allrates')
    basis = [t -> X[t, j] for j in axes(X, 2)]
    prob = DesignProblem(basis; p = P, prior, predict = POOL)
    truths = [synthetic_truths(X, prior; n_truths = N_TRUTHS, noise = NOISE, kind = :prior)]

    F = Symmetric(X[:, 1:P]' * X[:, 1:P] / N_POOL)
    λ, V = eigen(F; sortby = -)
    floor = if isempty(extra)
        0.0
    else
        XU = X[:, P+1:end] .* sqrt.(prior[P+1:end])'
        Q = Matrix(qr(X[:, 1:P]).Q)
        sum(abs2, XU - Q * (Q' * XU)) / N_POOL
    end
    @printf("\nextra variance %g: %d modeled rates, %d unmodeled; Fisher eigenvalues %.2e … %.2e (ratio %.1e); floor %.3g\n",
            v, P, length(extra), λ[1], λ[end], λ[1] / λ[end], floor); flush(stdout)

    # ------------------------------------------------------------ training sets
    rng = Xoshiro(7)
    sets = Tuple{Int,String,Vector{Int}}[]
    for n in NS
        push!(sets, (n, "even grid", even_grid(n)))
        push!(sets, (n, "log grid", log_grid(n)))
        for d in 1:N_RANDOM
            push!(sets, (n, "random $d", sort(randperm(rng, N_POOL)[1:n])))
        end
    end
    optimized = tmap(NS) do n
        c = WithNoise(Risk(), NOISE^2 / n)
        r = optimize_weights(c, prob, POOL, fill(1 / N_POOL, N_POOL); cap = 1 / n, maxiter = MAXITER_FR)
        rounded, quant = select_points(r.weights, n), quantile_round(r.weights, n)
        S_r, S_q = cost(c, prob, POOL[rounded]), cost(c, prob, POOL[quant])
        e = exchange(c, prob, POOL, S_q <= S_r ? quant : rounded; maxiter = EXCHANGE_SWEEPS)
        @printf("  n = %4d: relaxed %.4g, n largest %.4g, quantile %.4g, exchanged %.4g (%d swaps)\n",
                n, r.cost, S_r, S_q, e.cost, e.swaps); flush(stdout)
        (n, rounded, quant, e.selected)
    end
    for (n, rounded, quant, exchanged) in optimized
        push!(sets, (n, "n largest weights", rounded))
        push!(sets, (n, "quantile", quant))
        push!(sets, (n, "exchanged", exchanged))
    end
    write_table(joinpath(OUT, "training_times_extra_$tag.csv"), ["n", "design", "times"],
                [(n, name, join(round.(TIMES[train]; digits = 4), " ")) for (n, name, train) in sets
                 if !startswith(name, "random")])

    # ------------------------------------------------------------ scores
    scored = tmap(sets) do (n, name, train)
        r = risks(X, prior, train, P, truths)
        (n, name, r.expected, only(r.heldout))
    end
    write_table(joinpath(OUT, "budget_extra_$tag.csv"), ["n", "design", "expected_risk", "heldout_risk", "floor"],
                [(s..., floor) for s in scored])

    kind_name(name) = startswith(name, "random") ? "random" : name
    heldout_of(n, kind) = [s[4] for s in scored if s[1] == n && kind_name(s[2]) == kind]
    q(x, a) = quantile(filter(isfinite, x), a)
    title = v == 0 ? "well specified" : "unmodeled rates, prior variance $v"
    fig = plot(size = (800, 500), yscale = :log10, xscale = :log10, legend = :topright,
               xlabel = "training points n", ylabel = "held-out risk",
               title = "$P exponentials ($title)", titlefontsize = 11)
    med = [q(heldout_of(n, "random"), 0.5) for n in NS]
    lo, hi = [q(heldout_of(n, "random"), 0.1) for n in NS], [q(heldout_of(n, "random"), 0.9) for n in NS]
    plot!(fig, NS, med; ribbon = (med .- lo, hi .- med), fillalpha = 0.25, label = "random (median, 10–90%)",
          marker = :circle, ms = 3)
    for kind in ("even grid", "log grid", "n largest weights", "quantile", "exchanged")
        plot!(fig, NS, [only(heldout_of(n, kind)) for n in NS]; label = kind, marker = :circle, ms = 3)
    end
    floor > 0 && hline!(fig, [floor]; label = "floor (fit on whole pool)", color = :gray, linestyle = :dash)
    save_figure(fig, joinpath(OUT, "budget_extra_$tag.png"))

    # ------------------------------------------------------------ per-direction errors
    nd = argmin(n -> abs(n - DIRECTION_RATIO * P), NS)
    along(train) = (C = error_covariance(X, prior, train); [V[:, i]' * C * V[:, i] for i in 1:P])
    per = Dict{String,Vector{Float64}}()
    rand_rows = reduce(hcat, [along(train) for (n, name, train) in sets if n == nd && startswith(name, "random")])
    per["random (median)"] = vec(median(rand_rows; dims = 2))
    for (n, name, train) in sets
        n == nd && !startswith(name, "random") && (per[name] = along(train))
    end
    prior_along = [V[:, i]' * Diagonal(prior[1:P]) * V[:, i] for i in 1:P]
    write_table(joinpath(OUT, "directions_extra_$tag.csv"),
                ["direction", "eigenvalue", "prior_variance", (k for k in sort(collect(keys(per))))...],
                [(i, λ[i], prior_along[i], (per[k][i] for k in sort(collect(keys(per))))...) for i in 1:P])
    fig = plot(layout = (2, 1), size = (800, 800), yscale = :log10, left_margin = 10Plots.mm, legend = :topleft,
               xlabel = "Fisher eigendirection (1 = stiffest)")
    for k in ("random (median)", "even grid", "log grid", "quantile", "exchanged")
        plot!(fig[1], 1:P, per[k]; label = k, marker = :circle, ms = 3)
        plot!(fig[2], 1:P, λ .* per[k]; label = k, marker = :circle, ms = 3)
    end
    plot!(fig[1], 1:P, prior_along; label = "prior variance (nothing learned)", color = :gray, linestyle = :dash)
    plot!(fig[1]; ylabel = "E (vᵢᵀ(θ̂ − θ))²", title = "Parameter error by direction, n = $nd ($title)", titlefontsize = 11)
    plot!(fig[2]; ylabel = "λᵢ E (vᵢᵀ(θ̂ − θ))²", title = "Each direction's share of the prediction risk")
    save_figure(fig, joinpath(OUT, "directions_extra_$tag.png"))
end
