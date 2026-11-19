# The cluster-expansion candidate pool, shared by cluster_expansion.jl, cluster_expansion_paper.jl
# and aspect_ratio.jl. Include after common.jl.
#
# experiments/data/correlations.csv and cluster_sizes.csv hold the pool (see experiments/README.md;
# experiments/data/export_juce.jl writes them for JuCE's 2,346-structure FCC binary pool). Without
# them, a demo pool is built: every distinct binary configuration on periodic rings of 1–10 sites.

const DATA = joinpath(@__DIR__, "data")

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

"""
    load_pool(k_true) -> (X, sizes, source)

The pool's correlation matrix `X` (structures × clusters, natural ordering), restricted to the
first `k_true` clusters (the "true" model), with each cluster's `(size, diameter)`.
"""
function load_pool(k_true)
    if isfile(joinpath(DATA, "correlations.csv"))
        X = readdlm(joinpath(DATA, "correlations.csv"), ',', Float64)
        S = readdlm(joinpath(DATA, "cluster_sizes.csv"), ',', Float64)
        K = min(k_true, size(X, 2))
        println("loaded ", size(X, 1), " structures × ", size(X, 2), " clusters from experiments/data;",
                " the true model uses the first ", K)
        return X[:, 1:K], [(Int(S[i, 1]), S[i, 2]) for i in 1:K], "juce"
    end
    X, sizes = demo_pool()
    println("no experiments/data/correlations.csv: using the demo pool (", size(X, 1),
            " distinct ring configurations × ", size(X, 2), " clusters)")
    return X, sizes, "demo"
end

"""
    cluster_prior(sizes)

Prior variance per cluster: shrinks with size and diameter (the intuition behind the natural
ordering), σ² = 1/(size · diameter/d_NN)², with diameters relative to the nearest-neighbor pair so
the data's length unit does not matter. σ² = 1 for the empty and point clusters.
"""
function cluster_prior(sizes)
    d_nn = minimum(d for (size, d) in sizes if size >= 2)
    return [size <= 1 ? 1.0 : 1 / (size * diam / d_nn)^2 for (size, diam) in sizes]
end

"Basis functions on the pool: a point is a structure's row index into `X`."
pool_basis(X) = [t -> X[t, j] for j in axes(X, 2)]

"""
    synthetic_truths(X, prior; n_truths, noise, kind, rng) -> (; Θ, Y, Y_obs, noise)

Synthetic energies on the whole pool for `n_truths` true models, generated once and shared by
every design so that designs are compared on the same energies (common random numbers):
the true coefficients `Θ` (clusters × truths), `Y = X Θ` (noiseless, pool × truths) and
`Y_obs = Y + noise · ε` (what a training set observes).

- `kind = :prior`: θ ~ N(0, Σ) with the design's own prior.
- `kind = :unit`: θ ~ N(0, I), JuCE's synthetic setup (examples/01_fcc_binary.jl); a truth the
  prior did not anticipate, to test whether a design's advantage survives a misspecified prior.
"""
function synthetic_truths(X, prior; n_truths = 50, noise = 1e-3, kind = :prior, rng = Xoshiro(41))
    σ = kind === :prior ? sqrt.(prior) : kind === :unit ? ones(size(X, 2)) :
        throw(ArgumentError("kind must be :prior or :unit"))
    Θ = σ .* randn(rng, size(X, 2), n_truths)
    Y = X * Θ
    return (; Θ, Y, Y_obs = Y .+ noise .* randn(rng, size(Y)...), noise)
end

"""
    risks(X, prior, train, p, truths) -> (; expected, heldout, expected_alias, param_alias, aliasing_norm)

The errors of the fit θ̂ = M⁺y on the structures `train`, modeling the first `p` clusters. Risks are
measured on the held-out structures (the rest of the pool):

- `expected`: the GAD's expected risk, from the design alone (no labels): the prior-averaged mean
  squared prediction error, (‖M_test,M (I − B) Σ_M^{1/2}‖² + ‖(M_test,U − M_test,M A) Σ_U^{1/2}‖²
  + σ² ‖M_test,M M⁺‖²) / n_test. This is `cost(WithNoise(Risk(), ·), …)` with the held-out
  structures as prediction points, written out here so that every design can use its own
  held-out set without refactorizing.
- `heldout`: the measured mean squared error against the noiseless energies `truths.Y`, after
  fitting the noisy `truths.Y_obs`, averaged over the truths.

Aliasing is the parameter-space counterpart, θ̂_M − B θ_M = A θ_U + M⁺ε (subtracting B θ_M removes
the data-insufficiency part, so this isolates aliasing for any n and p):

- `expected_alias` (expected aliasing): E‖θ̂_M − B θ_M‖² = ‖A Σ_U^{1/2}‖²_F + σ²‖M⁺‖²_F, from the
  design alone: the plan's S_alias (8) plus the noise term τV of (9a).
- `param_alias` (parameter aliasing): ‖θ̂_M − B θ_M‖² for the actual fits, averaged over the truths.
- `aliasing_norm`: the GAD paper's aliasing norm ‖A‖₂ = σ_max(A), unweighted and label-free, for
  comparison with the paper's figures (see the plan, *Expected aliasing, and why not the paper's ‖A‖*).
"""
function risks(X, prior, train, p, truths)
    test = setdiff(axes(X, 1), train)
    M, Mt = X[train, :], X[test, :]
    P = pinv(M[:, 1:p])
    A = P * M[:, p+1:end]
    D = sqrt.(prior)
    E_M = (Mt[:, 1:p] * (I - P * M[:, 1:p])) .* D[1:p]'
    E_U = (Mt[:, p+1:end] .- Mt[:, 1:p] * A) .* D[p+1:end]'
    expected = (sum(abs2, E_M) + sum(abs2, E_U) + truths.noise^2 * sum(abs2, Mt[:, 1:p] * P)) / length(test)
    θ̂ = P * truths.Y_obs[train, :]
    heldout = mean(abs2, Mt[:, 1:p] * θ̂ .- truths.Y[test, :])
    expected_alias = sum(abs2, A .* D[p+1:end]') + truths.noise^2 * sum(abs2, P)
    param_alias = sum(abs2, θ̂ .- (P * M[:, 1:p]) * truths.Θ[1:p, :]) / size(θ̂, 2)
    return (; expected, heldout, expected_alias, param_alias, aliasing_norm = opnorm(A, 2))
end
