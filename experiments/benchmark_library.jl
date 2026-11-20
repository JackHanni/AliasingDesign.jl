# Where the library spends its time, on the JuCE pool (or the demo pool) and a 1D problem.
#
#   julia --project=experiments experiments/benchmark_library.jl
#
# Times each piece of an `optimize_weights` iteration (weighted fit, cost, first variation), a
# finite-n cost and its parts, and one `exchange` trial, at a few p. BLAS uses every thread
# (as a user's session would); Julia threads are not used.

include("common.jl")
include("ce_pool.jl")
using AliasingDesign: weighted_fit, fit_cost, fit_first_variation, design_matrices

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "4")))
println("BLAS threads: ", BLAS.get_num_threads())

# Median wall time of f() over `reps` runs, after one warm-up.
function timeit(f; reps = 3)
    f()
    return median([@elapsed(f()) for _ in 1:reps])
end
report(label, t) = (@printf("  %-46s %9.4f s\n", label, t); flush(stdout))

X, SIZES, _ = load_pool(typemax(Int))
K, N = size(X, 2), size(X, 1)
prior = cluster_prior(SIZES)
pool = collect(1:N)
c = WithNoise(Risk(), 1e-6 / 600)

report("DesignProblem construction (closures + QR)",
       timeit(() -> DesignProblem(pool_basis(X); p = 1, prior, predict = pool); reps = 1))
report("design_matrix only (closures)", timeit(() -> design_matrix(pool_basis(X), pool); reps = 1))
base = DesignProblem(pool_basis(X); p = 1, prior, predict = pool)

for p in (100, 300, 1000)
    prob = with_modeled(base, p)
    println("p = $p (N = $N pool points, q = $(K - p))"); flush(stdout)
    w = fill(1 / N, N)
    M_TM, M_TU = design_matrices(prob, pool)
    fit = weighted_fit(M_TM, M_TU, w)
    report("weighted_fit (QR + Qᵀ M_TU)", timeit(() -> weighted_fit(M_TM, M_TU, w)))
    report("fit_cost (risk + noise)", timeit(() -> fit_cost(c, prob, fit)))
    report("fit_first_variation (risk + noise)", timeit(() -> fit_first_variation(c, prob, fit)))
    report("optimize_weights, 5 iterations",
           timeit(() -> optimize_weights(c, prob, pool, w; cap = 1 / 600, η = 8, maxiter = 5); reps = 1))
    sel = sort(randperm(Xoshiro(1), N)[1:600])
    pts = pool[sel]
    report("cost at 600 points (one exchange trial)", timeit(() -> cost(c, prob, pts)))
    report("  of which design_matrices (closures)", timeit(() -> design_matrices(prob, pts)))
    report("  of which pinv(M_TM)", timeit(() -> pinv(X[sel, 1:p])))
    println("  exchange sweep at n = 600, 20 candidates = 12,000 such trials"); flush(stdout)
end

println("1D: Legendre, 60 basis functions, p = 20, 2,000-point grid, risk on the grid")
grid = collect(range(-1, 1; length = 2000))
prob1 = DesignProblem(legendre_basis(60); p = 20, prior = k -> 1 / k^2, predict = grid)
w1 = fill(1 / length(grid), length(grid))
report("optimize_weights, 200 iterations",
       timeit(() -> optimize_weights(Risk(), prob1, grid, w1; maxiter = 200); reps = 1))
pts1 = sample_points(grid, w1, 30)
report("optimize_points, 50 iterations",
       timeit(() -> optimize_points(Risk(), prob1, pts1; domain = (-1, 1), maxiter = 50); reps = 1))
