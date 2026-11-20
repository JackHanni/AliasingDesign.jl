# The Fisher–Rao step-size rules of `optimize_weights` (`step`), compared:
#   fixed (`:grow`, grow = 1), `:grow` (×1.5 after every accepted step), `:clean` (×1.5 only after a
#   step accepted at the first try), `:ratio` (trust-region style), `:bb` (Barzilai–Borwein).
#
#   julia --project=experiments experiments/benchmark_optimizer.jl
#
# On the JuCE pool (the paper experiment's settings: capped at 1/600, η = 8) with a budget of
# MAXITER_POOL iterations, and on 1D grids with up to 20,000: final cost, KKT residual, iterations,
# whether it converged (rtol = 1e-8), and wall time.

include("common.jl")
include("ce_pool.jl")

BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "4")))
const RULES = ((:grow, 1.0, "fixed"), (:grow, 1.5, "grow"), (:clean, 1.5, "clean"),
               (:ratio, 1.5, "ratio"), (:bb, 1.5, "bb"))
const MAXITER_POOL = parse(Int, get(ENV, "MAXITER_POOL", "1000"))
row(label, rule, r, t) = (@printf("  %-36s %-6s cost %.10e  residual %.2e  iterations %5d  converged %-5s %8.2f s\n",
                                  label, rule, r.cost, r.residual, r.iterations, r.converged, t); flush(stdout))

X, SIZES, _ = load_pool(typemax(Int))
N = size(X, 1)
base = DesignProblem(pool_basis(X); p = 1, prior = cluster_prior(SIZES), predict = collect(1:N))
c = WithNoise(Risk(), 1e-6 / 600)
for p in (100, 300), (step, grow, name) in RULES
    prob = with_modeled(base, p)
    w = fill(1 / N, N)
    t = @elapsed r = optimize_weights(c, prob, collect(1:N), w; cap = 1 / 600, η = 8,
                                      maxiter = MAXITER_POOL, step, grow)
    row("JuCE pool, p = $p, ≤ $MAXITER_POOL iterations", name, r, t)
end

grid = collect(range(-1, 1; length = 1000))
cases = (("Legendre risk, p = 10 of 30", DesignProblem(legendre_basis(30); p = 10, prior = k -> 1 / k^2, predict = grid), Risk(), nothing),
         ("Legendre aliasing, p = 10 of 30", DesignProblem(legendre_basis(30); p = 10, prior = k -> 1 / k^2), Aliasing(), nothing),
         ("Legendre aliasing, capped 1/40", DesignProblem(legendre_basis(30); p = 10, prior = k -> 1 / k^2), Aliasing(), 1 / 40),
         ("Legendre risk + noise, p = 20 of 60", DesignProblem(legendre_basis(60); p = 20, prior = k -> 1 / k^2, predict = grid), WithNoise(Risk(), 1e-4), nothing))
for (label, prob, cst, cap) in cases, (step, grow, name) in RULES
    w = fill(1 / length(grid), length(grid))
    optimize_weights(cst, prob, grid, w; maxiter = 2, step, grow, cap)          # compile
    t = @elapsed r = optimize_weights(cst, prob, grid, w; maxiter = 20_000, step, grow, cap)
    row(label, name, r, t)
end
