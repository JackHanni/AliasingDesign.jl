# Equation of state (plan: Experiments → Equation of state).
#
#   julia --project=experiments experiments/equation_of_state.jl
#
# Which volumes should DFT be run at to fit E(V) with third-order Birch–Murnaghan (BM3)?
#
# Basis. BM3 is a cubic in V^(−2/3). On a ±15% volume range the plan's basis V^(−2k/3) is badly
# conditioned (every power is ≈ 1 near V₀), so this script uses the Eulerian strain
# f = (v^(−2/3) − 1)/2, v = V/V₀, scaled to [−1, 1]-ish: φ_k = (f/f_max)^k. Polynomials in f and
# in V^(−2/3) span the same functions at every truncation, so the model is still exactly BM3
# (modeled k = 0..3, p = 4) with higher-order strain terms unmodeled.
#
# Truth and prior. The truth is a Rose universal binding curve for Cu. The prior on the
# coefficients is calibrated on a *different* metal (Al), so the truth never shapes the prior
# (the plan's calibration/validation split). Noise: 1 meV/atom per DFT energy.
#
# Output: for n = 5, 7, 9 volumes, the cost and the Monte Carlo errors of V₀, B₀, B₀′ for evenly
# spaced (common practice), Chebyshev, and optimized volumes; plus the bias alone (noise-free fit).

include("common.jl")
import ForwardDiff

const OUT = outdir("equation_of_state")
const EV_PER_A3_TO_GPA = 160.21766

# Rose universal binding curve E(V) = −E_c (1 + a) e^{−a}, a = (r − r₀)/ℓ, r the Wigner–Seitz
# radius, ℓ = √(E_c / (12π B₀ r₀)). Energies in eV, volumes in Å³ per atom, B₀ in eV/Å³.
function rose(; Ec, B0_GPa, V0)
    B0 = B0_GPa / EV_PER_A3_TO_GPA
    r(V) = cbrt(3V / 4π)
    ℓ = sqrt(Ec / (12π * B0 * r(V0)))
    return V -> (a = (r(V) - r(V0)) / ℓ; -Ec * (1 + a) * exp(-a))
end

const CU = (; Ec = 3.49, B0_GPa = 140.0, V0 = 11.81)    # truth
const AL = (; Ec = 3.39, B0_GPa = 76.0, V0 = 16.60)     # calibration only

const VRANGE = (0.85, 1.15)                             # v = V/V₀
const F_MAX = maximum(abs, ((v^(-2 / 3) - 1) / 2 for v in VRANGE))
strain(v) = (v^(-2 / 3) - 1) / 2 / F_MAX
strain_basis(nbasis) = [v -> strain(v)^k for k in 0:nbasis-1]
const NBASIS = 8
const P = 4
const PREDICT = collect(range(VRANGE...; length = 100))
const GRID = collect(range(VRANGE...; length = 121))
const σ_noise = 1e-3                                    # eV/atom

# Prior: fit the calibration metal's curve with all NBASIS terms; use a log-linear envelope of
# |θ_k| as the prior standard deviation (a smooth decay, not the exact coefficients). The envelope
# is fit on k ≥ 2: θ₁ ≈ 0 for every metal because E′(V₀) = 0 at the minimum.
function calibrated_prior(metal)
    E = rose(; metal...)
    v = collect(range(VRANGE...; length = 400))
    θ = design_matrix(strain_basis(NBASIS), v) \ E.(v .* metal.V0)
    k = 2:NBASIS-1
    slope, intercept = [k ones(length(k))] \ log.(abs.(θ[k .+ 1]))
    return [exp(2 * (intercept + slope * kk)) for kk in 0:NBASIS-1], θ
end

# V₀, B₀ (GPa) and B₀′ of a fitted E(v) (v = V/V₀_ref), from E′ = 0, B = V E″(V), B′ = −1 − V E‴/E″.
function eos_parameters(E, V0_ref)
    dE(v) = ForwardDiff.derivative(E, v)
    d2E(v) = ForwardDiff.derivative(dE, v)
    d3E(v) = ForwardDiff.derivative(d2E, v)
    v = 1.0
    for _ in 1:50
        v -= dE(v) / d2E(v)
    end
    B0 = v / V0_ref * d2E(v) * EV_PER_A3_TO_GPA
    B0p = -1 - v * d3E(v) / d2E(v)
    return (; V0 = v * V0_ref, B0, B0p)
end

prior, θ_al = calibrated_prior(AL)
println("calibrated prior σ_k (from Al): ", join((@sprintf("%.1e", sqrt(s)) for s in prior), ", "))
prob = DesignProblem(strain_basis(NBASIS); p = P, prior, predict = PREDICT)

truthE = rose(; CU...)
truth_v(v) = truthE(v * CU.V0)
truth = eos_parameters(truth_v, CU.V0)
@printf("Cu truth: V₀ = %.3f Å³, B₀ = %.1f GPa, B₀′ = %.3f\n", truth.V0, truth.B0, truth.B0p)

rows = []
rng = Xoshiro(7)
fig = plot(; xlabel = "V / V₀", ylabel = "n", yticks = [5, 7, 9], legend = :outerright,
           title = "Volumes chosen for a BM3 fit (Cu, 1 meV noise)", size = (900, 350))
for n in (5, 7, 9)
    c = WithNoise(Risk(), σ_noise^2 / n)
    pip = pipeline(c, prob, GRID, n; domain = VRANGE)
    designs = ["evenly spaced" => equally_spaced(n, VRANGE),
               "Chebyshev" => chebyshev(n, VRANGE),
               "optimized" => pip.points]
    for (k, (dname, v)) in enumerate(designs)
        M = design_matrix(strain_basis(P), v)
        fitted(θ) = x -> sum(θ[j] * strain(x)^(j - 1) for j in 1:P)
        bias = eos_parameters(fitted(M \ truth_v.(v)), CU.V0)
        draws = [eos_parameters(fitted(M \ (truth_v.(v) .+ σ_noise .* randn(rng, n))), CU.V0) for _ in 1:1000]
        rmse(f) = sqrt(mean((getfield(d, f) - getfield(truth, f))^2 for d in draws))
        push!(rows, (n, dname, cost(c, prob, v), cost(Risk(), prob, v),
                     bias.V0 - truth.V0, bias.B0 - truth.B0, bias.B0p - truth.B0p,
                     rmse(:V0), rmse(:B0), rmse(:B0p), join(round.(v; digits = 4), " ")))
        @printf("  n=%d %-14s cost=%.3e  bias(B₀)=%+.2f GPa  RMSE: V₀=%.4f Å³ B₀=%.2f GPa B₀′=%.3f\n",
                n, dname, rows[end][3], rows[end][6], rows[end][8], rows[end][9], rows[end][10])
        scatter!(fig, v, fill(n + (k - 2) * 0.35, n); label = n == 5 ? dname : "", color = k,
                 markersize = 5)
    end
end
write_table(joinpath(OUT, "eos_designs.csv"),
            ["n", "design", "cost_with_noise", "bias_cost", "bias_V0", "bias_B0_GPa", "bias_B0p",
             "rmse_V0", "rmse_B0_GPa", "rmse_B0p", "volumes_over_V0"], rows)
save_figure(fig, joinpath(OUT, "eos_designs.png"))
