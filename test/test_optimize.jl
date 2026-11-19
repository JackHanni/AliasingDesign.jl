@testset "optimizers and discretization" begin
    rng = Xoshiro(4)
    σ² = 0.7
    xp = collect(range(-1, 1; length = 11))

    @testset "KKT helpers" begin
        g = [1.0, 2.0, 2.0, 5.0]
        # Support {2, 3} with g = 2 there; zero weights have g ≥ 2: an optimum.
        w = [0.0, 0.5, 0.5, 0.0]
        λ = kkt_multiplier(g, w)
        @test λ ≈ 2
        @test kkt_residual([3.0, 2.0, 2.0, 5.0], w, λ) == 0
        # Point 1 has g < λ but zero weight: adding weight there would help.
        @test kkt_residual(g, w, λ) ≈ 1
        # With a cap, a capped weight may have g < λ but not g > λ.
        wc = [0.5, 0.25, 0.25, 0.0]
        @test kkt_residual(g, wc, kkt_multiplier(g, wc; cap = 0.5); cap = 0.5) == 0
        @test kkt_residual([3.0, 2.0, 2.0, 5.0], wc, 2.0; cap = 0.5) ≈ 1
    end

    @testset "project_capped" begin
        w = project_capped([5.0, 1.0, 1.0, 1.0, 2.0], 0.3)
        @test sum(w) ≈ 1
        @test maximum(w) <= 0.3 + 1e-15
        @test w[1] ≈ 0.3
        @test w[2] / w[5] ≈ 0.5                  # proportions among uncapped weights kept
        @test project_capped([1.0, 1.0], 0.5) ≈ [0.5, 0.5]
        @test_throws ArgumentError project_capped(ones(3), 0.3)
    end

    @testset "Fisher–Rao flow: toy example reaches t̄ = 0" begin
        prob = toy_problem(σ²)
        pts = collect(range(-1, 1; length = 21))
        w0 = normalize(collect(range(1, 3; length = 21)), 1)    # leans right: t̄ > 0
        r = optimize_weights(Aliasing(), prob, pts, w0; η = 8)
        @test r.converged
        @test abs(sum(r.weights .* pts)) < 1e-6
        @test r.cost < 1e-12
        @test sum(r.weights) ≈ 1
        @test all(>=(0), r.weights)
    end

    @testset "Fisher–Rao flow: Fourier aliasing reaches the uniform optimum" begin
        prob = DesignProblem(fourier_basis(4); p = 5, prior = j -> 1 / max(fourier_frequency(j), 1)^2)
        grid = 2π .* (0:39) ./ 40
        w0 = normalize(1 .+ 0.5 .* cos.(grid), 1)
        r = optimize_weights(Aliasing(), prob, grid, w0; η = 8, maxiter = 20_000)
        @test r.cost < 1e-8 * cost(Aliasing(), prob, grid, w0)
    end

    @testset "Fisher–Rao flow: robust when weights hit zero or G_MM turns singular" begin
        # Predicting on a quarter of the circle drives many weights to exactly 0. This run used to
        # produce NaN weights (exp overflow at zero weights: 0 · Inf) and singular trial steps.
        fp = j -> (k = fourier_frequency(j); k == 0 ? 1.0 : float(k)^(-2))
        prob = DesignProblem(fourier_basis(12); p = 7, prior = fp, predict = collect(range(0, π / 2; length = 60)))
        grid = collect(2π .* (0:127) ./ 128)
        c = WithNoise(Risk(), 1e-3)
        r = optimize_weights(c, prob, grid, fill(1 / 128, 128); η = 8, maxiter = 1_000)
        @test all(isfinite, r.weights)
        @test all(>=(0), r.weights)
        @test sum(r.weights) ≈ 1
        @test r.cost < cost(c, prob, grid, fill(1 / 128, 128))
    end

    @testset "Fisher–Rao flow: capped weights on a pool" begin
        prob = DesignProblem(monomial_basis(6); p = 3, prior = k -> 1 / k^2)
        pool = collect(range(-1, 1; length = 30))
        n = 5
        r = optimize_weights(WithNoise(Aliasing(), 1e-4), prob, pool, fill(1 / 30, 30);
                             cap = 1 / n, η = 8, maxiter = 5_000)
        @test r.converged
        @test maximum(r.weights) <= 1 / n + 1e-12
        @test sum(r.weights) ≈ 1
        g = first_variation(WithNoise(Aliasing(), 1e-4), prob, pool, r.weights)
        @test kkt_residual(g, r.weights, r.λ; cap = 1 / n) <= 1e-6 * maximum(abs, g)
    end

    @testset "sample_points: arcsine quantiles are the Chebyshev nodes" begin
        N = 2000
        edges = range(-1, 1; length = N + 1)
        centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
        mass = (asin.(edges[2:end]) .- asin.(edges[1:end-1])) ./ π   # exact cell masses
        for n in (3, 7)
            chebyshev = sort(cos.((2(1:n) .- 1) .* π ./ 2n))
            @test sample_points(centers, mass, n; domain = (-1, 1)) ≈ chebyshev atol = 1e-5
        end
    end

    @testset "sample_points: basics" begin
        g = collect(range(0, 1; length = 11))
        s = sample_points(g, fill(1 / 11, 11), 4)
        @test issorted(s)
        @test all(0 .<= s .<= 1)
        # All weight on one grid point: every sample lands in its cell.
        w = zeros(11); w[4] = 1
        @test all(0.25 .<= sample_points(g, w, 3) .<= 0.35)
        @test_throws ArgumentError sample_points(reverse(g), fill(1 / 11, 11), 4)
    end

    @testset "select_points" begin
        @test select_points([0.1, 0.4, 0.0, 0.3, 0.2], 2) == [2, 4]
        @test_throws ArgumentError select_points([0.5, 0.5], 3)
    end

    @testset "particle flow: random points reach the Gauss nodes" begin
        # Legendre P₀..P₄ modeled and P₅ unmodeled: with n = p, the aliasing is zero exactly
        # at the roots of P₅, the Gauss nodes.
        p = 5
        prob = DesignProblem(legendre_basis(p + 1); p)
        r = optimize_points(Aliasing(), prob, sort(2rand(rng, p) .- 1); domain = (-1, 1))
        @test r.converged
        @test r.cost < 1e-12
        @test sort(r.points) ≈ gauss_nodes(p) atol = 1e-5
    end

    @testset "particle flow: stays in the domain and never raises the cost" begin
        prob = DesignProblem(monomial_basis(7); p = 4, prior = k -> 1 / k^2, predict = xp)
        for c in (WithNoise(Aliasing(), 1e-3), Risk(), WithNoise(Risk(L2Norm()), 1e-3))
            start = sort(2rand(rng, 6) .- 1)
            r = optimize_points(c, prob, start; domain = (-1, 1), maxiter = 300)
            @test all(-1 .<= r.points .<= 1)
            @test r.cost <= cost(c, prob, start)
            @test r.cost ≈ cost(c, prob, r.points)
        end
    end

    @testset "exchange" begin
        prob = DesignProblem(monomial_basis(6); p = 3, prior = k -> 1 / k^2)
        pool = collect(range(-1, 1; length = 25))
        start = [2, 9, 15, 22, 24]
        S0 = cost(Aliasing(), prob, pool[start])

        r = exchange(Aliasing(), prob, pool, start)
        @test r.cost <= S0
        @test length(r.selected) == 5
        @test allunique(r.selected)
        @test r.cost ≈ cost(Aliasing(), prob, pool[r.selected])

        # Trying every swap gives a local optimum: no single swap improves it.
        full = exchange(Aliasing(), prob, pool, start; candidates = length(pool))
        best_swap = minimum(cost(Aliasing(), prob, pool[replace(full.selected, i => j)])
                            for i in full.selected for j in setdiff(1:25, full.selected))
        @test best_swap >= full.cost - 1e-15

        @test_throws ArgumentError exchange(Aliasing(), prob, pool, [1, 1, 2, 3, 4])
    end
end
