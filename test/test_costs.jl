@testset "costs" begin
    rng = Xoshiro(2)
    σ² = 0.7
    xp = collect(range(-1, 1; length = 11))  # prediction points

    @testset "default norm is Frobenius" begin
        @test Aliasing() == Aliasing(Frobenius())
        @test Risk() == Risk(Frobenius())
    end

    @testset "toy example: closed forms" begin
        prob = toy_problem(σ²; predict = xp)
        pts = [-0.5, 0.25, 1.0, 0.6]
        t̄ = sum(pts) / length(pts)

        # Aliasing: S = σ² t̄². A is 1×1, so both norms agree.
        for nrm in (Frobenius(), L2Norm())
            @test cost(Aliasing(nrm), prob, pts) ≈ σ² * t̄^2
            @test cost_gradient(Aliasing(nrm), prob, pts) ≈ fill(2σ² * t̄ / 4, 4)
        end

        # Risk: the prediction error at x is −(x − t̄)θ₁, so risk = σ² mean((x − t̄)²).
        # Only one column of M_P E_θ Σ^{1/2} is nonzero, so both norms agree.
        r = σ² * sum((xp .- t̄) .^ 2) / length(xp)
        dr = -2σ² * sum(xp .- t̄) / length(xp) / 4
        for nrm in (Frobenius(), L2Norm())
            @test cost(Risk(nrm), prob, pts) ≈ r
            @test cost_gradient(Risk(nrm), prob, pts) ≈ fill(dr, 4)
        end
    end

    @testset "toy example: first variation" begin
        prob = toy_problem(σ²; predict = xp)
        pts = collect(range(-1, 1; length = 9))
        w = normalize(rand(rng, 9), 1)
        t̄ = sum(w .* pts)

        @test cost(Aliasing(), prob, pts, w) ≈ σ² * t̄^2

        # δS/δρ(t) = 2σ² t̄ t, up to an additive constant.
        δ = first_variation(Aliasing(), prob, pts, w)
        @test δ .- 2σ² * t̄ .* pts ≈ fill(δ[1] - 2σ² * t̄ * pts[1], 9)

        # Risk: δ/δρ(t) = −2σ² mean(x − t̄) t, up to an additive constant.
        δ = first_variation(Risk(), prob, pts, w)
        c = -2σ² * sum(xp .- t̄) / length(xp)
        @test δ .- c .* pts ≈ fill(δ[1] - c * pts[1], 9)
    end

    @testset "Gauss points are optimal for Legendre aliasing" begin
        p = 5
        prob = DesignProblem(legendre_basis(p + 1); p)
        @test cost(Aliasing(), prob, gauss_nodes(p)) < 1e-20
        @test norm(cost_gradient(Aliasing(), prob, gauss_nodes(p))) < 1e-8
    end

    # A non-orthogonal basis with a decaying prior: the generic case.
    prob = DesignProblem(monomial_basis(7); p = 4, prior = k -> 1 / k^2, predict = xp)

    @testset "properties: $(c)" for c in COSTS
        pts = 2rand(rng, 6) .- 1

        @test cost(c, prob, pts) >= 0
        @test cost(c, prob, pts) ≈ cost(c, prob, reverse(pts))

        # The squared operator norm never exceeds the squared Frobenius norm.
        if c.norm isa L2Norm
            frob = typeof(c).name.wrapper(Frobenius())
            @test cost(c, prob, pts) <= cost(frob, prob, pts) + 1e-12
        end
    end

    @testset "cost_gradient matches finite differences: $(c), n = $n" for c in COSTS,
                                                                                n in (2, 4, 7)
        # n = 2 < p exercises the I − B term of the risk; n = 4 = p is the interpolation threshold.
        pts = 2rand(rng, n) .- 1
        g = cost_gradient(c, prob, pts)
        @test size(g) == size(pts)
        @test g ≈ fd_gradient(x -> cost(c, prob, x), pts) rtol = 1e-5
    end

    @testset "first_variation matches finite differences: $(c)" for c in COSTS
        pts = collect(range(-1, 1; length = 15))
        w = normalize(rand(rng, 15) .+ 0.1, 1)
        δ = first_variation(c, prob, pts, w)
        @test size(δ) == size(pts)

        # Directions that keep the total weight at 1.
        for _ in 1:3
            η = randn(rng, 15)
            η .-= sum(η) / 15
            @test dot(δ, η) ≈ fd_directional(v -> cost(c, prob, pts, v), w, η) rtol = 1e-5
        end
    end
end
