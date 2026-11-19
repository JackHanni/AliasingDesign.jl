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

    @testset "edge cases and errors" begin
        # Risk needs prediction points.
        @test_throws ArgumentError cost(Risk(), DesignProblem(monomial_basis(4); p = 2), [-0.5, 0.0, 0.5])

        # No unmodeled terms: nothing can alias.
        full = DesignProblem(monomial_basis(3); p = 3, predict = xp)
        @test cost(Aliasing(), full, [-0.9, -0.1, 0.4, 0.8]) == 0
        @test cost_gradient(Aliasing(), full, [-0.9, -0.1, 0.4, 0.8]) == zeros(4)

        # Bad weights.
        @test_throws ArgumentError cost(Aliasing(), prob, [-0.5, 0.0, 0.5], [0.5, 0.5])
        @test_throws ArgumentError cost(Aliasing(), prob, [-0.5, 0.0, 0.5], [1.2, -0.1, -0.1])

        # Noise level must be nonnegative.
        @test_throws ArgumentError WithNoise(Aliasing(), -1.0)
    end

    @testset "risk operator: same as M_P E_θ Σ^{1/2} / √n_P up to an orthogonal factor" begin
        # cost_op returns the compressed Qᵀ X (M_P = Q R_P), so compare XᵀX: equal Gram matrices
        # mean equal singular values, hence equal costs in every norm.
        D = Diagonal(sqrt.(prob.prior))
        M_P = design_matrix(prob.basis, xp)
        gram(X) = X' * X
        for pts in (2rand(rng, 2) .- 1, 2rand(rng, 6) .- 1)       # n < p and n > p
            X = cost_op(Risk(), prob, pts)
            @test size(X, 1) <= length(prob.basis)
            @test gram(X) ≈ gram(M_P * error_operator(prob, pts) * D / sqrt(length(xp)))
        end
        g = collect(range(-1, 1; length = 9))
        w = normalize(rand(rng, 9) .+ 0.1, 1)
        @test gram(cost_op(Risk(), prob, g, w)) ≈ gram(M_P * error_operator(prob, g, w) * D / sqrt(length(xp)))
    end

    @testset "scaling the prior scales the cost: $(c)" for c in COSTS
        # Every cost is quadratic in Σ^{1/2}, so it is linear in a common scale of the prior.
        scaled = DesignProblem(prob.basis; p = prob.p, prior = 3 .* prob.prior, predict = prob.predict)
        pts = 2rand(rng, 6) .- 1
        w = normalize(rand(rng, 6) .+ 0.1, 1)
        @test cost(c, scaled, pts) ≈ 3 * cost(c, prob, pts)
        @test cost(c, scaled, pts, w) ≈ 3 * cost(c, prob, pts, w)
    end

    @testset "weighted cost: properties of $(c)" for c in COSTS
        pts = collect(range(-1, 1; length = 9))
        w = normalize(rand(rng, 9) .+ 0.1, 1)
        @test cost(c, prob, pts, w) >= 0
        @test cost(c, prob, reverse(pts), reverse(w)) ≈ cost(c, prob, pts, w)

        # With n ≥ p and equal weights, the weighted cost is the finite-n cost exactly (B = I).
        @test cost(c, prob, pts, fill(1 / 9, 9)) ≈ cost(c, prob, pts)

        if c.norm isa L2Norm
            frob = typeof(c).name.wrapper(Frobenius())
            @test cost(c, prob, pts, w) <= cost(frob, prob, pts, w) + 1e-12
        end
    end

    @testset "Fourier: textbook aliasing at equally spaced points" begin
        # K = 3 modeled frequencies, unmodeled up to 6, n = 9 equally spaced samples.
        K, kmax, n = 3, 6, 9
        fprob = DesignProblem(fourier_basis(kmax); p = 2K + 1)
        A = aliasing_operator(fprob, 2π .* (0:n-1) ./ n)
        col(k, f) = 2k + (f == :sin) - (2K + 1)         # column of cos/sin(kt) among the unmodeled
        row(k, f) = k == 0 ? 1 : 2k + (f == :sin)       # row of cos/sin(kt) among the modeled
        # Frequencies 4 and 5 do not alias (the first to alias is n − K = 6).
        @test norm(A[:, col(4, :cos):col(5, :sin)]) < 1e-12
        # Frequency 6 folds onto 3 (6 ≡ −3 mod 9), with a sign flip on the sine.
        @test A[row(3, :cos), col(6, :cos)] ≈ 1
        @test A[row(3, :sin), col(6, :sin)] ≈ -1
        @test sum(abs, A) ≈ 2
        # Uniform weights on a grid fine enough to integrate products exactly: A[ρ] = 0.
        grid = 2π .* (0:39) ./ 40
        @test norm(aliasing_operator(fprob, grid, fill(1 / 40, 40))) < 1e-12
    end

    @testset "noise term" begin
        # Constant model, n points: M_TM = 1, M⁺ = 1ᵀ/n, so V_alias = n‖M⁺‖² = 1 = tr(G⁻¹) with G = 1.
        toy = toy_problem(σ²; predict = xp)
        pts = [-0.5, 0.25, 1.0, 0.6]
        @test noise_variance(Aliasing(), toy, pts) ≈ 1
        @test noise_variance(Risk(), toy, pts) ≈ 1                  # G_𝒫 = 1 too
        @test cost(WithNoise(Aliasing(), 0.3), toy, pts) ≈ σ² * (sum(pts) / 4)^2 + 0.3

        # A- and I-optimality at a density, against the textbook formulas.
        g = collect(range(-1, 1; length = 9))
        w = normalize(rand(rng, 9) .+ 0.1, 1)
        M = design_matrix(monomial_basis(4), g)
        G = M' * Diagonal(w) * M
        M_PM = design_matrix(monomial_basis(4), xp)
        @test noise_variance(Aliasing(), prob, g, w) ≈ tr(inv(G))
        @test noise_variance(Risk(), prob, g, w) ≈ tr(inv(G) * M_PM' * M_PM) / length(xp)

        # The variance term stays finite below n = p (the minimum-norm fit).
        @test isfinite(noise_variance(Aliasing(), prob, [-0.3, 0.6]))
    end

    noisy = [WithNoise(c, 1e-3) for c in COSTS]

    @testset "noise term: τ = 0 is the base cost, $(c)" for c in COSTS
        pts = 2rand(rng, 6) .- 1
        w = normalize(rand(rng, 6) .+ 0.1, 1)
        @test cost(WithNoise(c, 0), prob, pts) == cost(c, prob, pts)
        @test cost(WithNoise(c, 0), prob, pts, w) == cost(c, prob, pts, w)
    end

    @testset "noise term: equal weights match the finite cost, $(c)" for c in noisy
        pts = collect(range(-1, 1; length = 9))
        @test cost(c, prob, pts, fill(1 / 9, 9)) ≈ cost(c, prob, pts)
    end

    @testset "noise term: cost_gradient matches finite differences, $(c), n = $n" for c in noisy,
                                                                                       n in (2, 4, 7)
        pts = 2rand(rng, n) .- 1
        @test cost_gradient(c, prob, pts) ≈ fd_gradient(x -> cost(c, prob, x), pts) rtol = 1e-5
    end

    @testset "noise term: first_variation matches finite differences, $(c)" for c in noisy
        pts = collect(range(-1, 1; length = 15))
        w = normalize(rand(rng, 15) .+ 0.1, 1)
        δ = first_variation(c, prob, pts, w)
        for _ in 1:3
            η = randn(rng, 15)
            η .-= sum(η) / 15
            @test dot(δ, η) ≈ fd_directional(v -> cost(c, prob, pts, v), w, η) rtol = 1e-5
        end
    end

    @testset "cost_gradient on other bases: $(name)" for (name, basis, pdom) in (
            ("Legendre", legendre_basis(7), 4),
            ("Gaussian bumps", gaussian_bumps(range(-1, 1; length = 7); width = 0.4), 4),
            ("Fourier", fourier_basis(4), 5))
        bprob = DesignProblem(basis; p = pdom, prior = k -> 1 / k^2, predict = xp)
        pts = 2rand(rng, 6) .- 1
        for c in (Aliasing(), Risk(L2Norm()))
            @test cost_gradient(c, bprob, pts) ≈ fd_gradient(x -> cost(c, bprob, x), pts) rtol = 1e-5
        end
    end
end
