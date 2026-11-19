@testset "operators" begin
    rng = Xoshiro(1)

    @testset "design_matrix" begin
        M = design_matrix(monomial_basis(3), [0.0, 1.0, 2.0])
        @test M == [1 0 0; 1 1 1; 1 2 4]
    end

    @testset "design_matrices" begin
        prob = DesignProblem(monomial_basis(5); p = 3)
        pts = [-0.5, 0.1, 0.7, 0.9]
        M_TM, M_TU = design_matrices(prob, pts)
        @test size(M_TM) == (4, 3)
        @test size(M_TU) == (4, 2)
        @test [M_TM M_TU] == design_matrix(prob.basis, pts)
    end

    @testset "aliasing_operator: toy example" begin
        # Fitting a constant when the truth has a slope gives A = mean of the points.
        prob = toy_problem(1.0)
        pts = [-0.5, 0.25, 1.0]
        @test aliasing_operator(prob, pts) ≈ fill(sum(pts) / 3, 1, 1)
        @test aliasing_operator(prob, [-1.0, 0.0, 1.0])[1, 1] ≈ 0 atol = 1e-12

        w = [0.2, 0.3, 0.5]
        @test aliasing_operator(prob, pts, w) ≈ fill(sum(w .* pts), 1, 1)
    end

    @testset "aliasing_operator: Gauss points remove aliasing of P_m" begin
        # Pₚ vanishes at the p Gauss nodes, so its column of M_TU is zero.
        p = 6
        prob = DesignProblem(legendre_basis(p + 1); p)
        @test norm(aliasing_operator(prob, gauss_nodes(p))) < 1e-10
        @test norm(aliasing_operator(prob, 2rand(rng, p) .- 1)) > 1e-6
    end

    @testset "aliasing_operator: uniform weights match the finite-sample A" begin
        # Least squares with n ≥ p points is the same as G_MM⁻¹ G_MU with equal weights.
        prob = DesignProblem(monomial_basis(6); p = 3)
        pts = 2rand(rng, 8) .- 1
        @test aliasing_operator(prob, pts, fill(1 / 8, 8)) ≈ aliasing_operator(prob, pts)
    end

    @testset "data_insufficiency_operator" begin
        prob = DesignProblem(legendre_basis(6); p = 4)

        # Enough points: every modeled direction is identified.
        @test norm(data_insufficiency_operator(prob, 2rand(rng, 6) .- 1)) < 1e-10

        # Too few points: I − B is the orthogonal projector onto an (p − n)-dim null space.
        P = data_insufficiency_operator(prob, 2rand(rng, 2) .- 1)
        @test size(P) == (4, 4)
        @test P ≈ P'
        @test P * P ≈ P
        @test tr(P) ≈ 2
    end

    @testset "error_operator" begin
        prob = DesignProblem(legendre_basis(6); p = 4)
        pts = 2rand(rng, 3) .- 1
        E = error_operator(prob, pts)
        @test size(E) == (6, 6)
        @test E[1:4, 1:4] ≈ data_insufficiency_operator(prob, pts)
        @test E[1:4, 5:6] ≈ -aliasing_operator(prob, pts)
        @test E[5:6, 1:4] == zeros(2, 4)
        @test E[5:6, 5:6] == I(2)
    end

    @testset "error_operator: weighted (many-samples limit)" begin
        prob = DesignProblem(monomial_basis(6); p = 4)
        pts = collect(range(-1, 1; length = 9))
        w = normalize(rand(rng, 9) .+ 0.1, 1)
        E = error_operator(prob, pts, w)
        @test size(E) == (6, 6)
        @test E[1:4, 1:4] == zeros(4, 4)  # B = I: no data insufficiency
        @test E[1:4, 5:6] ≈ -aliasing_operator(prob, pts, w)
        @test E[5:6, 1:4] == zeros(2, 4)
        @test E[5:6, 5:6] == I(2)

        # With n ≥ p, B = I already, so equal weights match the finite-sample operator.
        @test error_operator(prob, pts, fill(1 / 9, 9)) ≈ error_operator(prob, pts) atol = 1e-10
    end
end
