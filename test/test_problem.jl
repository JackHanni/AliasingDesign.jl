@testset "DesignProblem" begin
    basis = monomial_basis(5)

    prob = DesignProblem(basis; p = 3)
    @test nmodeled(prob) == 3
    @test nunmodeled(prob) == 2
    @test prob.prior == ones(5)
    @test prob.predict === nothing

    @test DesignProblem(basis; p = 3, prior = k -> 1 / k^2).prior ≈ [1 / k^2 for k in 1:5]

    @test_throws ArgumentError DesignProblem(basis; p = 0)
    @test_throws ArgumentError DesignProblem(basis; p = 6)
    @test_throws ArgumentError DesignProblem(basis; p = 3, prior = ones(4))
    @test_throws ArgumentError DesignProblem(basis; p = 3, prior = [1, 1, -1, 1, 1])

    @testset "with_modeled" begin
        xp = collect(range(-1, 1; length = 30))
        full = DesignProblem(monomial_basis(7); p = 4, prior = k -> 1 / k^2, predict = xp)
        pts = [-0.8, -0.3, 0.1, 0.5, 0.9]
        for p in (2, 4, 6)
            fresh = DesignProblem(monomial_basis(7); p, prior = k -> 1 / k^2, predict = xp)
            @test nmodeled(with_modeled(full, p)) == p
            @test cost(Risk(), with_modeled(full, p), pts) ≈ cost(Risk(), fresh, pts)
            @test cost(Aliasing(L2Norm()), with_modeled(full, p), pts) ≈ cost(Aliasing(L2Norm()), fresh, pts)
        end
        @test_throws ArgumentError with_modeled(full, 0)
        @test_throws ArgumentError with_modeled(full, 8)
    end
end
