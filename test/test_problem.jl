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
end
