@testset "bases" begin
    @testset "sizes" begin
        @test length(legendre_basis(5)) == 5
        @test length(monomial_basis(5)) == 5
        @test length(gaussian_bumps([-0.5, 0.0, 0.5]; width = 0.2)) == 3
        @test length(fourier_basis(3)) == 7
        @test length(eos_basis(6)) == 6
    end

    @testset "Legendre polynomials are orthogonal" begin
        # Gauss quadrature with 8 nodes is exact for degree ≤ 15.
        x = gauss_nodes(8)
        w = 2 .* (1 .- x .^ 2) ./ (8 .* AliasingDesign.legendre.(7, x)) .^ 2
        M = design_matrix(legendre_basis(6), x)
        @test M' * Diagonal(w) * M ≈ Diagonal([2 / (2k + 1) for k in 0:5]) atol = 1e-12
    end

    @testset "monomials" begin
        @test design_matrix(monomial_basis(4), [2.0]) == [1 2 4 8]
    end

    @testset "Gaussian bumps peak at their centers" begin
        bumps = gaussian_bumps([-0.5, 0.25]; width = 0.1)
        @test bumps[1](-0.5) == 1
        @test bumps[2](0.25) == 1
        @test bumps[1](-0.4) ≈ exp(-1 / 2)
    end

    @testset "Fourier ordering and frequencies" begin
        t = 0.7
        @test design_matrix(fourier_basis(2), [t]) ≈ [1 cos(t) sin(t) cos(2t) sin(2t)]
        @test fourier_frequency.(1:7) == [0, 1, 1, 2, 2, 3, 3]
    end

    @testset "equation-of-state basis spans third-order Birch–Murnaghan" begin
        # E(V) = E₀ + (9V₀B₀/16)[(η−1)³B₀′ + (η−1)²(6 − 4η)], η = (V₀/V)^{2/3}: a cubic in V^{−2/3}.
        E₀, V₀, B₀, B₀′ = -3.0, 1.0, 0.5, 4.5
        bm(V) = (η = (V₀ / V)^(2 / 3); E₀ + 9V₀ * B₀ / 16 * ((η - 1)^3 * B₀′ + (η - 1)^2 * (6 - 4η)))
        V = collect(range(0.85, 1.15; length = 12))
        M = design_matrix(eos_basis(4; V0 = V₀), V)
        θ = M \ bm.(V)
        @test M * θ ≈ bm.(V) rtol = 1e-12
    end
end
