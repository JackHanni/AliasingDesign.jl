using AliasingDesign
using LinearAlgebra
using Test

@testset "AliasingDesign.jl" begin
    @testset "design_matrix" begin
        basis = [t -> 1.0, t -> t, t -> t^2]
        M = design_matrix(basis, [0.0, 1.0, 2.0])
        @test M == [1 0 0; 1 1 1; 1 2 4]
    end

    @testset "aliasing_operator" begin
        # Fit a constant when the truth also has a slope: A is the mean of the points.
        pts = [-0.5, 0.25, 1.0]
        A = aliasing_operator(design_matrix([t -> 1.0], pts), design_matrix([t -> t], pts))
        @test A[1, 1] ≈ sum(pts) / length(pts)

        # Symmetric points remove the aliasing.
        pts = [-1.0, 0.0, 1.0]
        A = aliasing_operator(design_matrix([t -> 1.0], pts), design_matrix([t -> t], pts))
        @test abs(A[1, 1]) < 1e-12
    end
end
