using AliasingDesign
using LinearAlgebra
using Random
using Test

include("helpers.jl")

@testset "AliasingDesign.jl" begin
    include("test_problem.jl")
    include("test_operators.jl")
    include("test_costs.jl")
end
