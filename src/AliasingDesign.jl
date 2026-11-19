module AliasingDesign

using LinearAlgebra
using TSVD: tsvd

export DesignProblem, nmodeled, nunmodeled
export design_matrix, design_matrices
export aliasing_operator, data_insufficiency_operator, error_operator
export Aliasing, Risk, Frobenius, L2Norm
export cost, cost_gradient, first_variation

include("problem.jl")
include("operators.jl")
include("costs.jl")

end # module AliasingDesign
