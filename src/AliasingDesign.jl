module AliasingDesign

using LinearAlgebra
import ForwardDiff
using TSVD: tsvd

export DesignProblem, nmodeled, nunmodeled
export design_matrix, design_matrices
export aliasing_operator, data_insufficiency_operator, error_operator
export Aliasing, Risk, Frobenius, L2Norm
export cost, cost_op, cost_gradient, first_variation, WithNoise, noise_variance
export optimize_weights, kkt_multiplier, kkt_residual, project_capped
export sample_points, select_points, optimize_points, exchange
export legendre_basis, monomial_basis, gaussian_bumps, fourier_basis, fourier_frequency, eos_basis

include("problem.jl")
include("operators.jl")
include("costs.jl")
include("gradients.jl")
include("design.jl")
include("bases.jl")

end # module AliasingDesign
