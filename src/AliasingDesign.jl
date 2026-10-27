module AliasingDesign

using LinearAlgebra

export design_matrix, aliasing_operator

"""
    design_matrix(basis, points)

Evaluate each basis function at each point: `M[i, j] = basis[j](points[i])`.
"""
design_matrix(basis, points) = [f(t) for t in points, f in basis]

"""
    aliasing_operator(M_TM, M_TU)

The GAD aliasing operator `A = M_TM⁺ M_TU`: how unmodeled coefficients leak into
the fitted modeled coefficients.
"""
aliasing_operator(M_TM, M_TU) = pinv(M_TM) * M_TU

end # module AliasingDesign
