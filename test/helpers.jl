# Legendre polynomial Pₖ(t) by the three-term recurrence.
function legendre(k, t)
    k == 0 && return one(t)
    p0, p1 = one(t), t
    for j in 1:k-1
        p0, p1 = p1, ((2j + 1) * t * p1 - j * p0) / (j + 1)
    end
    return p1
end

legendre_basis(nbasis) = [t -> legendre(k, t) for k in 0:nbasis-1]
monomial_basis(nbasis) = [t -> t^k for k in 0:nbasis-1]

# Legendre–Gauss nodes on [−1, 1] (roots of Pₙ), by Golub–Welsch.
gauss_nodes(n) = eigvals(SymTridiagonal(zeros(n), [j / sqrt(4j^2 - 1) for j in 1:n-1]))

# Toy problem: fit a constant when the truth also has a slope with variance σ².
toy_problem(σ²; predict = nothing) = DesignProblem([t -> 1.0, t -> t]; p = 1, prior = [1.0, σ²], predict)

# Central-difference gradient of a scalar function of a vector.
function fd_gradient(f, x; h = 1e-6)
    map(eachindex(x)) do i
        e = zero(x)
        e[i] = h
        (f(x + e) - f(x - e)) / 2h
    end
end

# Central-difference derivative of f along direction η.
fd_directional(f, x, η; h = 1e-6) = (f(x + h * η) - f(x - h * η)) / 2h

# All four cost/norm combinations.
const COSTS = [Aliasing(Frobenius()), Aliasing(L2Norm()), Risk(Frobenius()), Risk(L2Norm())]
