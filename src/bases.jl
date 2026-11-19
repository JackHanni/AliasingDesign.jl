# Basis families for the experiments. Each returns a vector of closures, so it can be passed
# straight to `DesignProblem`; the first `p` entries are the modeled functions.

"Legendre polynomial `Pₖ(t)` by the three-term recurrence."
function legendre(k, t)
    k == 0 && return one(t)
    p0, p1 = one(t), t
    for j in 1:k-1
        p0, p1 = p1, ((2j + 1) * t * p1 - j * p0) / (j + 1)
    end
    return p1
end

"""
    legendre_basis(nbasis)

Legendre polynomials `P₀, …, P_{nbasis−1}` on `[−1, 1]`. Orthogonal under the uniform density.
"""
legendre_basis(nbasis) = [t -> legendre(k, t) for k in 0:nbasis-1]

"""
    monomial_basis(nbasis)

Monomials `1, t, …, t^{nbasis−1}`. Not orthogonal under any density on `[−1, 1]`.
"""
monomial_basis(nbasis) = [t -> t^k for k in 0:nbasis-1]

"""
    gaussian_bumps(centers; width)

Gaussian bumps `exp(−(t − c)² / 2width²)`, one per center, in the order given.
"""
gaussian_bumps(centers; width) = [t -> exp(-((t - c) / width)^2 / 2) for c in centers]

"""
    fourier_basis(kmax)

Real Fourier modes on the circle, ordered by frequency:
`1, cos t, sin t, cos 2t, sin 2t, …, cos(kmax t), sin(kmax t)` (`2kmax + 1` functions).
Modeling frequencies up to `K` means `p = 2K + 1`.
"""
function fourier_basis(kmax)
    basis = Function[t -> one(t)]
    for k in 1:kmax
        push!(basis, t -> cos(k * t), t -> sin(k * t))
    end
    return basis
end

"""
    fourier_frequency(j)

The frequency of the `j`-th function of `fourier_basis` (1-based): `0, 1, 1, 2, 2, …`.
"""
fourier_frequency(j) = j ÷ 2

"""
    eos_basis(nbasis; V0 = 1.0)

Equation-of-state basis `(V/V₀)^(−2k/3)` for `k = 0, …, nbasis − 1`. The third-order
Birch–Murnaghan energy is a cubic in `V^(−2/3)`, so it is the span of the first 4 (`p = 4`).
"""
eos_basis(nbasis; V0 = 1.0) = [V -> (V / V0)^(-2k / 3) for k in 0:nbasis-1]
