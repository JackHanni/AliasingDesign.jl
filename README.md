# AliasingDesign.jl

Choose where to sample so that a linear model's fit suffers as little as possible
from aliasing. Given a set of basis functions, AliasingDesign finds the sampling
density, sample points, or design matrix that minimizes either the aliasing or the
risk from the Generalized Aliasing Decomposition (GAD).

Theory of Predictive Modeling class project (BYU). Status: early development.

## Install (development)

```julia
using Pkg
Pkg.develop(path = "path/to/AliasingDesign.jl")
```

## Run the tests

For a readable report (a tree of test groups, which stubs are still pending, and failures
grouped by cause), run from the package folder:

```
julia run_tests.jl            # add --verbose for full stack traces
```

Or use the standard test runner:

```julia
using Pkg
Pkg.activate("path/to/AliasingDesign.jl")
Pkg.test()
```

## Reference

Transtrum, Hart, Jarvis, Whitehead, *Aliasing and Label-Independent Decomposition
of Risk*, [arXiv:2408.08294](https://arxiv.org/abs/2408.08294).
