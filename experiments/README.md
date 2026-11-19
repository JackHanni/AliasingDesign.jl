# Experiments

Scripts for the experiments in [`docs/project_plan.md`](../docs/project_plan.md). They use their own
environment (the package plus Plots), so the package itself does not depend on Plots.

```
julia --project=experiments experiments/<script>.jl          # quick run
FULL=1 julia --project=experiments experiments/<script>.jl   # the plan's sizes (6 starts, 20,000 iterations)
```

The first run installs and precompiles Plots, which takes a few minutes. Each script writes CSV
tables and PNG figures to `experiments/output/<name>/` (not tracked by git; rerun to regenerate).

| Script | Plan section | What it produces | Quick run |
|---|---|---|---|
| `benchmark_1d.jl` | 1D benchmark; Findings | Part A: the relaxed optimum for monomial, Gaussian-bump and Legendre bases × aliasing/risk × τ (distinct optima, support, cond(G), bias vs. τV). Part B: cost vs. n for the pipeline against equally spaced, Chebyshev, Gauss–Legendre and random points. | ~2 min |
| `fourier.jl` | Fourier fit | (a) heatmap of A at n = 9 equally spaced points; (b) Nyquist sweep; (c) coherent vs. incoherent aliasing, Frobenius and spectral; (d) predictions on [0, π/2] only, where uniform is no longer optimal. | ~2 min |
| `equation_of_state.jl` | Equation of state | Volumes for a BM3 fit of a Cu Rose curve (prior calibrated on Al), n = 5, 7, 9: cost and Monte Carlo errors of V₀, B₀, B₀′ for evenly spaced, Chebyshev and optimized volumes. | ~1 min |
| `cluster_expansion.jl` | Cluster expansion; Discrete domains | Capped Fisher–Rao → top-n rounding → exchange vs. random training sets: costs, held-out RMSE, the relaxation gap, and risk vs. p. | ~1 min (demo pool); longer on the JuCE pool |

`common.jl` holds the shared helpers (multi-start Fisher–Rao, the steps 2–4 pipeline, reference
designs, CSV and figure output).

## Cluster-expansion data

`cluster_expansion.jl` reads the candidate pool from `experiments/data/` (not tracked by git):

- `correlations.csv`: one row per candidate structure, one column per cluster in the natural
  ordering (empty, point, pairs short → long, triplets, …), no header.
- `cluster_sizes.csv`: one row per cluster, `size,diameter` in any length unit (the prior uses
  diameters relative to the nearest-neighbor pair), no header.

**From JuCE.** `experiments/data/export_juce.jl` writes both files for the 2,346-structure FCC
binary pool (JuCE's `data/struct_enum.out.1-10_fcc_binary`), using JuCE's standard design matrix
(`FromTrainingCells` + `SinCos(2)`, 2,344 clusters). It runs in JuCE's own environment, and it
lives in the ignored data folder because JuCE is private:

```
julia --project=$HOME/BYUResearch/JuCE.jl experiments/data/export_juce.jl    # ~5 min, once
```

The energies are synthetic, built like JuCE's `examples/01_fcc_binary.jl` (random true ECIs,
energies `M θ`), except that the ECIs are drawn from the prior, θ ~ (0, Σ), instead of with unit
variance, and 1 meV noise is added. The "true" model uses the first `K_TRUE` clusters in natural order (default
300: all pairs and triplets and most 4-body clusters; set `K_TRUE=...` to change it).

Without the files, the script runs on a demo pool: every distinct binary configuration on periodic
rings of 1–10 sites, with 1D cluster correlation functions.

Before bringing in real DFT energies, split them once into a calibration part (prior, synthetic
truth) and a validation part used only for the final held-out risk (plan: *Choosing the
prediction points* and *Cluster expansion*).

## Notes on reading the results

- Costs at τ = 0 can be degenerate (plan, *Findings*): monomial aliasing collapses toward t = 0,
  and Frobenius risk is minimized by sampling the prediction distribution. The scripts report a
  range of τ for that reason.
- Random designs with n = p can be nearly singular, so their upper quantiles can be huge (e.g. the
  90th percentile at p = n = 10 in the cluster-expansion risk-vs-p table). Medians are more stable.
- The relaxation gap in the cluster expansion bounds the distance to the best n-structure design
  only if the relaxed minimum is global; the cost is not convex, so treat it as empirical.
