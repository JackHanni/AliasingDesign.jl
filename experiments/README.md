# Experiments

Scripts for the experiments in [`docs/project_plan.md`](../docs/project_plan.md). They use their own
environment (the package plus Plots), so the package itself does not depend on Plots.

```
julia -t auto --project=experiments experiments/<script>.jl          # quick run, all cores
FULL=1 julia -t auto --project=experiments experiments/<script>.jl   # the plan's sizes (6 starts, 20,000 iterations)
```

`-t auto` runs independent starts and sweeps on separate threads (results are the same as with one
thread); without it everything runs on one core.

The first run installs and precompiles Plots, which takes a few minutes. Each script writes CSV
tables and PNG figures to `experiments/output/<name>/` (not tracked by git; rerun to regenerate).

| Script | Plan section | What it produces | Quick run |
|---|---|---|---|
| `benchmark_1d.jl` | 1D benchmark; Findings | Part A: the relaxed optimum for monomial, Gaussian-bump and Legendre bases × aliasing/risk × τ (distinct optima, support, cond(G), bias vs. τV). Part B: cost vs. n for the pipeline against equally spaced, Chebyshev, Gauss–Legendre and random points. | ~2 min |
| `fourier.jl` | Fourier fit | (a) heatmap of A at n = 9 equally spaced points; (b) Nyquist sweep; (c) coherent vs. incoherent aliasing, Frobenius and spectral; (d) predictions on [0, π/2] only, where uniform is no longer optimal. | ~2 min |
| `equation_of_state.jl` | Equation of state | Volumes for a BM3 fit of a Cu Rose curve (prior calibrated on Al), n = 5, 7, 9: cost and Monte Carlo errors of V₀, B₀, B₀′ for evenly spaced, Chebyshev and optimized volumes. | ~1 min |
| `cluster_expansion.jl` | Cluster expansion; Discrete domains | Capped Fisher–Rao → top-n rounding → exchange vs. random training sets: costs, held-out RMSE, the relaxation gap, and risk vs. p. | ~1 min (demo pool); longer on the JuCE pool |
| `cluster_expansion_paper.jl` | Cluster expansion (GAD paper scale, cf. Fig. 6) | n = 600 training structures, all 2,344 clusters true; expected risk, held-out risk and aliasing vs. p for natural-ordering, random (mean ± std) and optimized training sets. | ~20 min on 16 cores (default p grid); hours with `P_GRID=all` (checkpointed per training set) |
| `aspect_ratio.jl` | Cluster expansion | Expected risk, held-out risk and aliasing vs. the aspect ratio p/n, several n and random draws per ratio (all points plus mean ± std), random vs. natural ordering. | a few min |
| `ce_budget.jl` | Cluster expansion (tight budgets) | p fixed (the user's chosen clusters), n = p … 4p: held-out risk vs. n for natural, random, rounded, greedy and exchanged training sets, with the whole-pool floor; how many random structures match the optimized set; relaxed optimum vs. the selected sets. | ~1 min with `P_LIST=20`; ~2 h on 32 cores for the defaults |
| `sloppy.jl` | Sloppy models | Sum of decaying exponentials (Fisher eigenvalues spanning ~10⁶), well specified and with unmodeled rates: held-out risk vs. n for random, even and log grids and optimized times; parameter error along each Fisher eigendirection. | a few min |

`common.jl` holds the shared helpers (multi-start Fisher–Rao, the steps 2–4 pipeline, reference
designs, CSV and figure output). `ce_pool.jl` loads the cluster-expansion pool and its prior for the
three cluster-expansion scripts. Rows of the JuCE pool are in the enumeration's natural ordering
(smallest supercells first), so "natural ordering" training sets are the first n rows.

## Running on the BYU supercomputer

Long runs (e.g. `cluster_expansion_paper.jl` with `P_GRID=all`) go to BYU Research Computing as
batch jobs; see https://rc.byu.edu/wiki/?id=Getting+Started.

1. **Once:** enroll in RC two-factor authentication (separate from BYU Duo) at
   https://rc.byu.edu/account/authenticate/enroll, then `ssh <user>@ssh.rc.byu.edu`.
   Load Julia with `module load julia/1.12.6-dnqrzwe` (`module spider julia` lists the versions;
   `run.sh` loads this one).
2. **Get the code and data.** On the cluster: `git clone https://github.com/JackHanni/AliasingDesign.jl.git`.
   `experiments/data/` is not in git; from your own machine:
   `rsync -av experiments/data/ <user>@ssh.rc.byu.edu:AliasingDesign.jl/experiments/data/`.
3. **Install packages on the login node** (compute nodes have no internet), from the repository root:
   `module load julia/1.12.6-dnqrzwe`, then
   `julia --project=experiments -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'`.
4. **Submit** from the repository root, one job per experiment:

   ```
   sbatch experiments/slurm/run.sh experiments/aspect_ratio.jl
   sbatch experiments/slurm/run.sh experiments/cluster_expansion_paper.jl
   sbatch --time 06:00:00 --cpus-per-task 32 --mem 32G \
          --export=ALL,P_GRID=all experiments/slurm/run.sh experiments/cluster_expansion_paper.jl
   sbatch --export=ALL,TRUTH=unit experiments/slurm/run.sh experiments/aspect_ratio.jl
   ```

   `squeue -u $USER` lists your jobs, `scancel <jobid>` cancels one, and the log is
   `experiments/output/slurm-<jobid>.out`. For a quick test (≤ 1 h), add `--qos test`.
5. **Copy results back:** `rsync -av <user>@ssh.rc.byu.edu:AliasingDesign.jl/experiments/output/ experiments/output/`.

Login nodes are for editing and file transfer only: anything using more than an hour of CPU there is
killed. Outputs are small, so they can stay in the home directory (backed up).

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

- Two risks, both on the held-out structures. **Expected risk** is the GAD's prediction from the
  design alone (prior-averaged, no labels; what the optimizer minimizes). **Held-out risk** is
  measured: fit synthetic energies on the training set, mean squared error on the held-out set,
  averaged over synthetic truths. With `TRUTH=prior` the truths come from the design's own prior, so
  the two agree on average by construction (checked: within 0.5% over 20,000 truths) and the plot
  shows the spread. `TRUTH=unit` draws unit-variance ECIs (JuCE's synthetic setup), a truth the prior
  did not anticipate: whether an optimized design still wins there is the real test.
  `cluster_expansion_paper.jl` does both truth kinds in one run by default (`TRUTH=prior,unit`):
  the design-only quantities are computed once.

- Costs at τ = 0 can be degenerate (plan, *Findings*): monomial aliasing collapses toward t = 0,
  and Frobenius risk is minimized by sampling the prediction distribution. The scripts report a
  range of τ for that reason.
- Random designs with n = p can be nearly singular, so their upper quantiles can be huge (e.g. the
  90th percentile at p = n = 10 in the cluster-expansion risk-vs-p table). Medians are more stable.
- The relaxation gap in the cluster expansion bounds the distance to the best n-structure design
  only if the relaxed minimum is global; the cost is not convex, so treat it as empirical.
- The capped relaxation is degenerate for the cluster expansion with the pool as prediction set:
  uniform weights on the whole pool already reach the floor (the fit on every structure), so the
  Fisher–Rao weights stay uniform and top-n rounding picks arbitrary structures. Tight budgets need
  the discrete steps instead: `greedy_select` and `exchange(...; fast = true)` (`ce_budget.jl`).
