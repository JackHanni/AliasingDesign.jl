# AliasingDesign.jl

Julia package for optimal experimental design under GAD aliasing (class project). The plan and
derivations are in docs/project_plan.md, experiments in experiments/ (see experiments/README.md).

## On the BYU cluster

Follow BYU's agent rules: @/apps/instructions_for_ai_agents/BYU_ORC_AGENTS.md

- Run experiments as Slurm jobs, never on login nodes: `sbatch experiments/slurm/run.sh
  experiments/<script>.jl` from the repository root (usage and settings in that file).
- Outputs land in experiments/output/, which is git-ignored on purpose.

## Sending results back to the laptop

`origin` pushes over a deploy key (ssh alias `github-aliasing`). When a job finishes, push its
outputs to the `results` branch, keeping `main` free of outputs:

```
git checkout -B results
git add -f experiments/output
git commit -m "Results: <experiment> <date>"
git push -f origin results
git checkout main
```

Then tell the user the results are pushed; Claude on the laptop pulls them from there.

Never commit experiments/data/: it is a private JuCE export and this repository is public.
