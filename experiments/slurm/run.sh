#!/bin/bash
# Run one experiment script as a batch job on BYU Research Computing (Slurm).
# Follows BYU ORC's rules for agents and jobs: see BYU_ORC_AGENTS.md at the repository root.
#
#   sbatch experiments/slurm/run.sh experiments/cluster_expansion_paper.jl
#   sbatch experiments/slurm/run.sh experiments/aspect_ratio.jl experiments/cluster_expansion_paper.jl
#   sbatch --time 06:00:00 --cpus-per-task 32 --mem 32G \
#          --export=ALL,P_GRID=all experiments/slurm/run.sh experiments/cluster_expansion_paper.jl
#
# Several scripts run one after another in the same allocation, so short experiments can share a job
# (BYU asks for jobs of at least 10–30 minutes). Submit from the repository root, after `julia --project=experiments -e 'using Pkg;
# Pkg.instantiate(); Pkg.precompile()'` on a login node: compute nodes have no internet access.
# Settings (N_TRAIN, TRUTH, P_GRID, …) pass through --export or the environment of the submitting
# shell. Defaults below suit the default p grid; raise --time and --cpus-per-task for P_GRID=all.
#
#SBATCH --job-name aliasing-design
#SBATCH --time 01:00:00
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 16G
#SBATCH --output experiments/output/slurm-%j.out

set -euo pipefail
[ $# -ge 1 ] || { echo "usage: sbatch experiments/slurm/run.sh experiments/<script>.jl [more.jl ...]"; exit 1; }

module purge
module load julia/1.12.6-dnqrzwe          # `module spider julia` lists the versions on the cluster
export GKSwstype=100                     # Plots/GR: draw off-screen (no display on compute nodes)
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p experiments/output
for SCRIPT in "$@"; do
    echo "job $SLURM_JOB_ID on $(hostname): $SCRIPT with $SLURM_CPUS_PER_TASK threads ($(date))"
    julia --project=experiments "$SCRIPT"
done
