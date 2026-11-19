#!/bin/bash
# Run one experiment script as a batch job on BYU Research Computing (Slurm).
#
#   sbatch experiments/slurm/run.sh experiments/cluster_expansion_paper.jl
#   sbatch --time 04:00:00 --cpus-per-task 32 --mem 32G \
#          --export=ALL,P_GRID=all experiments/slurm/run.sh experiments/cluster_expansion_paper.jl
#
# Submit from the repository root, after `julia --project=experiments -e 'using Pkg;
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
SCRIPT=${1:?usage: sbatch experiments/slurm/run.sh experiments/<script>.jl}

module purge
module load julia/1.12.6-dnqrzwe          # `module spider julia` lists the versions on the cluster
export GKSwstype=100                     # Plots/GR: draw off-screen (no display on compute nodes)
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p experiments/output
echo "job $SLURM_JOB_ID on $(hostname): $SCRIPT with $SLURM_CPUS_PER_TASK threads"
julia --project=experiments "$SCRIPT"
