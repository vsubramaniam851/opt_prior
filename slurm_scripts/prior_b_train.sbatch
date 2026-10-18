#!/usr/bin/env bash
#SBATCH --job-name=prior-b-train
#SBATCH --account=aia
#SBATCH --partition=aia-h200
#SBATCH --qos=aia-main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-5%4
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err
#SBATCH --export=ALL
#SBATCH --no-requeue

set -eo pipefail
python3 -m slurm_scripts.run_prior_array B
