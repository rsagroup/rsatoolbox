#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=allen
#SBATCH --mail-type=END
#SBATCH --mail-user=hs3110@columbia.edu
#SBATCH --output=/moto/nklab/users/hs3110/pyrsa/stats/slurm-output/slurm_allen_summary.out


module purge
module load shared anaconda/3-2019.10

source activate rsa

pwd
which python


~/.conda/envs/rsa/bin/python allen_stats.py summarize
