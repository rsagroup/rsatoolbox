#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --job-name=allen_download
#SBATCH --mail-type=END
#SBATCH --mail-user=hs3110@columbia.edu
#SBATCH --output=/moto/nklab/users/hs3110/pyrsa/stats/slurm-output/slurm_allen_down_%a.out

index=$SLURM_ARRAY_TASK_ID
job=$SLURM_JOB_ID
ppn=$SLURM_JOB_CPUS_PER_NODE

module purge
module load shared anaconda/3-2019.10

source activate rsa

pwd
which python


~/.conda/envs/rsa/bin/python allen_stats.py download
