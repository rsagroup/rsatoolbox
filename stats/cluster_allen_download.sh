#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=allen_download
#SBATCH --mail-type=END
#SBATCH --mail-user=hs3110@columbia.edu
#SBATCH --output=slurm-output/slurm_allen_down_%a.out

index=$SLURM_ARRAY_TASK_ID
job=$SLURM_JOB_ID
ppn=$SLURM_JOB_CPUS_PER_NODE

module purge
module load anaconda3/5.3.1

source activate RSA

which python

~/.conda/envs/RSA/bin/python allen_stats.py download
