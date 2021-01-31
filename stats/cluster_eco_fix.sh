#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=eco_fix
#SBATCH --mail-type=END
#SBATCH --mail-user=hhs4@nyu.edu
#SBATCH --output=slurm-output/slurm_fix_%a.out

index=$SLURM_ARRAY_TASK_ID
job=$SLURM_JOB_ID
ppn=$SLURM_JOB_CPUS_PER_NODE

module purge
module load anaconda3/5.3.1

source activate RSA

which python

~/.conda/envs/RSA/bin/python stats.py -p /scratch/hhs4/ecoset/val fix_eco $index
