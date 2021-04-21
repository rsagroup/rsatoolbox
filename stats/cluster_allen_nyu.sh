#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=3500MB
#SBATCH --job-name=allen
#SBATCH --mail-type=END
#SBATCH --mail-user=hhs4@nyu.edu
#SBATCH --output=/scratch/hhs4/pyrsa/stats/slurm-output/slurm_allen_%a.out

index=$SLURM_ARRAY_TASK_ID
job=$SLURM_JOB_ID
ppn=$SLURM_JOB_CPUS_PER_NODE

module purge
module load anaconda3/2020.07

singularity exec --overlay /scratch/hhs4/anaconda/overlay-25GB-500K.ext3 \
    /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "source ~/.bashrc;
        conda activate rsa;
        pwd;
        which python;
        python allen_stats.py run"
