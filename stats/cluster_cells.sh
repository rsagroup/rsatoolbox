#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=cells
#SBATCH --mail-type=END
#SBATCH --mail-user=hhs4@nyu.edu
#SBATCH --output=slurm-output/slurm_cell_%a.out

index=$SLURM_ARRAY_TASK_ID
job=$SLURM_JOB_ID
ppn=$SLURM_JOB_CPUS_PER_NODE

module purge
module load anaconda3/2020.07

singularity exec --overlay /scratch/hhs4/anaconda/overlay-25GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "source ~/.bashrc;
        conda activate rsa;
        pwd;
        which python;
        python allen_stats.py cells"
