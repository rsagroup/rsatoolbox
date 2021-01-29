#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=boot_cv
#SBATCH --mail-type=END
#SBATCH --mail-user=hhs4@nyu.edu
#SBATCH --output=slurm-output/slurm_boot_cv_%a.out

index=$SLURM_ARRAY_TASK_ID
job=$SLURM_JOB_ID
ppn=$SLURM_JOB_CPUS_PER_NODE

overlay_ext3=/scratch/hhs4/anaconda/overlay-25GB-500K.ext3

singularity \
    exec --overlay $overlay_ext3:ro \
    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
    /bin/bash -c "/ext3/miniconda3/bin/activate /ext3/.env/rsa/; \
                  /ext3/.env/rsa/bin/python stats.py -p /scratch/hhs4/ecoset/val boot_cv 1"
