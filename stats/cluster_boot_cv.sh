#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
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
    /scratch/work/public/singularity/centos-7.8.2003.sif \
    /bin/bash -c "conda activate /ext3/.env/rsa \
                  python stats.py -p /scratch/hhs4/ecoset/val boot_cv 1"
