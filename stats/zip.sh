#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=2000MB
#SBATCH --job-name=zip_eco
#SBATCH --mail-type=END
#SBATCH --mail-user=hhs4@nyu.edu
#SBATCH --output=/scratch/hhs4/pyrsa/stats/slurm-output/slurm_zip.out


zip -rvmT sim_cell.zip sim_cell
