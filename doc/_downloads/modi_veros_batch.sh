#!/bin/bash -l
#
#SBATCH -p modi_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=verosbgc

echo started at `date`

singularity exec ~/modi_images/hpc-ocean-notebook_latest.sif bash -l ~/modi_mount/modi_veros_run.sh $1

echo finished at `date`

