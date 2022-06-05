#!/bin/bash -l
#
#SBATCH -p modi_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=verosbgc

echo started at `date`

srun -N 1 -n 1 --exclusive singularity exec ~/modi_images/hpc-ocean-notebook_latest.sif ~/modi_mount/modi_veros_run.sh

echo finished at `date`

