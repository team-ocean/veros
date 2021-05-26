#!/bin/bash -l
#
#SBATCH -p mycluster
#SBATCH -A myaccount
#SBATCH --job-name=veros_mysetup
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email.xyz

# load module dependencies
# module load petsc4py mpi4py h5py ...

export OMP_NUM_THREADS=1

# adapt srun command to your available scheduler / MPI implementation
veros resubmit -i my_run -n 8 -l 7776000 \
    -c "srun --mpi=pmi2 -- veros run my_setup.py -b jax -n 4 4" \
    --callback "sbatch veros_batch.sh"
