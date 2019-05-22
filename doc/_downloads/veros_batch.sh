#!/bin/bash -l
#
#SBATCH -p mycluster
#SBATCH -A myaccount
#SBATCH --job-name=veros_mysetup
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email.xyz

# load module dependencies
module load bohrium

# only needed if not found automatically
export BH_CONFIG=/path/to/bohrium/config.ini

# if needed, you can modify the internal Bohrium compiler flags
export BH_OPENMP_COMPILER_FLG="-x c -fPIC -shared -std=gnu99 -O3 -Werror -fopenmp"

# set number of threads to cpus-per-task
export OMP_NUM_THREADS=4

# adapt srun command to your available scheduler / MPI implementation
veros resubmit -i my_run -n 8 -l 7776000 \
    -c "srun --mpi=pmi2 -- python my_setup.py -b bohrium -v debug -n 4 4" \
    --callback "sbatch veros_batch.sh"
