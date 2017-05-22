#!/bin/bash
#
#SBATCH -p mycluster
#SBATCH -A myaccount
#SBATCH --job-name=veros_mysetup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email.xyz

# only needed if not found automatically
export BH_CONFIG=/path/to/bohrium/config.ini

# removes "-march=native" from compiler flags, not always needed
export BH_OPENMP_COMPILER_FLG="-x c -fPIC -shared -std=gnu99 -O3 -Werror -fopenmp"

# only needed if working with a virtual python environment
source $HOME/venvs/veros/bin/activate

veros resubmit my_run 8 7776000 "python my_setup.py -b bohrium -v debug" --callback "sbatch veros_batch.sh"
