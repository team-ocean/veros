#!/bin/bash -l
#
# Activate conda in your $PATH
# This ensures that we discover every conda environment
# before we try to activate it.
source $CONDA_DIR/etc/profile.d/conda.sh

# Activate the existing veros environment
conda activate ~/modi_mount/conda-env-veros

# Change your current directory to available Veros-BGC setup
cd `dirname "$1"`

# Run your setup in the current directory (~/modi_mount/bgc_global_4deg)
python3 $1

