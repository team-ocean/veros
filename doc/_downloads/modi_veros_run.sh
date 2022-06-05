#!/bin/bash -l
#
# Defines where the package should be installed.
# Since the ~/modi_mount directory content is
# available on each node, we define the package(s) to be installed
# here so that the node can find it once the job is being executed.
export CONDA_PKGS_DIRS=~/modi_mount/conda_dir

# Activate conda in your $PATH
# This ensures that we discover every conda environment
# before we try to activate it.
source $CONDA_DIR/etc/profile.d/conda.sh

# Check if the directory with package(s) exists, otherwise create it
if [ ! -d $CONDA_PKGS_DIRS ]; then
        mkdir $CONDA_PKGS_DIRS
fi

# As per https://veros.readthedocs.io/en/latest/introduction/get-started.html#installation
# we download and install the Veros environment unless it is already available
if [ ! -d "veros" ]
then
      git clone https://github.com/team-ocean/veros.git -b v0.2.3
else
      echo "Veros source directory exists"
      echo "Make sure it contains the rigth model version!"
fi

# Either activate the existing veros environment or create a new one
conda activate veros
if [ $? != 0 ]; then
      conda create -n veros -y python=3.7
      conda activate veros

      # Install the packages into the conda environment that was activated.
      pip3 install ./veros
      pip3 install veros-bgc
fi

# Either change your current directory to already available Veros-BGC setup
# or create a new one
cd bgc_global_4deg
if [ $? != 0 ]; then
      veros copy-setup bgc_global_4deg
      cd bgc_global_4deg
fi

# Change the default model run length from 0 to 10 yrs
sed -i 's/vs.runlen = 0/vs.runlen = 86400.0 * 360 * 10/g' bgc_global_four_degree.py

# Run your setup in the current directory (bgc_global_4deg)
python3 bgc_global_four_degree.py

