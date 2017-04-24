Installation
============

Prerequisites
-------------



Using Anaconda (multi-platform)
-------------------------------

1. `Download and install Anaconda <https://www.continuum.io/downloads>`_. Make sure to
   grab the 64-bit version of the Python 2.7 interpreter.

2. Install some dependencies:

    conda install -c bohrium bohrium
    conda install libhdf5 libnetcdf

3. Clone our repository:

    git clone https://github.com/dionhaefner/veros.git

4. Install Veros via

    conda develop ./veros


Ubuntu / Debian
---------------

1. Install some dependencies:

    sudo apt install git python-dev python-pip libhdf5-dev libnetcdf-dev

   and optionally

    sudo apt install bohrium

2. Clone our repository:

    git clone https://github.com/dionhaefner/veros.git

3. Install Veros via

    pip install -e ./veros
