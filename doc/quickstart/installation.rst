Installation
============

Using Anaconda (multi-platform)
-------------------------------

1. `Download and install Anaconda <https://www.continuum.io/downloads>`_. Make sure to
   grab the 64-bit version of the Python 2.7 interpreter.

2. Install some dependencies:::

       $ conda install libhdf5 libnetcdf
       $ conda install -c conda-forge git-lfs

   and optionally::

       $ conda install -c bohrium bohrium

3. Clone our repository: ::

       $ git clone https://github.com/dionhaefner/veros.git

4. Install Veros via::

       $ conda develop ./veros


Ubuntu / Debian
---------------

1. Install some dependencies: ::

      $ sudo apt-get install git python-dev python-pip libhdf5-dev libnetcdf-dev

   and optionally::

      $ sudo add-apt-repository ppa:bohrium/nightly
      $ sudo apt-get update
      $ sudo apt-get install bohrium

  If you want to clone the input files needed for running the larger setups, you will
  also need to `install git lfs <https://git-lfs.github.com/>`_.

2. Clone our repository: ::

      $ git clone https://github.com/dionhaefner/veros.git

3. Install Veros via::

      $ pip install -e ./veros
