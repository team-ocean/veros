Getting started
===============

Installation
------------

Using Anaconda (multi-platform)
+++++++++++++++++++++++++++++++

1. `Download and install Anaconda <https://www.continuum.io/downloads>`_. Make sure to
   grab the 64-bit version of the Python 2.7 interpreter.

2. Install some dependencies: ::

       $ conda install libhdf5 libnetcdf
       $ conda install -c conda-forge git-lfs

   and optionally::

       $ conda install -c bohrium bohrium

3. Clone our repository: ::

       $ git clone https://github.com/dionhaefner/veros.git

4. Install Veros via::

       $ conda develop ./veros


Using apt-get (Ubuntu / Debian)
+++++++++++++++++++++++++++++++

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


Setting up a model
------------------

To run Veros, you need to set up a model - i.e., specify which settings and model domain you want to use. This is done by subclassing the :class:`Veros base class <veros.Veros>` in a *setup script* that is written in Python. You should have a look at the pre-implemented model setups in the repository's :file:`setup` folder, or use the :command:`veros copy-setup` command to copy one into your current folder. A good place to start is the :class:`ACC2 model <acc2.ACC2>`.



Running Veros
-------------

After adapting your setup script, you are ready to run your first simulation. It is advisable to include something like::

   if __name__ == "__main__":
      simulation = MyVerosSetup()
      simulation.setup()
      simulation.run()

in your setup file, so you can run it as a script: ::

   $ python my_setup.py

However, you are not required to do so, and you are welcome to write include your simulation class into other Python files and call it dynamically or interactively (e.g. in an IPython session).

All Veros setups accept additional options via the command line when called as a script or as arguments to their :func:`__init__` function when called from another Python module. You can check the available commands through ::

   $ python my_setup.py --help

Reading Veros output
++++++++++++++++++++

All output is handled by :doc:`the available diagnostics <reference/diagnostics>`. The most basic diagnostic, snapshot, writes :doc:`some model variables <reference/variables>` to netCDF files in regular intervals (and puts them into your current working directory).

NetCDF is a binary format that is widely adopted in the geophysical modeling community. There are various packages for reading, visualizing and processing netCDF files (such as `ncview <http://meteora.ucsd.edu/~pierce/ncview_home_page.html>`_ and `ferret <http://ferret.pmel.noaa.gov/Ferret/>`_), and bindings for many programming languages (such as C, Fortran, MATLAB, and Python).

In fact, after installing Veros, you will already have installed the netCDF bindings for Python, so reading data from an output file and plotting it is as easy as::

   import matplotlib.pyplot as plt
   from netCDF4 import Dataset

   with Dataset("veros.snapshot.nc", "r") as datafile:
       # read variable "u" and save it to a NumPy array
       u = datafile.variables["u"][...]

   # plot surface velocity at the last time step included in the file
   plt.imshow(u[-1, -1, ...])
   plt.show()

For further reference refer to `the netcdf4-python documentation <http://unidata.github.io/netcdf4-python/>`_.

Using Bohrium
+++++++++++++

.. warning::

  While Bohrium yields significant speed-ups for large to very large setups, the overhead introduced by Bohrium often leads to (sometimes considerably) slower execution for problems below a certain threshold size (see also :ref:`when-to-use-bohrium`). You are thus advised to test carefully whether Bohrium is beneficial in your particular use case.

For large simulations, it is often beneficial to use the Bohrium backend for computations. When using Bohrium, all number crunching will make full use of your availble architecture, i.e., computations are executed in parallel on all of your CPU cores, or even GPU when using :envvar:`BH_STACK=opencl` (experimental). You may switch between NumPy and Bohrium with a simple command line switch: ::

   $ python my_setup.py -b bohrium

or, when running inside another Python module: ::

   simulation = MyVerosSetup(backend="bohrium")


Re-starting from a previous run
+++++++++++++++++++++++++++++++

Restart data (in HDF5 format) is written at the end of each simulation or after a regular time interval if the setting :ref:`restart_frequency <setting-restart_frequency>` is set to a finite value. To use this restart file as initial conditions for another simulation, you will have to point :ref:`restart_input_filename <setting-restart_input_filename>` of the new simulation to the corresponding restart file. This can (as all settings) also be given via command line: ::

   $ python my_setup.py -s restart_input_filename /path/to/restart_file.h5

Enhancing the model
-------------------

Veros was written with extensibility in mind. If you already know some Python and have worked with NumPy, you are pretty much ready to write your own extension. The model code is located in the :file:`veros` subfolder, while all of the numerical routines are located in :file:`veros/core`.

We believe that the best way to learn how Veros works is to read its source code. Starting from the :py:class:`Veros base class <veros.Veros>`, you should be able to work your way through the flow of the program, and figure out where to add your modifications. If you installed Veros through :command:`pip -e` or :command:`setup.py develop`, all changes you make will immediately be reflected when running the code.

In case you want to add additional output capabilities or compute additional quantities without changing the main solution of the simulation, you should consider :doc:`adding a custom diagnostic <reference/diagnostics>`.

Running tests and benchmarks
++++++++++++++++++++++++++++

If you want to make sure that your changes did not break anything, you can run our test suite that compares the results of each subroutine to pyOM2.
To do that, you will need to compile the Python interface of pyOM2 on your machine, and then point the testing suite to the library location, e.g. through::

   $ python run_tests.py /path/to/pyOM2/py_src/pyOM_code.so

from Veros's :file:`test` folder.

If you deliberately introduced breaking changes, you can disable them during testing by prefixing them with::

   if not veros.pyom_compatibility_mode:
       # your changes

Automated benchmarks are provided in a similar fashion. The benchmarks run some dummy problems with varying problem sizes and all available computational backends: ``numpy``, ``bohrium-openmp``, ``bohrium-opencl``, ``fortran`` (pyOM2), and ``fortran-mpi`` (parallel pyOM2). For options and further information run::

   $ python run_benchmarks.py --help

from the :file:`test` folder. Timings are written in JSON format.
