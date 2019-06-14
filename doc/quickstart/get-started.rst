Getting started
===============

Installation
------------

Using pip (multi-platform)
++++++++++++++++++++++++++

.. note::

  You should only install Veros via pip if you want to get going as quickly as possible,
  and do not plan to access or modify the Veros source code. The prefered way to install Veros
  is through Anaconda (see below).

If you already have Python installed, the quickest way to get a working Veros installation
is to run::

  $ pip install veros --user

and optionally::

  $ pip install bohrium --user

to use Veros with Bohrium (Linux and OSX only).


Using Anaconda (multi-platform, recommended)
++++++++++++++++++++++++++++++++++++++++++++

1. `Download and install Anaconda <https://www.continuum.io/downloads>`_. Make sure to
   grab the 64-bit version of the Python interpreter.

2. Clone the Veros repository: ::

      $ git clone https://github.com/dionhaefner/veros.git

2. Create a new conda environment for Veros, and install all relevant dependencies,
   by running ::

       $ conda env create -f environment-unix.yml

   on Linux and OSX, or ::

       $ conda env create -f environment-windows.yml

  on Windows.

3. To use Veros, just activate your new conda environment! This can be done through either
   :program:`conda activate veros`, :program:`source activate veros`, or :program:`activate veros`,
   depending on your platform and Anaconda installation.


On bare metal (Ubuntu / Debian)
+++++++++++++++++++++++++++++++

1. Install some dependencies: ::

      $ sudo apt-get install git python3-dev python3-pip libhdf5-dev

2. Clone our repository: ::

      $ git clone https://github.com/dionhaefner/veros.git

3. Install Veros (preferably in a virtual environment) via::

      $ pip3 install -e ./veros --user

4. Optionally, install Bohrium via::

      $ pip3 install bohrium --user


Setting up a model
------------------

To run Veros, you need to set up a model - i.e., specify which settings and model domain you want to use. This is done by subclassing the :class:`Veros setup base class <veros.VerosSetup>` in a *setup script* that is written in Python. You should have a look at the pre-implemented model setups in the repository's :file:`setup` folder, or use the :command:`veros copy-setup` command to copy one into your current folder. A good place to start is the :class:`ACC model <acc.ACCSetup>`: ::

    $ veros copy-setup acc

By working through the existing models, you should quickly be able to figure out how to write your own simulation. Just keep in mind this general advice:

- You can (and should) use any (external) Python tools you want in your model setup. Before implementing a certain functionality, you should check whether it is already provided by a common library. Especially `the SciPy module family <https://www.scipy.org/>`_ provides countless implementations of common scientific functions (and SciPy is installed along with Veros).

- If you decorate your methods with :func:`@veros_method <veros.veros_method>`, the variable :obj:`np` inside that function will point to the currently used backend (i.e., NumPy or Bohrium). Thus, if you want your setup to be able to dynamically switch between backends, you should write your methods like this: ::

      from veros import Veros, veros_method

      class MyVerosSetup(Veros):
          ...
          @veros_method
          def my_function(self):
              arr = np.array([1,2,3,4]) # "np" uses either NumPy or Bohrium

- If you are curious about the general procedure in which a model is set up and ran, you should read the source code of :class:`veros.VerosSetup` (especially the :meth:`setup` and :meth:`run` methods). This is also the best way to find out about the order in which methods and routines are called.

- Out of all functions that need to be implemented by your subclass of :class:`veros.VerosSetup`, the only one that is called in every time step is :meth:`set_forcing` (at the beginning of each iteration). This implies that, to achieve optimal performance, you should consider moving calculations that are constant in time to other functions.

If you want to learn more about setting up advanced configurations, you should :doc:`check out our tutorial </tutorial/wave-propagation>` that walks you through the creation of a realistic configuration with an idealized Atlantic.

Running Veros
-------------

After adapting your setup script, you are ready to run your first simulation. It is advisable to include something like::

   @veros.tools.cli
   def run(*args, **kwargs):
       simulation = MyVerosSetup()
       simulation.setup()
       simulation.run()

   if __name__ == "__main__":
       run()


in your setup file, so you can run it as a script: ::

   $ python my_setup.py

However, you are not required to do so, and you are welcome to write include your simulation class into other Python files and call it dynamically or interactively (e.g. in an IPython session).

All Veros setups decorated with :func:`veros.tools.cli` accept additional options via the command line when called as a script or as arguments to their :func:`__init__` function when called from another Python module. You can check the available commands through ::

   $ python my_setup.py --help

Reading Veros output
++++++++++++++++++++

All output is handled by :doc:`the available diagnostics </reference/diagnostics>`. The most basic diagnostic, snapshot, writes :doc:`some model variables </reference/variables>` to netCDF files in regular intervals (and puts them into your current working directory).

NetCDF is a binary format that is widely adopted in the geophysical modeling community. There are various packages for reading, visualizing and processing netCDF files (such as `ncview <http://meteora.ucsd.edu/~pierce/ncview_home_page.html>`_ and `ferret <http://ferret.pmel.noaa.gov/Ferret/>`_), and bindings for many programming languages (such as C, Fortran, MATLAB, and Python).

In fact, after installing Veros, you will already have installed the netCDF bindings for Python, so reading data from an output file and plotting it is as easy as::

   import matplotlib.pyplot as plt
   import h5netcdf

   with h5netcdf.File("veros.snapshot.nc", "r") as datafile:
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

For large simulations, it is often beneficial to use the Bohrium backend for computations. When using Bohrium, all number crunching will make full use of your available architecture, i.e., computations are executed in parallel on all of your CPU cores, or even GPU when using :envvar:`BH_STACK=opencl` or :envvar:`BH_STACK=cuda`. You may switch between NumPy and Bohrium with a simple command line switch: ::

   $ python my_setup.py -b bohrium

or, when running inside another Python module: (must be done before initializing you setup)::

   from veros import runtime_settings as rs

   rs.backend = "bohrium"

Re-starting from a previous run
+++++++++++++++++++++++++++++++

Restart data (in HDF5 format) is written at the end of each simulation or after a regular time interval if the setting :ref:`restart_frequency <setting-restart_frequency>` is set to a finite value. To use this restart file as initial conditions for another simulation, you will have to point :ref:`restart_input_filename <setting-restart_input_filename>` of the new simulation to the corresponding restart file. This can (as all settings) also be given via command line: ::

   $ python my_setup.py -s restart_input_filename /path/to/restart_file.h5

Running Veros on multiple processes via MPI
+++++++++++++++++++++++++++++++++++++++++++

.. note::

  This assumes that you are familiar with running applications through MPI, and is most useful on large architectures like a compute cluster. For smaller architectures, it is usually easier to stick to Bohrium.

Running Veros through MPI requires some addititonal dependencies:

- A recent MPI implementation, such as OpenMPI or MPICH
- ``mpi4py`` that is linked to the correct MPI library
- A parallel-enabled version of the HDF5 library
- ``h5py`` built against this parallel version of HDF5
- For optimal performance, PETSc and ``petsc4py``, linked to the rest of the stack

After you have installed everything, you can start Veros on multiple processes like so:::

   $ mpirun -n 4 python my_setup.py -n 2 2

In this case, Veros would run on 4 processes, each process computing one-quarter of the domain. The arguments of the `-n` flag specify the number of chunks in x and y-direction, respectively.

You can combine MPI and Bohrium like so:::

   $ OMP_NUM_THREADS=2 mpirun -n 2 python my_setup.py -n 2 1 -b bohrium

This starts 2 independent processes, each being parallelized by Bohrium using 2 threads (hybrid run).

.. seealso::

   For more information, see :doc:`/tutorial/cluster`.

Enhancing Veros
---------------

Veros was written with extensibility in mind. If you already know some Python and have worked with NumPy, you are pretty much ready to write your own extension. The model code is located in the :file:`veros` subfolder, while all of the numerical routines are located in :file:`veros/core`.

We believe that the best way to learn how Veros works is to read its source code. Starting from the :py:class:`Veros base class <veros.VerosSetup>`, you should be able to work your way through the flow of the program, and figure out where to add your modifications. If you installed Veros through :command:`pip -e` or :command:`setup.py develop`, all changes you make will immediately be reflected when running the code.

In case you want to add additional output capabilities or compute additional quantities without changing the main solution of the simulation, you should consider :doc:`adding a custom diagnostic </reference/diagnostics>`.

A convenient way to implement your modifications is to create your own fork of Veros on GitHub, and submit a `pull request <https://github.com/dionhaefner/veros/pulls>`_ if you think your modifications could be useful for the Veros community.

.. seealso::

   More information is available in :doc:`our developer guide </tutorial/dev>`.
