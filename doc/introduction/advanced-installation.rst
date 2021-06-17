Advanced installation
=====================

Because Veros should be usable by both beginners and experts, it has a number of *optional* dependencies that are not strictly required to run Veros, but highly recommended in high-performance contexts.

.. note::

    Veros notifies you when a slower code path has to be taken due to a missing dependency. So unless you are getting a warning, there is usually no need to install optional dependencies (except JAX and MPI).


.. list-table:: Summary of optional dependencies
    :widths: auto
    :width: 100%
    :header-rows: 1

    * - Dependency
      - Supported platforms
      - When to use
    * - JAX
      - Linux, OSX, Windows
      - To run 4x faster on CPU, and for GPU support
    * - Veros Cython extensions
      - Linux, OSX
      - 20% speedup when using JAX
    * - MPI + mpi4py
      - Linux, OSX
      - To run in parallel
    * - mpi4jax
      - Linux, OSX
      - To run in parallel with JAX
    * - PETSc + petsc4py
      - Linux, OSX
      - | Faster linear solver when using
        | more than 10 processes (or GPUs)

.. note::

    On this page, we give all installation instructions via ``pip install``. If you used conda to install Veros, consider replacing them with ``conda install``.

Using JAX
---------

Using the JAX backend, Veros is typically about 4x faster than with NumPy, so this should be the first thing to try if you want to get more performance.

JAX is available on all major platforms and can be installed via::

   $ pip install jax jaxlib

To use JAX on GPU, you have to install a CUDA-enabled version of jaxlib, e.g.::

   $ pip install jax jaxlib==0.1.67+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html

(see also `the JAX installation guide <https://github.com/google/jax#installation>`__).

Veros also supplies Cython extensions that optimize certain bottlenecks in JAX. You can make sure they are installed by running::

   $ pip install cython
   $ python setup.py build_ext --inplace

in the Veros repository root.

Using MPI
---------

To run Veros on more than one process you need to use MPI. This requires that you install an MPI implementation (such as OpenMPI) on your system. Additionally, you have to install ``mpi4py`` to interface with it::

   $ pip install mpi4py

Then, you can :ref:`run Veros in parallel via MPI <mpi-exec>`.

For optimal performance on many processes, Veros supports using PETSc as a linear solver. To use it, you will have to install the PETSc library and ``petsc4py`` Python package::

   $ PETSC_DIR=/path/to/petsc3.12 pip install petsc4py==3.12

Note that the versions of PETSc and ``petsc4py`` have to match.


Using JAX + MPI
---------------

To use JAX together with MPI, you need to install ``mpi4jax`` after installing ``mpi4py``::

   $ pip install mpi4jax
