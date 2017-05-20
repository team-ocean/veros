Benchmarks
==========

.. note::

   The following benchmarks are for general orientation only. Since Bohrium is still in heavy development, we expect to be able to increase performance for all problem sizes (especially with the ``opencl`` backend). Benchmark results are highly platform dependent; your mileage may vary.

The following figures present some benchmarks that compare the performance of Veros and pyOM 2.1 depending on the problem size. In these figures, "component" refers to the used backend:

- ``numpy``: Veros called with ``-b numpy`` (default backend)
- ``bohrium``: Veros called with ``-b bohrium``
- ``bohrium-opencl``: Veros called with ``-b bohrium`` and ``BH_STACK=opencl``
- ``fortran``: Veros using the pyOM 2.1 Fortran library for computations
- ``fortran-mpi``: Veros using the pyOM 2.1 Fortran library with MPI support for computations (one process per CPU)

Each simulation has been run for 100 time steps, changing only the number of total elements in the domain.

.. figure:: /_images/benchmarks/desktop-bench.png

   Benchmarks on a Desktop PC with an Intel i7 6700 Processor (4 cores), 16GB of memory, an SSD, and a NVIDIA GeForce GTX 1050Ti GPU (4GB memory). Running on Ubuntu 17.04.

.. figure:: /_images/benchmarks/aegir-bench.png

   Benchmarks on a cluster node with 16 CPUs and 64GB of memory.
