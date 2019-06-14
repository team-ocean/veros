Benchmarks
==========

.. note::

   The following benchmarks are for general orientation only. Benchmark results are highly platform dependent; your mileage may vary.

Veros compared to PyOM2
-----------------------

The following figurew present some benchmarks that compare the performance of Veros and pyOM 2.1 depending on the problem size:

Without MPI
+++++++++++

.. figure:: /_images/benchmarks/acc.svg
   :width: 600px

   Benchmarks on a Desktop PC with 4 CPU cores (I) and a cluster node with 24 CPU cores and an NVidia Tesla P100 GPU (II). Line fits suggest a linear scaling with constant overhead for all components.

With MPI
++++++++

.. figure:: /_images/benchmarks/acc-4node.png
   :width: 600px

   Benchmarks on 4 cluster nodes with 32 CPUs each (128 processes in total).


Veros runtime estimates on the `DC3 <http://nutrik.dk/dc3.html>`_ cluster
-------------------------------------------------------------------------

The following figure presents estimates of Veros runtime required for 1 year prediction on CPU and GPU nodes of the DC3 cluster.
The estimates were done for ACC, Wave propagation and Global cases with different spatial and temporal resolution.

For more details on the cases configuration see :doc:`the setup gallery </reference/setup-gallery>`.

.. figure:: /_images/benchmarks/veros_dc3perf.svg
   :width: 600px

   Runtime estimates on a cluster node with 32 CPU cores (Bh CPU) and on a cluster node with 24 CPU cores and an NVidia Tesla P100 GPU (Bh GPU).
