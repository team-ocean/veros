Benchmarks
==========

.. note::

   The following benchmarks are for general orientation only. Benchmark results are highly platform dependent; your mileage may vary.

The following figure presents some benchmarks that compare the performance of Veros and pyOM 2.1 depending on the problem size:

.. figure:: /_images/benchmarks/acc.svg
   :width: 600px

   Benchmarks on a Desktop PC with 4 CPU cores (I) and a cluster node with 24 CPU cores and an NVidia Tesla P100 GPU (II). Line fits suggest a linear scaling with constant overhead for all components.

Veros runtime estimates on `DC3 <http://nutrik.dk/dc3.html>`_ cluster
--------------------------------------------------------------------

The following figure presents estimates of Veros runtime required for 1 year prediction on CPU and GPU nodes of DC3 cluster.
The estimates were done for ACC, Wave propagation and Global cases with different spatial and temporal resolution showen in the table below.

.. list-table::
   :widths: 25 25 25 25 25
   :header-rows: 1

   * - Notation
     - Case
     - Horizontal grid
     - Vertical levels
     - Time step (s)
   * - ACC1DEG
     - ACC
     - :math:`1^o \times 1^o`
     - :math:`60`
     - :math:`3600`
   * - ACC1/4DEG
     - ACC
     - :math:`0.25^o \times 0.25^o`
     - :math:`60`
     - :math:`900`
   * - WP1DEG
     - Wave propagation
     - :math:`1.15^o \times 1^o`
     - :math:`60`
     - :math:`2700`
   * - WP1/2DEG
     - Wave propagation
     - :math:`1.15^o \times 0.5^o`
     - :math:`60`
     - :math:`2400`
   * - GL1DEG
     - Global
     - :math:`1^o \times 1^o`
     - :math:`115`
     - :math:`2700`

For more details on the cases configuration see :doc:`setup gallery </reference/setup>`.

.. figure:: /_images/benchmarks/veros_dc3perf.svg
   :width: 600px

   Runtime estimates on a cluster node with 32 CPU cores (Bh CPU) and on a cluster node with 24 CPU cores and an NVidia Tesla P100 GPU (Bh GPU).

