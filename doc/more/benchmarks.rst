Benchmarks
==========

.. note::

   The following benchmarks are for general orientation only. Benchmark results are highly platform dependent; your mileage may vary.

The following figure presents some benchmarks that compare the performance of Veros and pyOM 2.1 depending on the problem size:

.. figure:: /_images/benchmarks/acc.svg
   :width: 600px

   Benchmarks on a Desktop PC with 4 CPU cores (I) and a cluster node with 24 CPU cores and an NVidia Tesla P100 GPU (II). Line fits suggest a linear scaling with constant overhead for all components.

Veros runtime estimates on `DC3 <http://nutrik.dk/dc3.html>` cluster
--------------------------------------------------------------------

The following tables present estimates of Veros runtime required for 1 year prediction on CPU and GPU nodes of DC3 cluster.
The estimates were done for ACC, Wave propagation and Global cases with different spatial and temporal resolution.
For details of the cases configuration see :doc:`/reference/setup.rst`

In the tables below, HGrid is a horizontal grid resolution in degrees and VGrid is a number of vertical model levels.

CPU node - Intel Xeon E5-2683v4 2.1GHz (32 cores)

   | Case                      | Spatial-temporal resolution                           |                                                         |
   | ------------------------- | ----------------------------------------------------- | ------------------------------------------------------- |
   | ACC                       | HGrid: $1^o x 1^o$ / Time step: 3600 s / VGrid: 60    | HGrid: $0.25^o x 0.25^o$ / Time step: 900 s / VGrid: 60 |
   |                           | 2 hrs 20 min                                          | Not available                                           |
   |                           |                                                       |                                                         |
   | Wave propagation          | HGrid: $1.15^o x 1^o$ / Time step: 2700 s / VGrid: 60 | HGrid: $1.15^o x 0.5^o$ / Time step: 2400 s / VGrid: 60 |
   |                           | 7 hrs 35 min                                          | 17 hrs                                                  |
   |                           |                                                       |                                                         |
   | Global                    | HGrid: $1^o x 1^o$ / Time step: 2700 s / VGrid: 115   |                                                         |
   |                           | 12 hrs 18 min                                         |                                                         |
   |                           |                                                       |                                                         |

GPU node - NVidia Tesla P100

   | Case                      | Spatial-temporal resolution                           |                                                         |
   | ------------------------- | ----------------------------------------------------- | ------------------------------------------------------- |
   | ACC                       | HGrid: $1^o x 1^o$ / Time step: 3600 s / VGrid: 60    | HGrid: $0.25^o x 0.25^o$ / Time step: 900 s / VGrid: 60 |
   |                           | 2 hrs                                                 | 13 hrs                                                  |
   |                           |                                                       |                                                         |
   | Wave propagation          | HGrid: $1.15^o x 1^o$ / Time step: 2700 s / VGrid: 60 | HGrid: $1.15^o x 0.5^o$ / Time step: 2400 s / VGrid: 60 |
   |                           | 4 hrs 35 min                                          | 6 hrs                                                   |
   |                           |                                                       |                                                         |
   | Global                    | HGrid: $1^o x 1^o$ / Time step: 2700 s / VGrid: 115   |                                                         |
   |                           | 6 hrs 45 min                                          |                                                         |
   |                           |                                                       |                                                         |
