Running Veros on a cluster
==========================

.. note::

   Since Bohrium does not (yet) support distributed memory architectures, Veros is currently limited to running on a single computational node.

In cloud computing, it is common that scheduling constraints limit the maximum execution time of a given process. Processes that exceed this time are killed. To prevent that long-running processes have to be restarted manually after each timeout, one usually makes use of a "resubmit" mechanism: The long-running process is split into chunks that each finish before a timeout is triggered, with subsequent runs starting from the restart files that the previous process has written.
