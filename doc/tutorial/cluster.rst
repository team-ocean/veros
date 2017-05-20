Running Veros on a cluster
==========================

.. note::

   Since Bohrium does not (yet) support distributed memory architectures, Veros is currently limited to running on a single computational node.

This tutorial walks you through some of the most common challenges that are specific to large, shared architectures like clusters and supercomputers. In case you are still having trouble setting up or running Veros on a large architecture after reading it, you should first contact the administrator of your cluster. Otherwise, you should of course feel free to `open an issue <https://github.com/dionhaefner/veros/issues>`_.

Installation
++++++++++++

Probably the easiest way to install Veros on a cluster is to, once again, :doc:`use Anaconda </quickstart/get-started>`. Since it is mostly platform independent and does not require elevated permissions, Anaconda is the perfect way to try out Veros without too much hassle.

If you are an administrator and want to make Veros accessible to multiple users on your cluster, we recommend that you do *not* install Veros system-wide, since it severely limits the possibilities of the users: First of all, they won't be able to install additional Python modules they might want to use for post-processing or development. And second of all, the source code (and playing with it) is supposed to be a critical part of the Veros experience. Instead, you could e.g. use `virtualenv <https://virtualenv.pypa.io/en/stable/>`_ to create a lightweight Python environment for every user that they can freely manage.

Usage
+++++

If you want to run Veros on a shared computing architecture, there are several issues that require special handling:

1. **Preventing timeouts**: In cloud computing, it is common that scheduling constraints limit the maximum execution time of a given process. Processes that exceed this time are killed. To prevent that long-running processes have to be restarted manually after each timeout, one usually makes use of a *resubmit* mechanism: The long-running process is split into chunks that each finish before a timeout is triggered, with subsequent runs starting from the restart files that the previous process has written.

2. **Allocation of resources**: Most applications use MPI to distribute work across processors; however, this is not supported by Bohrium. We therefore need to make sure that just one single process on a single node is started for our simulation (Bohrium will then divide the workload among different threads using OpenMP).

To solve these issues, the scheduling manager needs to be told exactly how it should run our model, which is usually being done by writing a batch script that prepares the environment and states which resources to request. The exact set-up of such a script will vary depending on the scheduling manager running on your cluster, and how exactly you chose to install Veros and Bohrium. One possible way to write such a batch script for the scheduling manager SLURM is presented here:

.. literalinclude:: /_downloads/veros_batch.sh
   :language: bash

which is :download:`saved as veros_batch.sh </_downloads/veros_batch.sh>` in the model setup folder and called using ``sbatch``.

This script makes use of the :command:`veros resubmit` command and its :option:`--callback` option to create a script that automatically re-runs itself in a new process after each successful run (see also :doc:`/reference/cli`). Upon execution, a job is created on one node, using 16 processors in one process, that runs the Veros setup located in :file:`my_setup.py` a total of eight times for 90 days (7776000 seconds) each, with identifier ``my_run``. Note that the ``--callback "sbatch veros_batch.sh"`` part of the command is needed to actually create a new job after every run, to prevent the script from being killed after a timeout.
