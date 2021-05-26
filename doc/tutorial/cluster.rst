Running Veros on a cluster
==========================

This tutorial walks you through some of the most common challenges that are specific to large, shared architectures like clusters and supercomputers.

In case you having trouble setting up or running Veros on a cluster, you should first contact your cluster administrator. Otherwise, feel free to `open an issue <https://github.com/team-ocean/veros/issues>`__.

Installation
++++++++++++

Probably the easiest way to try out Veros on a cluster is to, once again, :doc:`use Anaconda </introduction/get-started>`. Since Anaconda is platform independent and does not require elevated permissions, it is the perfect way to try out Veros without too much hassle.

However, **in high-performance contexts, we advise against using Anaconda**. Getting optimal performance requires a software stacked that is linked to the correct system libraries, in particular MPI (see also :doc:`/introduction/advanced-installation`). This requires that Python packages that depend on C libraries (such as ``mpi4py``, ``mpi4jax``, ``petsc4py``) are built from source, e.g. via ``pip install --no-binary``.

Usage
+++++

Your cluster's scheduling manager needs to be told exactly how it should run our model, which is usually being done by writing a batch script that prepares the environment and states which resources to request. The exact set-up of such a script will vary depending on the scheduling manager running on your cluster, and how exactly you chose to install Veros. One possible way to write such a batch script for the scheduling manager SLURM is presented here:

.. literalinclude:: /_downloads/veros_batch.sh
   :language: bash

which is :download:`saved as veros_batch.sh </_downloads/veros_batch.sh>` in the model setup folder and called using ``sbatch``.

This script makes use of the ``veros resubmit`` command and its ``--callback`` option to create a script that automatically re-runs itself in a new process after each successful run (see also :doc:`/reference/cli`). Upon execution, a job is created on one node, using 16 processors in one process, that runs the Veros setup located in :file:`my_setup.py` a total of eight times for 90 days (7776000 seconds) each, with identifier ``my_run``. Note that the ``--callback "sbatch veros_batch.sh"`` part of the command is needed to actually create a new job after every run, to prevent the script from being killed after a timeout.
