ERDA installation
=================

Electronic Research Data Archive
--------------------------------

The Electronic Research Data Archive (`ERDA <https://www.erda.dk>`__) at University of Copenhagen (`KU/UCPH <https://www.ku.dk/english/>`__) is meant for storing, sharing, analyzing and archiving research data.
ERDA delivers safe central storage space for own and shared files, interactive analysis tools in addition to archiving for safe-keeping and publishing.

.. _jupyter services target:

Jupyter Data Analysis Services
------------------------------

ERDA integrates a set of `Jupyter <https://jupyter.org>`__ services, which can be used to easily perform a wide range of data analysis and visualization tasks directly on your ERDA data.
The system relies on the `JupyterLab <https://jupyterlab.readthedocs.io/en/stable/>`__ web interface to provide interactive and flexible notebooks or Linux command line (Terminal) with direct and efficient access to your ERDA home directory.
To get access to these services, ERDA provides a Jupyter button in the navigation menu.

.. figure:: /_images/tutorial/erda_welcome.png
   :width: 100%
   :align: center

   ERDA navigation menu.

Upon clicking it, the page to **Select a Jupyter Service** appears.
On this page you are presented with a set of horizontal service tabs at the top, and each tab presents and describes the individual service and how it is configured in the **Service Description**.
Below the description there is a **Start SERVICE** button, which you can click to open a connection to that particular service in a new web browser tab or window.

.. figure:: /_images/tutorial/erda_dag_spawn.png
   :width: 100%
   :align: center

   **Select a Jupyter Service** menu.

By default, it will take you to your personal home page on the **Jupyter service** as shown below, which is provided via a hosted version of JupyterHub.
That is, the standard infrastructure to provide individual isolated Jupyter notebook instances to multiple users sharing a pool of actual compute nodes.

.. figure:: /_images/tutorial/erda_jservice_homepage.png
   :width: 100%
   :align: center

   Top fragment of **Jupyter service** home page.

Upon clicking the **Start My Server**, the site will give you an option to chose (from dropdown menu), which Notebook image you want to Spawn.
Select **HPC Notebook** as shown below and press **Start** button.

.. figure:: /_images/tutorial/erda_dag_image.png
   :width: 100%
   :align: center

   Top fragment of **Jupyter service** home page with selected **HPC Notebook** image.

Upon spawning the **HPC Notebook** image, you will be redirected straight to the JupyterLab interface as shown below.
The JupyterLab interface is the same in all available Services (DAG and MODI).

.. figure:: /_images/tutorial/erda_dag_terminal.png
   :width: 100%
   :align: center

   JupyterLab interface on DAG.

Follow Veros installation instructions below with respect to selected Services.

Data Analysis Gateway (DAG)
+++++++++++++++++++++++++++

In order to install Veros on a DAG instance do the following after launching the **Terminal**:

1. Clone the Veros repository

   .. exec::

      from veros import __version__ as veros_version
      if "+" in veros_version:
          veros_version, _ = veros_version.split("+")
      print(".. code-block::\n")
      print(f"   $ git clone https://github.com/team-ocean/veros.git -b v{veros_version}")

   (or `any other version of Veros <https://github.com/team-ocean/veros/releases>`__).

2. Change current direcory to the Veros root directory ::

      $ cd veros

3. Create a new conda environment for Veros, and install all relevant dependencies by running ::

      $ conda env create -f conda-environment.yml

4. To use Veros, activate your new conda environment via ::

      $ conda activate veros

5. Copy a pre-implemented :class:`Global 4deg <global_4deg.GlobalFourDegreeSetup>` model setup from the :doc:`/reference/setup-gallery` ::

      $ veros copy-setup global_4deg

6. Change current directory to the setup directory ::

      $ cd global_4deg/

7. Modify model parameters with `nano text editor <https://www.nano-editor.org>`__ or another one ::

      $ nano global_4deg.py

8. Run the model in serial mode on one CPU core ::

      $ veros run global_4deg.py

9. In case you want to run Veros in parallel mode, you need to reinstall HDF5 library with parallel I/O support ::

      $ conda install "h5py=*=mpi_mpich*" --force-reinstall

10. To run the model in parallel mode on 4 CPU cores execute ::

      $ mpirun -np 4 veros run global_4deg.py -n 2 2


MPI Oriented Development and Investigation (MODI)
+++++++++++++++++++++++++++++++++++++++++++++++++

In order to install Veros with `Biogeochemistry plugin <https://veros-bgc.readthedocs.io/en/latest/>`__ start **Ocean HPC Notebook** on **Jupyter service** home page following :ref:`the instructions above <jupyter services target>`.

1. Change your current directory to ~/modi_mount by double-clicking the modi_mount folder circled in red

.. figure:: /_images/tutorial/erda_modi_terminal.png
   :width: 100%
   :align: center

   JupyterLab interface on MODI.

2. Download :download:`modi_veros_batch.sh </_downloads/modi_veros_batch.sh>` and :download:`modi_veros_run.sh </_downloads/modi_veros_run.sh>` scripts on your PC/Laptop and upload them to MODI (press circled in red arrow button as on the figure above).

3. Launch **Terminal** and change the directory there to ~/modi_mount ::

      $ cd ~/modi_mount

4. Submit Veros job to `Slurm <https://slurm.schedmd.com/quickstart.html>`__ queue in order to install Veros with `Biogeochemistry plugin <https://veros-bgc.readthedocs.io/en/latest/>`__ plus create and run BGC setup ::

      $ sbatch ./modi_veros_batch.sh

.. note::
   It's particularly important to run sbatch commands from the ~/modi_mount directory for jobs to succeed.

`Slurm <https://slurm.schedmd.com/quickstart.html>`__  is an open source, fault-tolerant, and highly scalable cluster management and job scheduling system for large and small Linux clusters.
There are a couple of basic Slurm commands that can be used to get an overview of the MODI cluster and manage your jobs, such as:

**sinfo** outputs the available partitions (modi_devel, modi_short, modi_long), their current availability (e.g. up or down), the maximum time a job can run before it is automatically terminated, the number of associated nodes and their individual state ::

       $ spj483_ku_dk@848874c4e509:~$ sinfo
       PARTITION   AVAIL  TIMELIMIT  NODES  STATE NODELIST
       modi_devel*    up      15:00      1    mix modi004
       modi_devel*    up      15:00      7   idle modi[000-003,005-007]
       modi_short     up 2-00:00:00      1    mix modi004
       modi_short     up 2-00:00:00      7   idle modi[000-003,005-007]
       modi_long      up 7-00:00:00      1    mix modi004
       modi_long      up 7-00:00:00      7   idle modi[000-003,005-007]

**sbatch** is used to submit a job (batch) script for later execution. The script will typically contain one or more srun commands to launch parallel tasks ::

       $ spj483_ku_dk@848874c4e509:~/modi_mount$ sbatch submit.sh
       Submitted batch job 10030

where 10030 is {JOBID}.

**squeue** shows queued jobs and their status, e.g. pending (PD) or running (R) ::

       $ spj483_ku_dk@848874c4e509:~$ squeue
       JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
       10030 modi_shor veros_bg spj483_k  R       0:09      1 modi005

**scancel** cancels job allocation to release a node ::

       $ scancel 10030




