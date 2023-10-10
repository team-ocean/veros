Running Veros on ERDA
=====================

ERDA
----

The Electronic Research Data Archive (`ERDA <https://www.erda.dk>`__) at the University of Copenhagen (`KU/UCPH <https://www.ku.dk/english/>`__) is meant for storing, sharing, analyzing and archiving research data.
ERDA delivers safe central storage space for private and shared files, interactive analysis tools, and data archiving for safe-keeping and publishing.

.. _erda-jupyter:

Getting started with ERDA's Jupyter server
------------------------------------------

ERDA integrates a set of `Jupyter <https://jupyter.org>`__ services, which can be used to easily perform a wide range of data analysis and visualization tasks directly on your ERDA data.
The system relies on the `JupyterLab <https://jupyterlab.readthedocs.io/en/stable/>`__ web interface to provide interactive Python notebooks or Linux command line access (Terminal) with direct and efficient access to your ERDA home directory.
To get access to these services, ERDA provides a Jupyter button in the navigation menu.

.. figure:: /_images/erda/erda_welcome.png
   :width: 100%
   :align: center

   ERDA navigation menu.

Upon clicking it, the page to **Select a Jupyter Service** appears.
On this page you are presented with a set of horizontal service tabs at the top, and each tab presents and describes the individual service and how it is configured in the **Service Description**.

.. note::

      ERDA offers 2 services, DAG and MODI. MODI offers more powerful hardware, but you have to use a scheduling system to use it (:ref:`see below <modi>`). If you are unsure what to use, you should start with DAG.

Below the description there is a **Start SERVICE** button, which you can click to open a connection to that particular service in a new web browser tab or window.

.. figure:: /_images/erda/erda_dag_spawn.png
   :width: 100%
   :align: center

   Select a Jupyter Service menu.

By default, it will take you to your personal home page on the **Jupyter service** as shown below, which is provided via a hosted version of JupyterHub.
That is, the standard infrastructure to provide individual isolated Jupyter notebook containers to multiple users sharing a pool of actual compute nodes.

.. figure:: /_images/erda/erda_jservice_homepage.png
   :width: 100%
   :align: center

   Top fragment of Jupyter service home page.

After clicking **Start My Server**, the site will give you an option to chose which notebook image you want to spawn.
Select **HPC Notebook** as shown below and press the **Start** button.

.. figure:: /_images/erda/erda_dag_image.png
   :width: 100%
   :align: center

   Top fragment of Jupyter service home page with selected HPC Notebook image.

This will spawn the **HPC Notebook** image and redirect you straight to the JupyterLab interface as shown below.
The JupyterLab interface is the same in all available Services (DAG and MODI).

.. figure:: /_images/erda/erda_dag_terminal.png
   :width: 100%
   :align: center

   JupyterLab interface on DAG.

Follow the Veros installation instructions below with respect to the selected service.

Data Analysis Gateway (DAG)
+++++++++++++++++++++++++++

In order to install Veros on a DAG instance do the following after launching the **Terminal**:

1. Clone the Veros repository:

   .. exec::

      from veros import __version__ as veros_version
      if "0+untagged" in veros_version:
            veros_version = "main"
      else:
            veros_version = f"v{veros_version}"
      if "+" in veros_version:
            veros_version, _ = veros_version.split("+")
      print(".. code-block::\n")
      print("   $ cd ~/modi_mount")
      print(f"   $ git clone https://github.com/team-ocean/veros.git -b {veros_version}")

   (or `any other version of Veros <https://github.com/team-ocean/veros/releases>`__).

2. Change the current directory to the Veros root directory::

      $ cd veros

3. Create a new conda environment for Veros, and install all relevant dependencies by running::

      $ conda env create -f conda-environment.yml

4. To use Veros, activate your new conda environment via::

      $ conda activate veros

5. Make a folder for your Veros setups, and switch to it::

      $ mkdir ~/vs-setups
      $ cd ~/vs-setups

6. Copy the :doc:`global 4deg </reference/setups/4deg>` model template from the :doc:`setup gallery </reference/setup-gallery>`::

      $ veros copy-setup global_4deg

7. Change the current directory to the setup directory::

      $ cd global_4deg/

.. _erda-jupyter-editor:

8. One can modify model parameters with the **JupyterLab editor**. To do that you need to navigate to your setup directory in the JupyterLab file browser (left panel) of the **JupyterLab interface** and double-click the :file:`global_4deg.py` file (circled in red) as in the figure below

.. figure:: /_images/erda/erda_dag_edit_file.png
   :width: 100%
   :align: center

   JupyterLab editor on DAG.

Press :command:`CTRL+S` (:command:`CMD+S` on MacOS) on a keyboard to save your changes and close the file by pressing the cross button (circled in red).

9. Run the model in serial mode on one CPU core::

      $ veros run global_4deg.py

10. In case you want to run Veros in parallel mode, you need to reinstall the HDF5 library with parallel I/O support::

      $ conda install "h5py=*=mpi_mpich*" --force-reinstall

11. To run the model in parallel mode on 4 CPU cores execute::

      $ mpirun -np 4 veros run global_4deg.py -n 2 2

.. _modi:

MPI Oriented Development and Investigation (MODI)
+++++++++++++++++++++++++++++++++++++++++++++++++

In order to install Veros with the `veros-bgc biogeochemistry plugin <https://veros-bgc.readthedocs.io/en/latest/>`__ start an **Ocean HPC Notebook** from the **Jupyter service** home page following :ref:`the instructions above <erda-jupyter>`.

1. Launch the **Terminal**, change your current directory to ~/modi_mount and clone the Veros repository:

   .. exec::

      from veros import __version__ as veros_version
      if "0+untagged" in veros_version:
            veros_version = "main"
      else:
            veros_version = f"v{veros_version}"
      if "+" in veros_version:
            veros_version, _ = veros_version.split("+")
      print(".. code-block::\n")
      print("   $ cd ~/modi_mount")
      print(f"   $ git clone https://github.com/team-ocean/veros.git -b {veros_version}")

2. Create a new conda environment for Veros::

      $ conda create --prefix ~/modi_mount/conda-env-veros -y python=3.11

3. To use the new environment, activate it via::

      $ conda activate ~/modi_mount/conda-env-veros

4. Install Veros, its biogeochemistry plugin and all relevant dependencies by running::

      $ pip3 install ./veros
      $ pip3 install veros-bgc

5. Copy the ``bgc_global_4deg`` model template from the `setup gallery <https://veros-bgc.readthedocs.io/en/latest/reference/setup-gallery.html>`__::

      $ veros copy-setup bgc_global_4deg

6. Change your current directory in the JupyterLab file browser (left panel) of the **JupyterLab interface** to ~/modi_mount by double-clicking the modi_mount folder (circled in red).

.. figure:: /_images/erda/erda_modi_terminal.png
   :width: 100%
   :align: center

   JupyterLab interface on MODI.

7. Download the :download:`modi_veros_batch.sh </_downloads/modi_veros_batch.sh>` and :download:`modi_veros_run.sh </_downloads/modi_veros_run.sh>` scripts on your PC/Laptop and upload them to MODI (press circled in red arrow button as on the figure above).

8. Navigate to your setup directory in the JupyterLab file browser and modify (if needed) the model parameters in the :file:`bgc_global_four_degree.py` file with the **JupyterLab editor** following :ref:`the instructions above <erda-jupyter-editor>`.

9. To run your BGC setup submit a job to MODI's `Slurm <https://slurm.schedmd.com/quickstart.html>`__ queue::

      $ sbatch ./modi_veros_batch.sh ~/modi_mount/bgc_global_4deg/bgc_global_four_degree.py

.. note::

   It's particularly important to run ``sbatch`` commands from the ~/modi_mount directory for jobs to succeed.

Slurm is an open source, fault-tolerant, and highly scalable cluster management and job scheduling system for large and small Linux clusters.
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
