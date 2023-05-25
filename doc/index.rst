:tocdepth: 5


.. image:: /_images/veros-logo-400px.png
   :align: center

|

Versatile Ocean Simulation in Pure Python
=========================================

Veros, *the versatile ocean simulator*, aims to be the swiss army knife of ocean modeling. It is a full-fledged primitive equation ocean model that supports anything between idealized toy models and `realistic, high-resolution, global ocean simulations <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002717>`_. And because Veros is written in pure Python, the days of struggling with complicated model setup workflows, ancient programming environments, and obscure legacy code are finally over.

*In a nutshell, we want to enable high-performance ocean modelling with a clear focus on flexibility and usability.*

Veros supports a NumPy backend for small-scale problems, and a
high-performance `JAX <https://github.com/google/jax>`_ backend
with CPU and GPU support. It is fully parallelized via MPI and supports
distributed execution on any number of nodes, including multi-GPU architectures (see also :doc:`our benchmarks </more/benchmarks>`).

The dynamical core of Veros is based on `pyOM2 <https://wiki.cen.uni-hamburg.de/ifm/TO/pyOM2>`_, an ocean model with a Fortran backend and Fortran and Python frontends.

If you want to learn more about the background and capabilities of Veros, you should check out :doc:`introduction/introduction`. If you are already convinced, you can jump right into action, and :doc:`learn how to get started <introduction/get-started>` instead!

.. image:: /_images/tagline.png
   :scale: 50%
   :class: no-scaled-link
   :alt: ... because the Baroque is over.

.. seealso::

   We outline some of our design philosophy and current direction in `this blog post <https://dionhaefner.github.io/2021/04/higher-level-geophysical-modelling/>`__.

.. toctree::
   :maxdepth: 2
   :caption: Start here

   introduction/introduction
   introduction/get-started
   introduction/advanced-installation

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial/analysis
   tutorial/cluster
   tutorial/dev
   tutorial/erda

.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/setup-gallery
   reference/settings
   reference/variables
   reference/diagnostics
   reference/cli
   reference/public-api

.. toctree::
   :maxdepth: 1
   :caption: Plug-ins

   plugins/user-guide
   Biogeochemistry plugin (external) <https://veros-bgc.readthedocs.io>
   Extra setups plugin (external) <https://veros-extra-setups.readthedocs.io>
   Sea ice plugin (external) <https://veris.readthedocs.io>
   plugins/developer-guide

.. toctree::
   :maxdepth: 2
   :caption: More Information

   more/benchmarks
   more/external_tools
   more/howtocite
   more/publications
   Visit us on GitHub <https://github.com/team-ocean/veros>
