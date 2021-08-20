:tocdepth: 5


.. image:: /_images/veros-logo-400px.png
   :align: center

|

Versatile Ocean Simulation in Pure Python
=========================================

Veros, *the versatile ocean simulator*, aims to be the swiss army knife of ocean modeling. It is a full-fledged :abbr:`GCM (general circulation model)` that supports both idealized toy models and realistic set-ups.

Thanks to its interplay with `JAX <https://github.com/google/jax>`_, Veros runs efficiently on your laptop, gaming PC (including GPU support), or full-scale cluster (see also :doc:`our benchmarks </more/benchmarks>`).

In a nutshell, we want to enable ocean modelling with a clear focus on flexibility and usability.

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

   tutorial/wave-propagation
   tutorial/dev
   tutorial/cluster

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
   Biogeochemistry plugin (Veros-BGC) documentation <https://veros-bgc.readthedocs.io>
   Extra setups plugin <https://veros-extra-setups.readthedocs.io>
   plugins/developer-guide

.. toctree::
   :maxdepth: 2
   :caption: More Information

   more/benchmarks
   more/publications
   more/contact
   Visit us on GitHub <https://github.com/team-ocean/veros>
