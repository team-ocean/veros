A short introduction to Veros
=============================

The vision
----------

Veros is an adaptation of `pyOM2 <https://wiki.zmaw.de/ifm/TO/pyOM2>`_,
developed by Carsten Eden at Hamburg University, Germany. In contrast to pyOM2, however, this implementation
does not rely on a Fortran backend for computations - everything runs in
pure Python, down to the last parameterization. We belive that using this approach it is possible to create an open ocean model that is

1. **Easy to access**: Python modules are simple to install, and projects like `Anaconda <https://www.continuum.io/anaconda-overview>`_ are doing a great job in creating platform-independent environments.
2. **Easy to use**: Anyone with some experience can use their favorite Python tools to set up, control, and post-process Veros.
3. **Easy to verify**: Python code tends to be concise and easy to read, even for people with little practical programming experience. This enables a wide range of people to spot errors in our code, solidifying it in the process.
4. **Easy to modify**: Due to the popularity of Python, its dynamic code structure, and OOP-capabilities, Veros can be extended and modified with minimal effort.

This

At a glance
-----------

.. note::
  This section provides a quick overview of the capabilities and limitations of
  Veros. For a comprehensive description of the physics and numerics behind Veros,
  please refer to `the documentation of pyOM2 <https://wiki.zmaw.de/ifm/TO/pyOM2>`_.

Features
++++++++

The model grid
++++++++++++++

.. figure:: c-grid.svg
   :width: 100%

   The structure of the Arakawa C-grid.

Available parameterizations
+++++++++++++++++++++++++++


Pre-configured model setups
+++++++++++++++++++++++++++

:ref:`api-setup`
