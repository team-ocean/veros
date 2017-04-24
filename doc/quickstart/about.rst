A short introduction to Veros
=============================

This model is an adaptation of `pyOM2 <https://wiki.zmaw.de/ifm/TO/pyOM2>`_,
developed by Carsten Eden at Hamburg
University, Germany. In contrast to the original pyOM2, this implementation
does not rely on a Fortran backend for computations - everything runs in
pure Python, down to the last parameterization. This allows scientists to jump
right into action and adapt the model as they please, without having to deal with
ancient code, and to run and process the model dynamically, using our favorite
programming language.


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
