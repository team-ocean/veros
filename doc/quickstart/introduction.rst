A short introduction to Veros
=============================

The vision
----------

Veros is an adaptation of `pyOM2 <https://wiki.zmaw.de/ifm/TO/pyOM2>`_ (v2.1.0), developed by Carsten Eden (Institut für Meereskunde, Hamburg University). In contrast to pyOM2, however, this implementation does not rely on a Fortran backend for computations - everything runs in pure Python, down to the last parameterization. We belive that using this approach it is possible to create an open source ocean model that is:

1. **Easy to access**: Python modules are simple to install, and projects like `Anaconda <https://www.continuum.io/anaconda-overview>`_ are doing a great job in creating platform-independent environments.
2. **Easy to use**: Anyone with some experience can use their favorite Python tools to set up, control, and post-process Veros.
3. **Easy to verify**: Python code tends to be concise and easy to read, even for people with little practical programming experience. This enables a wide range of people to spot errors in our code, solidifying it in the process.
4. **Easy to modify**: Due to the popularity of Python, its dynamic code structure, and :abbr:`OOP (object-oriented programming)`-capabilities, Veros can be extended and modified with minimal effort.

However, using Python over a compiled language like Fortran usually comes at a high computational cost. We try to overcome this gap for large models by providing an interface to `Bohrium <https://github.com/bh107/bohrium>`_, a framework that acts as a high-performance replacement for NumPy. Bohrium takes care of all parallelism in the background for us, so we can concentrate on writing a nice, readable ocean model.

In case you are curious about how Veros is currently stacking up against pyOM2 in terms of performance, you should check out :doc:`our benchmarks </more/benchmarks>`.

Features
--------

.. note::
  This section provides a quick overview of the capabilities and limitations of Veros. For a comprehensive description of the physics and numerics behind Veros, please refer to `the documentation of pyOM2 <https://wiki.zmaw.de/ifm/TO/pyOM2>`_. You can also obtain a copy of the PDF documentation :download:`here </_downloads/pyOM2.pdf>`.

The model domain
++++++++++++++++

The numerical solution is calculated using finite differences on an *Arakawa C-grid*, which is staggered in every dimension. *Tracers* (like temperature and salinity) are calculated at different positions than zonal, meridional, and vertical *fluxes* (like the velocities u, v, and w). The following figure shows the relative positions of the so-called T, U, V, and ζ grid points (W not shown):

.. figure:: /_images/c-grid.svg
   :width: 80%
   :align: center

   The structure of the Arakawa C-grid.

Veros supports both Cartesian and pseudo-spherical (i.e., including additional metric terms) coordinate systems. Islands or holes in the domain are fully supported by the streamfunction solver. Zonal boundaries can either be cyclic or regraded as walls.

Available parameterizations
+++++++++++++++++++++++++++

At its core, Veros currently offers the following solvers, numerical schemes, parameterizations, and closures:

- **Surface pressure**:
- **Equation of state**:
- **Friction**:
- **Advection**:
- **Diffusion**:
- **Isoneutral mixing**: (optional)
- **:abbr:`IW (internal waves)`**: IDEMIX by (optional)
- **:abbr:`EKE (eddy kinetic energy)`**: (optional)
- **:abbr:`TKE (turbulent kinetic energy)`**: (optional)

Diagnostics
+++++++++++

Diagnostics are reposible for

:doc:`/reference/diagnostics`


Pre-configured model setups
+++++++++++++++++++++++++++

:doc:`/reference/setup`

Current limitations
+++++++++++++++++++

Veros is

- It does not yet implement any of the more recent pyOM2.2 features such as the ROSSMIX parameterization, IDEMIX v2.0 and v3.0, open boundary conditions, or cyclic meridional boundaries.
- Since the grid
- Ice sheet model
