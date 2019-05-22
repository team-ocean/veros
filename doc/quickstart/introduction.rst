A short introduction to Veros
=============================

The vision
----------

Veros is an adaptation of `pyOM2 <https://wiki.zmaw.de/ifm/TO/pyOM2>`_ (v2.1.0), developed by Carsten Eden (Institut für Meereskunde, Hamburg University). In contrast to pyOM2, however, this implementation does not rely on a Fortran backend for computations - everything runs in pure Python, down to the last parameterization. We believe that using this approach it is possible to create an open source ocean model that is:

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

.. figure:: /_images/introduction/c-grid.svg
   :width: 80%
   :align: center

   The structure of the Arakawa C-grid.

Veros supports both Cartesian and pseudo-spherical (i.e., including additional metric terms) coordinate systems. Islands or holes in the domain are fully supported by the streamfunction solver. Zonal boundaries can either be cyclic or regraded as walls (with free-slip boundary conditions).

Available parameterizations
+++++++++++++++++++++++++++

At its core, Veros currently offers the following solvers, numerical schemes, parameterizations, and closures:

**Surface pressure**:
 - a high-performance streamfunction solver via an iterative Poisson solver

**Equation of state**:
 - the full 48-term TEOS equation of state
 - various linear and nonlinear model equations from [Vallis2006]_

**Friction**:
 - harmonic or biharmonic lateral friction
 - linear or quadratic bottom friction
 - interior Rayleigh friction
 - explicit or fully implicit harmonic vertical friction

**Advection**:
 - a classical second-order central difference scheme
 - a second-order scheme with a superbee flux-limiter

**Diffusion**:
 - harmonic or biharmonic lateral diffusion
 - explicit or implicit harmonic vertical diffusion

**Isoneutral mixing**:
 - lateral mixing of tracers along neutral surfaces following [Griffies1998]_ (optional)

**Internal wave breaking**:
 - IDEMIX as in [OlbersEden2013]_ (optional)

**EKE model** (eddy kinetic energy):
 - meso-scale eddy mixing closure after [Gent1995]_, either with constant coefficients or calculated using the prognostic EKE closure by [EdenGreatbatch2008]_ (optional)

**TKE model** (turbulent kinetic energy):
 - prognostic TKE model for vertical mixing as introduced in [Gaspar1990]_ (optional)

Diagnostics
+++++++++++

Diagnostics are responsible for handling all model output, runtime checks of the solution, and restart file handling. They are implemented in a modular fashion, so additional diagnostics can be implemented easily. Already implemented diagnostics handle snapshot output, time-averaging of variables, monitoring of energy fluxes, and calculation of the overturning streamfunction.

For more information, see :doc:`/reference/diagnostics`.

Pre-configured model setups
+++++++++++++++++++++++++++

Veros supports a wide range of model configurations. Several setups are already implemented that highlight some of the capabilities of Veros, and that serve as a basis for users to set up their own configuration: :doc:`/reference/setup-gallery`.

Current limitations
+++++++++++++++++++

Veros is still in early development. There are several open issues that we would like to fix later on:

**Physics**:
 - Veros does not yet implement any of the more recent pyOM2.2 features such as the ROSSMIX parameterization, IDEMIX v3.0, open boundary conditions, or cyclic meridional boundaries. It neither implements all of pyOM2.1's features - missing are e.g. the non-hydrostatic solver, IDEMIX v2.0, and the surface pressure solver.
 - Since the grid is required to be rectilinear, there is currently no natural way to handle the singularity at the North Pole. The northern and southern boundaries of the domain are thus always "walls".
 - There is currently no ice sheet model in Veros. Some realistic setups employ a simple ice mask that cut off atmospheric forcing for water that gets too cold instead.

**Technical issues**:
 - For the time being, Veros' dynamical core is still more or less a direct port of PyOM2. This means that numerics and physics are still tightly coupled, which makes for a far from optimal user experience. In a future version of Veros, we would like to introduce additional abstraction to make the core routines a lot more readable than they are now.

References
++++++++++

.. [EdenGreatbatch2008] Eden, Carsten, and Richard J. Greatbatch. "Towards a mesoscale eddy closure." Ocean Modelling 20.3 (2008): 223-239.

.. [OlbersEden2013] Olbers, Dirk, and Carsten Eden. "A global model for the diapycnal diffusivity induced by internal gravity waves." Journal of Physical Oceanography 43.8 (2013): 1759-1779.

.. [Gent1995] Gent, Peter R., et al. "Parameterizing eddy-induced tracer transports in ocean circulation models." Journal of Physical Oceanography 25.4 (1995): 463-474.

.. [Griffies1998] Griffies, Stephen M. "The Gent–McWilliams skew flux." Journal of Physical Oceanography 28.5 (1998): 831-841.

.. [Vallis2006] Vallis, Geoffrey K. "Atmospheric and oceanic fluid dynamics: fundamentals and large-scale circulation." Cambridge University Press, 2006.

.. [Gaspar1990] Gaspar, Philippe, Yves Grégoris, and Jean‐Michel Lefevre. "A simple eddy kinetic energy model for simulations of the oceanic vertical mixing: Tests at station Papa and Long‐Term Upper Ocean Study site." Journal of Geophysical Research: Oceans 95.C9 (1990): 16179-16193.
