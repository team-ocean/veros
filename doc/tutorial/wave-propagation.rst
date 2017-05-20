Creating an advanced model setup
================================

.. note::

   This guide is still work in progress.

This is a step-by-step guide that illustrates how even complicated
setups can be created with relative ease (thanks to the tools provided
by the scientific Python community). As an example, we will re-create
the :doc:`wave propagation setup </reference/setups/wave-propagation>`
, which is a global ocean model with an idealized Atlantic.

The vision
----------

The purpose of this model is to examine wave propagation along the
eastern boundary of the North Atlantic. Since it is difficult to track
propagating waves

Since the presence of the Pacific in the model is crucial to achieve a
realistic ocean circulation,

This leaves us with the following requirements for the final wave
propagation model:

-  Global model with a resolution of around 1 degree.
-  Idealized geometry in the Atlantic, so analytically derived wave
   properties hold.
-  A refined grid resolution at the eastern boundary of the Atlantic.
-  Zonally averaged wind stresses in the Atlantic to fill gaps in wind
   stress data.
-  A somehow interpolated initial state and forcings for cells that have
   been converted from land to ocean.

Step 1: The model skeleton
--------------------------

Instead of starting from scratch

Step 2: Preparing the forcing fields & initial conditions
---------------------------------------------------------

Step 3: Creating idealized geometries
-------------------------------------

Create a mask image
~~~~~~~~~~~~~~~~~~~

Modify the mask
~~~~~~~~~~~~~~~

Import to Veros
~~~~~~~~~~~~~~~
