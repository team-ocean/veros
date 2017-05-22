Creating an advanced model setup
================================

.. note::

   This guide is still work in progress.

This is a step-by-step guide that illustrates how even complicated
setups can be created with relative ease (thanks to the tools provided
by the scientific Python community). As an example, we will re-create
the :doc:`wave propagation setup </reference/setups/wave-propagation>`,
which is a global ocean model with an idealized Atlantic.

.. figure:: /_images/gallery/wave-propagation.png
   :align: center
   :scale: 75%

   The resulting stream function after about 1 year of integration.

The vision
----------

The purpose of this model is to examine wave propagation along the
eastern boundary of the North Atlantic. Since it is difficult to track
propagating waves along ragged geometry or through uneven forcing fields,
we will idealize the representation of the North Atlantic; and as
the presence of the Pacific in the model is crucial to achieve a
realistic ocean circulation, we want to use a global model.

This leaves us with the following requirements for the final wave
propagation model:

 #. a global model with a resolution of around 1 degree
 #. convert the eastern boundary of the Atlantic to a straight line, so analytically
    derived wave properties hold
 #. a refined grid resolution at the eastern boundary of the Atlantic
 #. zonally averaged forcings in the Atlantic
 #. a somehow interpolated initial state for cells that have been converted from
    land to ocean

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
