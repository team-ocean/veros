Creating an advanced model setup
================================

.. note::

   This guide is still work in progress.

This is a step-by-step guide that illustrates how even complicated setups can be
created with relative ease (thanks to the tools provided by the scientific
Python community). As an example, we will re-create the :doc:`wave propagation
setup </reference/setups/wave-propagation>`, which is a global ocean model with
an idealized Atlantic.

.. figure:: /_images/gallery/wave-propagation.png
   :align: center
   :scale: 75%

   The resulting stream function after about 1 year of integration.

The vision
----------

The purpose of this model is to examine wave propagation along the eastern
boundary of the North Atlantic. Since it is difficult to track propagating waves
along ragged geometry or through uneven forcing fields, we will idealize the
representation of the North Atlantic; and as the presence of the Pacific in the
model is crucial to achieve a realistic ocean circulation, we want to use a
global model.

This leaves us with the following requirements for the final wave propagation
model:

#. A global model with a resolution of around 2 degrees and meridional
   stretching.
#. Convert the eastern boundary of the Atlantic to a straight line,
   so analytically derived wave properties hold.
#. A refined grid resolution at the eastern boundary of the Atlantic.
#. Zonally averaged forcings in the Atlantic.
#. A somehow interpolated initial state for cells that have been converted
   from land to ocean in the North Atlantic.
#. Options for shelf and continental slope.
#. A multiplier setting for the Southern Ocean wind stress.

Model skeleton
--------------

Instead of starting from scratch, we can use the :doc:`global one degree model
</reference/setups/1deg>` as a template, which looks like this:

.. literalinclude:: /../veros/setup/global_1deg/global_one_degree.py
  :language: python

The biggest changes in the new wave propgation setup will be located in the
:func:`set_grid` :func:`set_topography` and :func:`set_initial_conditions`
methods to accomodate for the new geometry and the interpolation of initial
conditions to the modified grid, so we can concentrate on implementing those
first.

Step 1: Setup grid
------------------

.. warning::

    When using a non-uniform grid,

Step 2: Create idealized topography
-----------------------------------

Usually, to create an idealized topography, one would simply hand-craft some
input and forcing files that reflect the desired changes. However, since we want
our setup to have flexible resolution, we will have to write an algorithm that
creates these input files for any given number of grid cells. One convenient way
to achieve this is by creating some high-resolution *masks* representing the
target topography by hand, and then interpolate these masks to the desired
resolution.

Create a mask image
~~~~~~~~~~~~~~~~~~~

Before we can start, we need to download a high-resolution topography dataset.
There are many freely available topographical data sets on the internet; one of
them is `ETOPO5 <https://www.ngdc.noaa.gov/mgg/global/etopo5.HTML>`_ (with a
resolution of 5 arc-minutes), which we will be using throughout this tutorial.
To create a mask image from the topography file, you can use the :doc:`command
line tool </reference/cli>` `veros create-mask`, e.g. like ::

  $ veros create-mask ETOPO5_Ice_g_gmt4.nc

This creates a one-to-one representation of the topography file as a PNG image.
However, in the case of the 5 arc-minute topography, the resulting image
includes a lot of small islands and complicated coastlines that might cause
problems when being interpolated to a numerical grid with a much lower
resolution. To address this, the `create-mask` script accepts a `scale`
argument. When given, a Gaussian filter with standard deviation `scale` (in grid
cells) is applied to the resulting image, smoothing out small features. The
command ::

  $ veros create-mask ETOPO5_Ice_g_gmt4 --scale 3 3

results in the following mask:

.. figure:: /_images/tutorial-setup/mask-smooth.png
   :align: center
   :width: 600

   Smoothed topography mask

which looks good enough to serve as a basis for horizontal resolutions of around
one degree.

Modify the mask
~~~~~~~~~~~~~~~

We can now proceed to mold this realistic version of the global topography into
the desired idealized shape. You can use any image editor you have availble; one
possibility is the free software `GIMP <https://www.gimp.org/>`_. Inside the
editor, we can use the pencil tools to create a modified version of the
topography mask:

.. figure:: /../veros/setup/wave_propagation/topography_idealized.png
   :align: center
   :width: 600

   Idealized topography mask

In this modified version, I have

#. replaced the eastern boundary of the North Atlantic by a meridional line;
#. removed all lakes and inland seas;
#. thickened Central America (to prevent North and South America to become
   disconnected due to interpolation artifacts); and
#. removed the Arctic Ocean and Hudson Bay.

Now that our topography mask is finished, we can go ahead and implement it in
the Veros setup!

Import to Veros
~~~~~~~~~~~~~~~

To read the mask in PNG format, we are going to use the Python Imaging Library
(PIL).

Step 3: Interpolate forcings & initial conditions
-------------------------------------------------

.. figure:: /../veros/setup/wave_propagation/na_mask.png
   :align: center
   :width: 600

   Mask to identify grid cells in the North Atlantic

Step 4: Set up diagnostics & final touches
------------------------------------------
