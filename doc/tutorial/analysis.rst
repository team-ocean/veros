.. ipython:: python
    :suppress:

    import os
    from veros import tools

    OUTPUT_FILES = tools.get_assets("tutorial_analysis", os.path.join("tutorial", "analysis-assets.json"))

Analysis of Veros output
========================

In this tutorial, we will use `xarray <http://xarray.pydata.org/en/stable/>`__, `numpy <https://numpy.org>`__ and `matplotlib <https://matplotlib.org>`__ to load and analyze the model output. You can run these commands in `IPython <https://ipython.readthedocs.io/en/stable/>`__ or a `Jupyter Notebook <https://jupyter.org>`__. Just make sure to install the dependencies first::

  $ pip install numpy xarray matplotlib netcdf4

The analysis below is performed for 100 yr integration of :doc:`global_4deg </reference/setups/4deg>` from the :doc:`setup gallery </reference/setup-gallery>`.
The model output is preloaded from `remote public directory <https://sid.erda.dk/cgi-sid/ls.py?share_id=CD8UzHCj2Q;current_dir=inputdata/tutorial_analysis;flags=f>`__ and accessed through ``OUTPUT_FILES`` dictionary, which contains paths to 4 different files:

.. ipython:: python

    for key in OUTPUT_FILES.keys():
        print(OUTPUT_FILES[key])

So, when we open ``OUTPUT_FILES["averages"]``, it means we open ``4deg.averages.nc`` file from our local directory. At the very beginning we will need to load ``xarray``. To do so, execute the following code:

.. ipython:: python

    import xarray as xr

The ``xarray`` module provides data structure and API for working with labeled N-dimensional arrays. These labels have encoded information about how the arrays' values map to locations in space, time, etc.

Load and manipulate averages
----------------------------

In order to load our first output file and display its content execute the following two commands:

.. ipython:: python

    ds = xr.open_dataset(OUTPUT_FILES["averages"], decode_times=False)
    ds

We can easily access/modify individual data variable and its attributes. To demonstrate it let's convert the units of baratropic stream function from :math:`\frac{m^{3}}{s}` to :math:`Sv` for better convenience:

.. ipython:: python

    psi = ds.psi / 1e6
    psi.attrs["units"] = "Sv"

To select values of ``psi`` by its integer location over ``Time`` coordinate (last slice) and plot it execute:

.. ipython:: python
    :okwarning:

    @savefig psi.png width=5in
    psi.isel(Time=-1).plot.contourf(levels=50)

In order to compute the decadal mean (of the last 10yrs) of zonal-mean ocean salinity use the following command:

.. ipython:: python
    :okwarning:

    @savefig salt.png width=5in
    ds['salt'].isel(Time=slice(-10,None)).mean(dim=('Time', 'xt')).plot.contourf(levels=50, cmap='ocean')

One can also compute meridional-mean temperature. Since the model output is defined on a regular latitude/ longitude grid, the grid cell area decreases towards the pole.
For a rectangular grid the cosine of the latitude is proportional to the grid cell area, thus we can compute and use the following weights to adjust the temperature variable:

.. ipython:: python

    import numpy as np
    weights = np.cos(np.deg2rad(ds.yt))
    weights.name = "weights"
    weights
    temp_weighted = ds['temp'].isel(Time=-1).weighted(weights)

Now, we can calculate weighted mean temperature over meridians and plot it:

.. ipython:: python
    :okwarning:

    @savefig temp.png width=5in
    temp_weighted.mean(dim='yt').plot.contourf(vmin=-2, vmax=22, levels=25, cmap='inferno')

Explore overturning circulation
-------------------------------

.. ipython:: python

    ds = xr.open_dataset(OUTPUT_FILES["overturning"], decode_times=False)
    ds

Let's convert the units of meridional overturning circulation (MOC) from :math:`\frac{m^{3}}{s}` to :math:`Sv` and plot it:

.. ipython:: python
    :okwarning:

    vsf_depth = ds.vsf_depth / 1e6
    vsf_depth.attrs["long_name"] = "MOC"
    vsf_depth.attrs["units"] = "Sv"

    @savefig vsf_depth_2d.png width=5in
    vsf_depth.isel(Time=-1).plot.contourf(levels=50)

Plot time series
----------------

To inspect coordinates ``zw``, ``yu``, ``Time`` to be used for plotting of MOC time series execute:

.. ipython:: python

    ds['zw']
    ds['yu']
    vsf_depth['Time'].isel(Time=slice(10,))

We can see that the ``Time`` coordinate is given in days (a year corresponds to 360 days here). In order to have a more
meaningful x-axis in our figures, we divide the ``Time`` coordinate by the number of days per year and change its unit:

.. ipython:: python

    vsf_depth['Time'] = vsf_depth['Time'] / 360.
    vsf_depth.Time.attrs['units'] = 'year'

Let's select values of array by labels instead of integer location and plot a time series of the overturning minimum between 40°N and 60°N and 550-1800m depth:

.. ipython:: python

    @savefig vsf_depth_min.png width=5in
    vsf_depth.sel(zw=slice(-1810., -550.), yu=slice(40., 60.)).min(axis=(1,2)).plot()