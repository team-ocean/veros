Analysis of Veros output
========================

In this tutorial, we will use `xarray <http://xarray.pydata.org/en/stable/>`__ and `matplotlib <https://matplotlib.org>`__ to load, analyze, and plot the model output. We will also use the `cmocean colormaps <https://matplotlib.org/cmocean/>`__. You can run these commands in `IPython <https://ipython.readthedocs.io/en/stable/>`__ or a `Jupyter Notebook <https://jupyter.org>`__. Just make sure to install the dependencies first::

  $ pip install xarray matplotlib netcdf4 cmocean

The analysis below is performed for 100 yr integration of the :doc:`global_4deg </reference/setups/4deg>` setup from the :doc:`setup gallery </reference/setup-gallery>`.

If you want to run this analysis yourself, you can `download the data here <https://sid.erda.dk/cgi-sid/ls.py?share_id=CD8UzHCj2Q;current_dir=inputdata/tutorial_analysis;flags=f>`__. We access the files through the dictionary ``OUTPUT_FILES``, which contains the paths to the 4 different files:

.. ipython:: python

    OUTPUT_FILES = {
        "snapshot": "4deg.snapshot.nc",
        "averages": "4deg.averages.nc",
        "overturning": "4deg.overturning.nc",
        "energy": "4deg.energy.nc",
    }

.. ipython:: python
    :suppress:

    # actually, we are loading input files through the Veros asset mechanism
    import os
    from veros import tools
    OUTPUT_FILES = tools.get_assets("tutorial_analysis", os.path.join("tutorial", "analysis-assets.json"))

Let's start by importing some packages:

.. ipython:: python

    import xarray as xr
    import numpy as np
    import cmocean

Most of the heavy lifting will be done by ``xarray``, which provides a data structure and API for working with labeled N-dimensional arrays. ``xarray`` datasets automatically keep track how the values of the underlying arrays map to locations in space and time, which makes them immensely useful for analyzing model output.

Load and manipulate averages
----------------------------

In order to load our first output file and display its content execute the following two commands:

.. ipython:: python

    ds_avg = xr.open_dataset(OUTPUT_FILES["averages"], decode_timedelta=False)
    ds_avg

We can easily access/modify individual data variables and their attributes. To demonstrate this let's convert the units of the barotropic stream function from :math:`\frac{m^{3}}{s}` to :math:`Sv`:

.. ipython:: python

    ds_avg["psi"] = ds_avg.psi / 1e6
    ds_avg["psi"].attrs["units"] = "Sv"

Now, we select the last time slice of ``psi`` and plot it:

.. ipython:: python
    :okwarning:

    @savefig psi.png width=5in
    ds_avg["psi"].isel(Time=-1).plot.contourf(levels=50, cmap="cmo.balance")

In order to compute the decadal mean (of the last 10yrs) of zonal-mean ocean salinity use the following command:

.. ipython:: python
    :okwarning:

    @savefig salt.png width=5in
    (
        ds_avg["salt"]
        .isel(Time=slice(-10,None))
        .mean(dim=("Time", "xt"))
        .plot.contourf(levels=50, cmap="cmo.haline")
    )

One can also compute meridional mean temperature. Since the model output is defined on a regular latitude / longitude grid, the grid cell area decreases towards the pole.
To get an accurate mean value, we need to weight each cell by its area:

.. ipython:: python

    ds_snap = xr.open_dataset(OUTPUT_FILES["snapshot"])
    # use cell area as weights, replace missing values (land) with 0
    weights = ds_snap["area_t"].fillna(0)

Now, we can calculate the meridional mean temperature (via ``xarray``'s ``.weighted`` method) and plot it:

.. ipython:: python
    :okwarning:

    @savefig temp.png width=5in
    temp_weighted = (
        ds_avg["temp"]
        .isel(Time=-1)
        .weighted(weights)
        .mean(dim="yt")
        .plot.contourf(vmin=-2, vmax=22, levels=25, cmap="cmo.thermal")
    )

Explore overturning circulation
-------------------------------

.. ipython:: python

    ds_ovr = xr.open_dataset(OUTPUT_FILES["overturning"])
    ds_ovr

Let"s convert the units of meridional overturning circulation (MOC) from :math:`\frac{m^{3}}{s}` to :math:`Sv` and plot it:

.. ipython:: python
    :okwarning:

    ds_ovr["vsf_depth"] = ds_ovr.vsf_depth / 1e6
    ds_ovr.vsf_depth.attrs["long_name"] = "MOC"
    ds_ovr.vsf_depth.attrs["units"] = "Sv"

    @savefig vsf_depth_2d.png width=5in
    ds_ovr.vsf_depth.isel(Time=-1).plot.contourf(levels=50, cmap="cmo.balance")

Plot time series
----------------

Let's have a look at the ``Time`` coordinate of the dataset:

.. ipython:: python

    ds_ovr["Time"].isel(Time=slice(10,))

We can see that it has the type ``np.timedelta64``, which by default has a resolution of nanoseconds. In order to have a more
meaningful x-axis in our figures, we add another coordinate "years" by dividing ``Time`` by the length of a year (360 days in Veros):

.. ipython:: python

    years = ds_ovr["Time"] / np.timedelta64(360, "D")
    ds_ovr = ds_ovr.assign_coords(years=("Time", years.data))

Let's select values of array by labels instead of index location and plot a time series of the overturning minimum between 40°N and 60°N and 550-1800m depth, with years on the x-axis:

.. ipython:: python

    @savefig vsf_depth_min.png width=5in
    (
        ds_ovr.vsf_depth
        .sel(zw=slice(-1810., -550.), yu=slice(40., 60.))
        .min(dim=("yu", "zw"))
        .plot(x="years")
    )
