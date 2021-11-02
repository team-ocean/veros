.. ipython:: python
    :suppress:

    import os
    import shutil

    import matplotlib as mpl
    mpl.rcdefaults()

    from veros import tools

    OUTPUT_FILES = tools.get_assets("tutorial_analysis", os.path.join("tutorial", "analysis-assets.json"))


Analysis of Veros output
========================

In this tutorial, we will use `xarray <http://xarray.pydata.org/en/stable/>`__ and `matplotlib <https://matplotlib.org>`__ to load and analyze the model output. You can run these commands in `IPython <https://ipython.readthedocs.io/en/stable/>`__ or a `Jupyter Notebook <https://jupyter.org>`__. Just make sure to install the dependencies first:

```bash
$ pip install xarray matplotlib netcdf4
The analysis below is performed for 100 yr integration of :doc:`global_4deg </reference/setups/4deg>` from the :doc:`setup gallery </reference/setup-gallery>`. The model output is preloaded in our case and it has 4 different files:

.. code:: 

    $HOME/global_4deg/4deg.averages.nc
    $HOME/global_4deg/4deg.overturning.nc
    $HOME/global_4deg/4deg.energy.nc
    $HOME/global_4deg/4deg.snapshot.nc

So, when we load "averages", it means we load ``4deg.averages.nc`` file from our local directory. At the very beginning we will need to load ``xarray``. To do so, execute the following code:

.. ipython:: python

    import xarray as xr

Load and manipulate averages
----------------------------

.. ipython:: python

    ds = xr.open_dataset(OUTPUT_FILES["averages"], decode_times=False)
    ds

Let's change the units of geographical coordinates:

.. ipython:: python
    :suppress:

    ds.xt.attrs["long_name"] = "longitude"
    ds.xt.attrs["units"] = "deg"
    ds.yt.attrs["long_name"] = "latitude"
    ds.yt.attrs["units"] = "deg"
    ds.zt.attrs["long_name"] = "depth"
    ds.zt.attrs["units"] = "m"

.. ipython:: python

    ds.xu.attrs["long_name"] = "longitude"
    ds.xu.attrs["units"] = "deg"
    ds.yu.attrs["long_name"] = "latitude"
    ds.yu.attrs["units"] = "deg"

and convert the units of baratropic stream function (BSF) from :math:`\frac{m^{3}}{s}` to :math:`Sv` for better convenience:

.. ipython:: python

    psi = ds.psi / 1e6
    psi.attrs["units"] = "Sv"

Now, we are ready to plot BSF:

.. ipython:: python
    :okwarning:

    @savefig psi.png width=5in
    psi.isel(Time=-1).plot.contourf(levels=50)

One can, for instance, compute annual mean meridional temperature and plot it in one line command:

.. ipython:: python
    :okwarning:

    @savefig temp.png width=5in
    ds['temp'].isel(Time=-1).mean(dim='xt').plot.contourf(vmin=-2, vmax=27, levels=30, cmap='inferno')

In order to compute the decadal mean (of the last 10yrs) of meridional ocean salinity use the following similar command:

.. ipython:: python
    :okwarning:

    @savefig salt.png width=5in
    ds['salt'].isel(Time=slice(-10,None)).mean(dim=('Time', 'xt')).plot.contourf(levels=50, cmap='viridis')

Overturning circulation
-----------------------

.. ipython:: python

    ds = xr.open_dataset(OUTPUT_FILES["overturning"], decode_times=False)
    ds

.. ipython:: python
    :suppress:

    ds.xt.attrs["long_name"] = "longitude"
    ds.xt.attrs["units"] = "deg"
    ds.yt.attrs["long_name"] = "latitude"
    ds.yt.attrs["units"] = "deg"
    ds.zt.attrs["long_name"] = "depth"
    ds.zt.attrs["units"] = "m"
    ds.xu.attrs["long_name"] = "longitude"
    ds.xu.attrs["units"] = "deg"
    ds.yu.attrs["long_name"] = "latitude"
    ds.yu.attrs["units"] = "deg"
    ds.zw.attrs["long_name"] = "depth"
    ds.zw.attrs["units"] = "m"

Let's convert the units of meridional overturning circulation (MOC) from :math:`\frac{m^{3}}{s}` to :math:`Sv` and plot MOC:

.. ipython:: python
    :okwarning:

    vsf_depth = ds['vsf_depth']
    vsf_depth = ds.vsf_depth / 1e6
    vsf_depth.attrs["long_name"] = "MOC"
    vsf_depth.attrs["units"] = "Sv"

    @savefig vsf_depth_2d.png width=5in
    vsf_depth.isel(Time=-1).plot.contourf(levels=50)

Time series
-----------

To inspect coordinates ``zw``, ``yu``, ``Time`` to be used for plotting of time series execute:

.. ipython:: python

    ds['zw']
    ds['yu']
    vsf_depth['Time'].isel(Time=slice(10,))

We can see that the ``Time`` coordinate is given in days (a year corresponds to 360 days here). In order to have a more
meaningful x-axis in our figures, we divide the ``Time`` coordinate by the number of days per year and change its unit:

.. ipython:: python

    vsf_depth['Time'] = vsf_depth['Time'] / 360.
    vsf_depth.Time.attrs['units'] = 'year'

We also plot a time series of the overturning minimum between 40°N and 60°N and 550-1800m depth:

.. ipython:: python

    @savefig vsf_depth_min.png width=5in
    vsf_depth.sel(zw=slice(-1810., -550.), yu=slice(40., 60.)).min(axis=(1,2)).plot()