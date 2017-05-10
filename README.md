![Veros](doc/_images/veros-logo-400px.png?raw=true)

[![Documentation Status](https://readthedocs.org/projects/veros/badge/?version=latest)](http://veros.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://travis-ci.org/dionhaefner/veros.svg?branch=master)](https://travis-ci.org/dionhaefner/veros)

Veros, the *versatile ocean simulator*, is just that: A powerful tool that makes high-performance ocean modeling approachable and fun. Since it is a pure Python module, the days of struggling with complicated model setup workflows, ancient programming environments, and obscure legacy code are finally over.

Veros supports both a NumPy backend for small-scale problems and a fully parallelized high-performance backend [powered by Bohrium](https://github.com/bh107/bohrium) using either OpenMP (CPU) or OpenCL (GPU). The underlying numerics are based on [pyOM2](https://wiki.cen.uni-hamburg.de/ifm/TO/pyOM2), an ocean model developed by Carsten Eden (Institut für Meereskunde, Hamburg University).

Veros is currently being developed at Niels Bohr Institute, Copenhagen University.

## Features

Veros provides

- a fully staggered 3-D grid geometry (*C-grid*)
- support for both idealized and realistic configurations in cartesian or pseudo-spherical coordinates
- several friction and advection schemes to choose from
- isoneutral mixing, eddy-kinetic energy, turbulent kinetic energy, and internal wave energy parameterizations
- several pre-implemented diagnostics such as energy fluxes, variable time averages, and a vertical overturning stream function (written to netCDF output)
- pre-configured idealized and realistic set-ups that are ready to run and easy to adapt
- accessibility, readability, and extensibility - thanks to the power of Python!

## Installation

### Dependencies

Currently, the only officially supported Python version for Veros is Python 2.7 (64-bit), which should be the standard Python interpreter on most systems.

Veros only has two external library dependencies, namely `HDF5` and `netCDF`. The installation procedure of those libraries varies between platforms. Probably the easiest way to install Veros and its dependencies (includung Bohrium) is [Anaconda Python](https://www.continuum.io/downloads) that ships with a package manager (``conda``).

If you do not want to use Anaconda, the most convenient way is to use your operating system's package manager. On Debian / Ubuntu, you can e.g. use

    $ sudo apt-get install libhdf5-dev libnetcdf-dev

Similar package managers on OSX are [Homebrew](https://brew.sh/) or [MacPorts](https://www.macports.org/), which both provide the required dependencies as pre-compiled binaries.

### Installing Veros

As soon as you have a working environment, installing Veros is simple:

1. Clone the repository to your hard-drive:

       $ git clone https://github.com/dionhaefner/veros.git

   Note that you need to have [Git LFS](https://git-lfs.github.com/) installed if you want to download the forcing files required for running the larger models.

2. Install it, preferably with

       $ pip install -e veros

   If you use the `-e` flag, any changes you make to the model code are immediately reflected without having to re-install.

In case you want to use the Bohrium backend, you will have to install [Bohrium](https://github.com/bh107/bohrium), e.g. through `conda` or `apt-get`, or by building it from source.

## Basic usage

To run Veros, you need to set up a model - i.e., specify which settings and model domain you want to use. This is done by subclassing the ``Veros`` base class in a *setup script* that is written in Python. You should have a look at the pre-implemented model setups in the repository's ``setup`` folder, or use the ``veros copy-setup`` command to copy one into your current folder. A good place to start is the [``ACC2`` model](https://github.com/dionhaefner/veros/blob/master/setup/acc2/acc2.py).

After setting up your model, all you need to do is call the ``setup`` and ``run`` methods on your setup class. The pre-implemented setups can all be executed as scripts, e.g. through

    $ python acc2.py

For more information on using Veros, have a look at [our documentation](http://veros.readthedocs.io/en/latest/).
