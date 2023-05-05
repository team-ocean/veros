<p align="center">
<img src="doc/_images/veros-logo-400px.png?raw=true">
</p>

<p align="center">
<i>Versatile Ocean Simulation in Pure Python</i>
</p>

<p align="center">
  <a href="http://veros.readthedocs.io/?badge=latest">
    <img src="https://readthedocs.org/projects/veros/badge/?version=latest" alt="Documentation status">
  </a>
  <a href="https://github.com/team-ocean/veros/actions/workflows/test-all.yml">
    <img src="https://github.com/team-ocean/veros/actions/workflows/test-all.yml/badge.svg" alt="Test status">
  </a>
  <a href="https://codecov.io/gh/team-ocean/veros">
    <img src="https://codecov.io/gh/team-ocean/veros/branch/main/graph/badge.svg" alt="Code Coverage">
  </a>
  <a href="https://zenodo.org/badge/latestdoi/87419383">
    <img src="https://zenodo.org/badge/87419383.svg" alt="DOI">
  </a>
</p>

Veros, *the versatile ocean simulator*, aims to be the swiss army knife of ocean modeling. It is a full-fledged primitive equation ocean model that supports anything between idealized toy models and [realistic, high-resolution, global ocean simulations](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002717). And because Veros is written in pure Python, the days of struggling with complicated model setup workflows, ancient programming environments, and obscure legacy code are finally over.

*In a nutshell, we want to enable high-performance ocean modelling with a clear focus on flexibility and usability.*

Veros supports a NumPy backend for small-scale problems, and a
high-performance [JAX](https://github.com/google/jax) backend
with CPU and GPU support. It is fully parallelized via MPI and supports
distributed execution on any number of nodes, including multi-GPU architectures (see also [our benchmarks](https://veros.readthedocs.io/en/latest/more/benchmarks.html)).

The dynamical core of Veros is based on [pyOM2](https://wiki.cen.uni-hamburg.de/ifm/TO/pyOM2), an ocean model with a Fortran backend and Fortran and Python frontends.

To learn more about Veros, make sure to [visit our documentation](https://veros.readthedocs.io/en/latest/).

#### How about a demonstration?

<p align="center">
  <a href="https://vimeo.com/391237951">
      <img src="doc/_images/veros-preview.gif?raw=true" alt="0.25×0.25° high-resolution model spin-up">
  </a>
</p>

<p align="center">
(0.25×0.25° high-resolution model spin-up, click for better
quality)
</p>

## Features

Veros provides

-   a fully staggered **3-D grid geometry** (*C-grid*)
-   support for both **idealized and realistic configurations** in
    Cartesian or pseudo-spherical coordinates
-   several **friction and advection schemes**
-   isoneutral mixing, eddy-kinetic energy, turbulent kinetic energy,
    and internal wave energy **parameterizations**
-   several **pre-implemented diagnostics** such as energy fluxes,
    variable time averages, and a vertical overturning stream function
    (written to netCDF4 output)
-   **pre-configured idealized and realistic set-ups** that are ready to
    run and easy to adapt
-   **accessibility and extensibility** - thanks to the
    power of Python!

## Veros for the impatient

A minimal example to install and run Veros:

```bash
$ pip install veros
$ veros copy-setup acc --to /tmp/acc
$ veros run /tmp/acc/acc.py
```

For more detailed installation instructions, have a look at [our
documentation](https://veros.readthedocs.io).

## Basic usage

To run Veros, you need to set up a model --- i.e., specify which settings
and model domain you want to use. This is done by subclassing the
`VerosSetup` base class in a *setup script* that is written in Python. You
should use the `veros copy-setup` command to copy one into your current
folder. A good place to start is the
[ACC model](https://github.com/team-ocean/veros/blob/main/veros/setups/acc/acc.py):

```bash
$ veros copy-setup acc
```

After setting up your model, all you need to do is call the `setup` and
`run` methods on your setup class. The pre-implemented setups can all be
executed via `veros run`:

```bash
$ veros run acc.py
```

For more information on using Veros, have a look at [our
documentation](http://veros.readthedocs.io).

## Contributing

Contributions to Veros are always welcome, no matter if you spotted an
inaccuracy in [the documentation](https://veros.readthedocs.io), wrote a
new setup, fixed a bug, or even extended Veros\' core mechanics. There
are 2 ways to contribute:

1.  If you want to report a bug or request a missing feature, please
    [open an issue](https://github.com/team-ocean/veros/issues). If you
    are reporting a bug, make sure to include all relevant information
    for reproducing it (ideally through a *minimal* code sample).
2.  If you want to fix the issue yourself, or wrote an extension for
    Veros - great! You are welcome to submit your code for review by
    committing it to a repository and opening a [pull
    request](https://github.com/team-ocean/veros/pulls). However,
    before you do so, please check [the contribution
    guide](http://veros.readthedocs.io/quickstart/get-started.html#enhancing-veros)
    for some tips on testing and benchmarking, and to make sure that
    your modifications adhere with our style policies. Most importantly,
    please ensure that you follow the [PEP8
    guidelines](https://www.python.org/dev/peps/pep-0008/), use
    *meaningful* variable names, and document your code using
    [Google-style
    docstrings](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

## How to cite

If you use Veros in scientific work, please consider citing [the following publication](https://gmd.copernicus.org/articles/11/3299/2018/):

```bibtex
@article{hafner_veros_2018,
	title = {Veros v0.1 – a fast and versatile ocean simulator in pure {Python}},
	volume = {11},
	issn = {1991-959X},
	url = {https://gmd.copernicus.org/articles/11/3299/2018/},
	doi = {10.5194/gmd-11-3299-2018},
	number = {8},
	journal = {Geoscientific Model Development},
	author = {Häfner, Dion and Jacobsen, René Løwe and Eden, Carsten and Kristensen, Mads R. B. and Jochum, Markus and Nuterman, Roman and Vinter, Brian},
	month = aug,
	year = {2018},
	pages = {3299--3312},
}
```

Or have a look at [our documentation](https://veros.readthedocs.io/en/latest/more/publications.html)
for more publications involving Veros.
