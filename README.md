# Veros
[![Documentation Status](https://readthedocs.org/projects/veros/badge/?version=latest)](http://veros.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://travis-ci.org/dionhaefner/veros.svg?branch=master)](https://travis-ci.org/dionhaefner/veros)

Veros, the *versatile ocean simulator*, is just that: A powerful tool that makes high-performance ocean modelling approachable and fun. Since it is a pure Python module, the days of struggling with complicated model setup workflows, ancient programming environments, and obscure legacy code are finally over.

Veros supports both a NumPy backend for small-scale problems and a fully parallelized high-performance backend [powered by Bohrium](https://github.com/bh107/bohrium). The underlying numerics are based on [pyOM2](https://wiki.cen.uni-hamburg.de/ifm/TO/pyOM2), developed by Carsten Eden at Hamburg Univeristy.

## Installation

Installing Veros can be easy as pie:

1. Clone the repository to your hard-drive:

     git clone https://github.com/dionhaefner/veros.git

2. Install it, e.g. with

     pip install -e veros

  (make sure to use the `-e` flag, so any changes you make to the model code are immediately reflected without having to re-install)

In case you want to use the Bohrium backend, you will have to install [Bohrium](https://github.com/bh107/bohrium), e.g. through `conda` or `apt-get`, or by building it from source.
