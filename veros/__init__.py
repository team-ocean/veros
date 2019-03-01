from __future__ import absolute_import

from veros._version import get_versions
__version__ = get_versions()['version']
del get_versions

# for some reason, netCDF4 and mpi4py have to be imported before h5py
try:
    from mpi4py import MPI
except ImportError:
    pass

import netCDF4
import h5py

del mpi4py
del netCDF4
del h5py

# runtime settings object
from veros.runtime import RuntimeSettings, RuntimeState
runtime_settings = RuntimeSettings()
runtime_state = RuntimeState()
del RuntimeSettings, RuntimeState

# logging
import veros.logs

# public API
from veros.decorators import veros_method
from veros.veros import VerosSetup
from veros.state import VerosState
from veros.veros_legacy import VerosLegacy
