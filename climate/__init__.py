import importlib
import numpy as np

from .timer import Timer
from .slice import make_slice

__all__ = [
    "data",
    "io",
    "pyom",
    "setup"
]

is_bohrium = np.__name__ == 'bohrium'

for module in __all__:
    importlib.import_module('.%s' % module, 'climate')

