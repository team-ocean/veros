import importlib
import numpy as np

from .timer import Timer

__all__ = [
    "model",
    "io",
    "pyom",
    "setup"
]

is_bohrium = np.__name__ == 'bohrium'

for module in __all__:
    importlib.import_module('.%s' % module, 'climate')

