import importlib

from .timer import Timer

__all__ = [
    "data",
    "io",
    "boussinesq"
]

for module in __all__:
    importlib.import_module('.%s' % module, 'climate')
