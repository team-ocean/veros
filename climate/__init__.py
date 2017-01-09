import importlib

from .timer import Timer

__all__ = [
    "data",
    "io",
    "pyom"
]

for module in __all__:
    importlib.import_module('.%s' % module, 'climate')
