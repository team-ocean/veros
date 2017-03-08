import importlib

from .timer import Timer

__all__ = [
    "model",
    "io",
    "pyom",
    "setup"
]

for module in __all__:
    importlib.import_module('.%s' % module, 'climate')
