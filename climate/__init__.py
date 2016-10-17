import importlib

__all__ = [
    "data"
]

for module in __all__:
    importlib.import_module('.%s' % module, 'climate')
