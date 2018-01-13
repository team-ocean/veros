from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .decorators import veros_method, veros_inline_method, veros_class_method
from .veros import Veros
from .veros_legacy import VerosLegacy
