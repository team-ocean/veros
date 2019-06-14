"""Veros, the versatile ocean simulator"""

import sys
import types

# black magic: ensure lazy imports for public API by overriding module.__class__

class _PublicAPI(types.ModuleType):
    @property
    def __version__(self):
        from veros._version import get_versions
        return get_versions()['version']

    @property
    def runtime_settings(self):
        if not hasattr(self, '_runtime_settings'):
            from veros.runtime import RuntimeSettings
            self._runtime_settings = RuntimeSettings()
        return self._runtime_settings

    @property
    def runtime_state(self):
        if not hasattr(self, '_runtime_state'):
            from veros.runtime import RuntimeState
            self._runtime_state = RuntimeState()
        return self._runtime_state

    @property
    def veros_method(self):
        from veros.decorators import veros_method
        return veros_method

    @property
    def VerosSetup(self):
        from veros.veros import VerosSetup
        return VerosSetup

    @property
    def VerosState(self):
        from veros.state import VerosState
        return VerosState

    @property
    def VerosLegacy(self):
        from veros.veros_legacy import VerosLegacy
        return VerosLegacy

sys.modules[__name__].__class__ = _PublicAPI

del sys
del types
del _PublicAPI
