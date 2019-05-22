from loguru import logger

from .state import VerosState


class DistributedVerosState(VerosState):
    """A proxy wrapper to temporarily synchronize a distributed state.

    Use `gather_arrays` to retrieve distributed variables from parent VerosState object,
    and `scatter_arrays` to sync changes back.
    """
    def __init__(self, parent_state):
        object.__setattr__(self, '_vs', parent_state)
        object.__setattr__(self, '_gathered', set())

    def gather_arrays(self, arrays):
        """Gather given variables from parent state object"""
        from .distributed import gather
        for arr in arrays:
            self._gathered.add(arr)
            logger.trace(' Gathering {}', arr)
            gathered_arr = gather(
                self._vs,
                getattr(self._vs, arr),
                self._vs.variables[arr].dims
            )
            setattr(self, arr, gathered_arr)

    def scatter_arrays(self):
        """Sync all changes with parent state object"""
        from .distributed import scatter
        for arr in sorted(self._gathered):
            logger.trace(' Scattering {}', arr)
            getattr(self._vs, arr)[...] = scatter(
                self._vs,
                getattr(self, arr),
                self._vs.variables[arr].dims
            )

    def __getattribute__(self, attr):
        if attr in ('_vs', '_gathered', 'gather_arrays', 'scatter_arrays'):
            return object.__getattribute__(self, attr)

        gathered = self._gathered
        if attr in gathered:
            return object.__getattribute__(self, attr)

        parent_state = self._vs
        if attr not in parent_state.variables:
            # not a variable: pass through
            return parent_state.__getattribute__(attr)

        raise AttributeError('Cannot access distributed variable %s since it was not retrieved' % attr)

    def __setattr__(self, attr, val):
        if attr in self._gathered:
            return object.__setattr__(self, attr, val)

        if attr not in self._vs.variables:
            # not a variable: pass through
            return self._vs.__setattr__(attr, val)

        raise AttributeError('Cannot access distributed variable %s since it was not retrieved' % attr)
