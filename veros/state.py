import abc
import math
import contextlib
from loguru import logger

from veros import (
    variables, settings, timer, plugins,
    runtime_settings as rs, runtime_state as rst
)


class TimerContainer:
    def __init__(self, start_inactive=True):
        self._timers = {}
        self._start_inactive = start_inactive

    def __getitem__(self, key):
        if key not in self._timers:
            self._timers[key] = timer.Timer(self._start_inactive)

        return self._timers[key]

    def __setitem__(self, key, val):
        return self._timers.__setitem__(key, val)

    def __iter__(self):
        return self._timers.__iter__

    def __next__(self):
        return self._timers.__next__

    def keys(self):
        return self._timers.keys()

    def items(self):
        return self._timers.items()

    def values(self):
        return self._timers.values()


class VerosStateBase(metaclass=abc.ABCMeta):
    pass


class VerosState(VerosStateBase):
    """Holds all settings and model state for a given Veros run."""
    __locked__ = False

    def __init__(self, use_plugins=None):
        self.variables = {}
        self.diagnostics = {}
        self.poisson_solver = None
        self.nisle = 0 # to be overwritten during streamfunction_init
        self.taum1, self.tau, self.taup1 = 0, 1, 2 # pointers to last, current, and next time step
        self.time, self.itt = 0., 0 # current time and iteration

        if use_plugins is not None:
            self._plugin_interfaces = tuple(plugins.load_plugin(p) for p in use_plugins)
        else:
            self._plugin_interfaces = tuple()

        settings.set_default_settings(self)

        for plugin in self._plugin_interfaces:
            settings.update_settings(self, plugin.settings)

        self.timers = TimerContainer(start_inactive=True)
        self.profile_timers = TimerContainer(start_inactive=True)
        self.__locked__ = True

    @contextlib.contextmanager
    def unlock(self):
        try:
            self.__locked__ = False
            yield
        finally:
            self.__locked__ = True

    def allocate_variables(self):
        self.variables.update(variables.get_standard_variables(self))

        for plugin in self._plugin_interfaces:
            plugin_vars = variables.get_active_variables(self, plugin.variables, plugin.conditional_variables)
            self.variables.update(plugin_vars)

        with self.unlock():
            for key, var in self.variables.items():
                setattr(self, key, variables.allocate(self, var.dims, dtype=var.dtype))

    def create_diagnostics(self):
        from veros import diagnostics
        self.diagnostics.update(diagnostics.create_default_diagnostics(self))

        for plugin in self._plugin_interfaces:
            for diagnostic in plugin.diagnostics:
                self.diagnostics[diagnostic.name] = diagnostic(self)

    def __setattr__(self, key, val):
        if not hasattr(self, 'variables') or key not in self.variables:
            if self.__locked__ and not hasattr(self, key):
                raise AttributeError('Unknown attribute {}'.format(key))

            return super().__setattr__(key, val)

        # validate array type, shape and dtype
        var = self.variables[key]

        if var.dtype is not None:
            expected_dtype = var.dtype
        else:
            expected_dtype = rs.float_type

        val = rst.backend_module.asarray(val, dtype=expected_dtype)

        expected_shape = variables.get_dimensions(self, var.dims)
        if val.shape != expected_shape:
            raise ValueError(
                'Got unexpected shape for variable {} (expected: {}, got: {})'
                .format(key, val.shape, expected_shape)
            )

        return super().__setattr__(key, val)

    def to_xarray(self):
        import xarray as xr

        coords = {}
        data_vars = {}

        for var_name, var in self.variables.items():
            data = variables.remove_ghosts(
                getattr(self, var_name), var.dims
            )
            data_vars[var_name] = xr.DataArray(
                data,
                dims=var.dims,
                name=var_name,
                attrs=dict(
                    long_description=var.long_description,
                    units=var.units,
                    scale=var.scale,
                )
            )

            for dim in var.dims:
                if dim not in coords:
                    if hasattr(self, dim):
                        dim_val = getattr(self, dim)
                        if isinstance(dim_val, int):
                            coords[dim] = range(dim_val)
                        else:
                            coords[dim] = variables.remove_ghosts(dim_val, (dim,))
                    else:
                        coords[dim] = range(variables.get_dimensions(self, (dim,))[0])

        data_vars = {k: v for k, v in data_vars.items() if k not in coords}

        attrs = dict(
            time=self.time,
            iteration=self.itt,
            tau=self.tau,
        )

        return xr.Dataset(data_vars, coords=coords, attrs=attrs)


class RestrictedVerosState(VerosStateBase):
    """A proxy wrapper around VerosState allowing access only to some variables.

    Use `gather_arrays` to retrieve distributed variables from parent VerosState object,
    and `scatter_arrays` to sync changes back.
    """

    def __init__(self, parent_state):
        super().__setattr__('_vs', parent_state)
        super().__setattr__('_gathered', set())

    def _gather_arrays(self, arrays):
        """Gather given variables from parent state object"""
        from .distributed import gather
        for arr in arrays:
            if arr not in self._vs.variables:
                continue

            self._gathered.add(arr)
            logger.trace(' Gathering {}', arr)
            gathered_arr = gather(
                self._vs.nx, self._vs.ny,
                getattr(self._vs, arr),
                self._vs.variables[arr].dims
            )
            setattr(self, arr, gathered_arr)

    def _scatter_arrays(self, arrays):
        """Sync all changes with parent state object"""
        from .distributed import scatter
        for arr in arrays:
            logger.trace(' Scattering {}', arr)
            setattr(self._vs, arr, scatter(
                self._vs.nx, self._vs.ny,
                getattr(self, arr),
                self._vs.variables[arr].dims
            ))

    def __getattribute__(self, attr):
        if attr in ('_vs', '_gathered', '_gather_arrays', '_scatter_arrays'):
            return super().__getattribute__(attr)

        gathered = self._gathered
        if attr in gathered:
            return super().__getattribute__(attr)

        parent_state = self._vs
        if attr not in parent_state.variables:
            # not a variable: pass through
            return parent_state.__getattribute__(attr)

        raise AttributeError('Cannot access variable %s since it was not retrieved' % attr)

    def __setattr__(self, attr, val):
        if attr in self._gathered:
            return super().__setattr__(attr, val)

        if attr not in self._vs.variables:
            # not a variable: pass through
            return self._vs.__setattr__(attr, val)

        raise AttributeError('Cannot access variable %s since it was not retrieved' % attr)

    def __repr__(self):
        return '{}(parent_state={})'.format(self.__class__.__name__, repr(self._vs))
