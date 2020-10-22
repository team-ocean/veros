import abc
import functools
import contextlib
from collections import defaultdict, namedtuple

from veros import (
    timer, plugins, backend,
    settings as settings_mod, variables as var_mod,
    runtime_settings as rs, runtime_state as rst
)


class Container:
    """A simple, mutable container"""
    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        for key, val in kwargs.items():
            self.__setattr__(key, val)

        return self

    def __repr__(self):
        attr_str = ',\n'.join(f'    {key}={val!s}' for key, val in vars(self).items())
        return f'{self.__class__.__qualname__}(\n{attr_str}\n)'

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class LockableContainer(Container):
    __slots__ = ()
    __locked__ = True

    @contextlib.contextmanager
    def unlock(self):
        force_setattr = super().__setattr__
        if self.__locked__:
            reset = True
            force_setattr('__locked__', False)

        try:
            yield

        finally:
            if reset:
                force_setattr('__locked__', True)

    def __setattr__(self, key, val):
        if self.__locked__:
            raise RuntimeError(
                '{0} is locked to modifications. If you know what you are doing, '
                'you can unlock it via the "{0}.unlock()" context manager.'
                .format(self.__class__.__qualname__)
            )
        return super().__setattr__(key, val)


class _VariableContainer(LockableContainer, metaclass=abc.ABCMeta):
    """A mutable container with fixed __slots__ and validation on __setattr__.

    Must be subclassed (to populate __slots__).
    """
    __slots__ = ('__meta__',)

    def __init__(self, __meta__):
        super().__init__(__meta__=__meta__)

        for key, var in __meta__.items():
            setattr(self, key, var_mod.allocate(self, var.dims, dtype=var.dtype))

    def __setattr__(self, key, val):
        # validate array type, shape and dtype
        var = self.__meta__[key]

        if var.dtype is not None:
            expected_dtype = var.dtype
        else:
            expected_dtype = rs.float_type

        val = rst.backend_module.asarray(val, dtype=expected_dtype)

        expected_shape = var_mod.get_dimensions(self, var.dims)
        if val.shape != expected_shape:
            raise ValueError(
                'Got unexpected shape for variable {} (expected: {}, got: {})'
                .format(key, val.shape, expected_shape)
            )

        return super().__setattr__(key, val)


def create_variable_container(variables):
    """Factory for variable containers"""
    return type('VariableContainer', _VariableContainer, dict(__slots__=variables.keys()))(variables)


class SettingsContainer(LockableContainer):
    __slots__ = tuple(settings_mod.SETTINGS.keys())


class VerosState:
    """Holds all settings and model state for a given Veros run."""
    __slots__ = (
        '_settings',
        '_variables',
        '_diagnostics',
        '_objects',
        '_dimensions',
        'dimension_vars'
    )

    def __init__(self, variables, settings, diagnostics, objects):
        self.dimension_vars = list(var_mod.DIMENSION_SETTINGS)

        self._settings = SettingsContainer()
        self._variables = VariableContainer(variables)
        self._diagnostics = {}
        self._objects = Container()
        self._dimensions = {}

        self.poisson_solver = None
        self.nisle = 0 # to be overwritten during streamfunction_init
        self.taum1, self.tau, self.taup1 = 0, 1, 2 # pointers to last, current, and next time step
        self.time, self.itt = 0., 0 # current time and iteration

        if use_plugins is not None:
            self._plugin_interfaces = tuple(plugins.load_plugin(p) for p in use_plugins)
        else:
            self._plugin_interfaces = tuple()

        settings_mod.set_default_settings(self._settings)

        for plugin in self._plugin_interfaces:
            settings.update_settings(self, plugin.settings)

        timer_factory = functools.partial(timer.Timer, inactive=True)
        self.timers = defaultdict(timer_factory)
        self.profile_timers = defaultdict(timer_factory)

    def __repr__(self):
        # TODO: write me
        return f'{self.__class__.__qualname__}()'

    @property
    def dimensions(self):
        def _get_val(dim):
            if hasattr(self.settings, dim):
                return getattr(self.settings, dim)
            if hasattr(self.objects, dim):
                return getattr(self.objects, dim)
            raise AttributeError(f'unknown value in dimension_vars: {dim}')

        vals = tuple(_get_val(dim) for dim in self.dimension_vars)

        if vals not in self._dimensions:
            dimension_type = namedtuple('Dimensions', self.dimension_vars)
            self._dimensions[vals] = dimension_type(vals)

        return self._dimensions[vals]

    @property
    def variables(self):
        if self._variables is None:
            raise RuntimeError()
        return self._variables

    @property
    def settings(self):
        return self._settings

    diagnostics = property()

    @diagnostics.getter
    def diagnostics(self):
        return self._diagnostics

    @diagnostics.setter
    def diagnostics(self, val):
        if not isinstance(val, dict):
            raise TypeError('Diagnostics object must be dict-like')

        self._diagnostics = val

    @property
    def objects(self):
        return self._objects

    def restrict(self, var_subset):
        pass

    def allocate_variables(self):
        self.variables.update(variables.get_standard_variables(self))

        for plugin in self._plugin_interfaces:
            plugin_vars = variables.get_active_variables(self, plugin.variables, plugin.conditional_variables)
            self.variables.update(plugin_vars)

        container_type = type('VerosVariables', VariableContainer, dict(__slots__=variables))
        self._variables = container_type(self._var_meta)

    def create_diagnostics(self):
        from veros import diagnostics
        self.diagnostics.update(diagnostics.create_default_diagnostics(self))

        for plugin in self._plugin_interfaces:
            for diagnostic in plugin.diagnostics:
                self.diagnostics[diagnostic.name] = diagnostic(self)

    def to_xarray(self):
        import xarray as xr

        coords = {}
        data_vars = {}

        for var_name, var in self.variables.items():
            data = var_mod.remove_ghosts(
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
                            coords[dim] = var_mod.remove_ghosts(dim_val, (dim,))
                    else:
                        coords[dim] = range(var_mod.get_dimensions(self, (dim,))[0])

        data_vars = {k: v for k, v in data_vars.items() if k not in coords}

        attrs = dict(
            time=self.time,
            iteration=self.itt,
            tau=self.tau,
        )

        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    def check_integrity(self):
        pass


class RestrictedVerosState:
    """A proxy wrapper around VerosState allowing access only to some variables.

    Use `gather_arrays` to retrieve distributed variables from parent VerosState object,
    and `scatter_arrays` to sync changes back.
    """

    def __init__(self, parent_state):
        super().__setattr__('_vs', parent_state)
        super().__setattr__('_gathered', set())

    def _gather_arrays(self, arrays, flush=False):
        """Gather given variables from parent state object"""
        from veros.distributed import gather
        for arr in arrays:
            # TODO: handle this
            if arr not in self._vs.variables:
                continue

            self._gathered.add(arr)
            gathered_arr = gather(
                self._vs.nx, self._vs.ny,
                getattr(self._vs, arr),
                self._vs.variables[arr].dims
            )

            if flush:
                backend.flush(gathered_arr)

            setattr(self, arr, gathered_arr)

    def _scatter_arrays(self, arrays, flush=False):
        """Sync all changes with parent state object"""
        from veros.distributed import scatter
        for arr in arrays:
            if flush:
                backend.flush(arr)

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
