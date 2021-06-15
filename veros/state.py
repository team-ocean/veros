import contextlib
from collections import defaultdict, namedtuple
from collections.abc import Mapping
from copy import deepcopy

from veros import (
    timer,
    plugins,
    settings as settings_mod,
    variables as var_mod,
    runtime_settings as rs,
    runtime_state as rst,
)


def make_namedtuple(**kwargs):
    return namedtuple("KernelOutput", list(kwargs.keys()))(*kwargs.values())


KernelOutput = make_namedtuple


class StrictContainer:
    """A mutable container with fixed fields (optionally typed)."""

    __fields__ = ()
    __field_types__ = ()

    def __init__(self, fields, *args, field_types=None, default=None, **kwargs):
        self.__fields__ = fields

        if field_types is None:
            self.__field_types__ = {}
        else:
            if not isinstance(field_types, dict) or not set(field_types.keys()) <= set(fields):
                raise ValueError("field_types must be a dict with fields as keys")

            self.__field_types__ = field_types

        for k in fields:
            if k in vars(self):
                raise ValueError(f"Name collision: {k}")

            if k.startswith("_"):
                raise ValueError(f"Fields cannot start with _ (got: {k}).")

            super().__setattr__(k, default)

    def __setattr__(self, key, val):
        if not key.startswith("_") and key not in self.__fields__:
            raise AttributeError(f"Unknown attribute {key}")

        if key in self.__field_types__:
            val = self.__field_types__[key](val)

        return super().__setattr__(key, val)

    def __contains__(self, val):
        return val in self.__fields__

    def fields(self):
        return self.__fields__

    def values(self):
        return (getattr(self, k) for k in self.__fields__)

    def items(self):
        return ((k, getattr(self, k)) for k in self.__fields__)

    def update(self, other=None, **new_fields):
        if other is not None:
            if new_fields:
                raise ValueError("Either other or new_fields can be given")

            if hasattr(other, "_fields"):
                # other is namedtuple
                new_fields = dict(zip(other._fields, other))
            elif isinstance(other, (dict, StrictContainer)):
                new_fields = other
            else:
                raise TypeError(f"Cannot update from {type(other)} type")

        for key, val in new_fields.items():
            if key not in self.__fields__:
                raise AttributeError(f"unknown attribute {key}")

        for key, val in new_fields.items():
            setattr(self, key, val)

        return self

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __repr__(self):
        attr_str = []

        for key, val in self.items():
            # poor-man's check for array-compatible types
            if hasattr(val, "shape") and hasattr(val, "dtype"):
                val_repr = f"{type(val)} with shape {val.shape}, dtype {val.dtype}"
            else:
                val_repr = repr(val)

            attr_str.append(f"    {key} = {val_repr}")

        attr_str = ",\n".join(attr_str)

        return f"{self.__class__.__qualname__}(\n{attr_str}\n)"


class Lockable:
    __locked__ = True

    @contextlib.contextmanager
    def unlock(self):
        lock_state = self.__locked__
        try:
            self.__locked__ = False
            yield
        finally:
            self.__locked__ = lock_state

    def __setattr__(self, key, val):
        if not key.startswith("_") and self.__locked__:
            clsname = self.__class__.__qualname__
            raise RuntimeError(
                f"{clsname} is locked to modifications. If you know what you are doing, "
                f'you can unlock it via the "{clsname}.unlock()" context manager.'
            )
        return super().__setattr__(key, val)


class StaticDictProxy(Mapping):
    def __init__(self, content, writeback=None):
        self._wrapped = content
        self._writeback = writeback

    def __len__(self):
        return self._wrapped.__len__()

    def __iter__(self):
        return self._wrapped.__iter__()

    def __getitem__(self, key):
        return self._wrapped.__getitem__(key)

    def __setitem__(self, key, val):
        if key in self:
            raise RuntimeError("Cannot overwrite existing values")

        if self._writeback is not None:
            self._writeback.__setitem__(key, val)

        self._wrapped.__setitem__(key, val)

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self._wrapped!r})"


class VerosSettings(Lockable, StrictContainer):
    def __init__(self, settings_meta):
        self.__metadata__ = settings_meta
        super().__init__(fields=settings_meta.keys())

        default_settings = {k: meta.type(meta.default) for k, meta in settings_meta.items()}

        with self.unlock():
            self.update(default_settings)

    def __setattr__(self, key, val):
        if key.startswith("_") or key not in self.__metadata__:
            return super().__setattr__(key, val)

        meta = self.__metadata__[key]
        val = meta.type(val)
        return super().__setattr__(key, val)


class VerosVariables(Lockable, StrictContainer):
    """ """

    def __init__(self, var_meta, dimensions):
        self.__metadata__ = var_meta
        self.__dimensions__ = dimensions

        active_vars = [key for key, val in var_meta.items() if val.active]
        super().__init__(fields=active_vars)

        with self.unlock():
            for key, val in var_meta.items():
                if not val.active:
                    continue

                allocate_kwargs = dict(dtype=val.dtype)

                if val.initial is not None:
                    allocate_kwargs.update(fill=val.initial)

                setattr(self, key, var_mod.allocate(dimensions, val.dims, **allocate_kwargs))

    def __getattr__(self, attr):
        orig_getattr = super().__getattribute__
        try:
            var = orig_getattr("__metadata__")[attr]
        except (KeyError, AttributeError):
            return orig_getattr(attr)

        if not var.active:
            raise RuntimeError(
                f"Variable {attr} is not active in this configuration. Check your settings and try again."
            )

        return orig_getattr(attr)

    def __setattr__(self, key, val):
        if key.startswith("_") or key not in self.__metadata__:
            return super().__setattr__(key, val)

        var = self.__metadata__[key]

        # check whether variable is active
        if not var.active:
            raise RuntimeError(
                f"Variable {key} is not active in this configuration. Check your settings and try again."
            )

        # validate array type, shape and dtype
        if var.dtype is not None:
            expected_dtype = var.dtype
        else:
            expected_dtype = rs.float_type

        val = rst.backend_module.asarray(val, dtype=expected_dtype)

        expected_shape = self._get_expected_shape(var.dims)
        if val.shape != expected_shape:
            raise ValueError(f"Got unexpected shape for variable {key} (expected: {expected_shape}, got: {val.shape})")

        return super().__setattr__(key, val)

    def _get_expected_shape(self, dims):
        return var_mod.get_shape(self.__dimensions__, dims)


class DistSafeVariableWrapper(VerosVariables):
    def __init__(self, parent_state, local_variables):
        # set internal attributes to be identical to given variables object
        for attr, val in vars(parent_state).items():
            if not attr.startswith("__"):
                continue
            super().__setattr__(attr, val)

        self.__parent_state__ = parent_state
        self.__local_variables__ = local_variables

    def __getattr__(self, attr):
        orig_getattr = super().__getattribute__
        if attr in orig_getattr("__metadata__") and attr not in orig_getattr("__local_variables__"):
            raise RuntimeError(
                f"Cannot access variable {attr} because it was not collected. "
                "Consider adding it to the local_variables argument of @veros_routine."
            )

        return orig_getattr(attr)

    def __setattr__(self, attr, val):
        if attr.startswith("_"):
            return super().__setattr__(attr, val)

        if attr in self.__metadata__ and attr not in self.__local_variables__:
            raise RuntimeError(
                f"Cannot access variable {attr} because it was not collected. "
                "Consider adding it to the local_variables argument of @veros_routine."
            )

        return super().__setattr__(attr, val)

    def _gather_variables(self):
        from veros.distributed import gather

        var_meta = self.__metadata__

        for var in self.__local_variables__:
            if var not in var_meta:
                raise ValueError(f"encountered unknown variable {var} in local variables")

            if not var_meta[var].active:
                continue

            gathered_var = gather(getattr(self.__parent_state__, var), self.__dimensions__, self.__metadata__[var].dims)
            setattr(self, var, gathered_var)

    def _scatter_variables(self):
        from veros.distributed import scatter, barrier

        barrier()
        var_meta = self.__metadata__

        for var in self.__local_variables__:
            if var not in var_meta:
                raise ValueError(f"encountered unknown variable {var} in local variables")

            if not var_meta[var].active:
                continue

            scattered_var = scatter(getattr(self, var), self.__dimensions__, self.__metadata__[var].dims)
            setattr(self.__parent_state__, var, scattered_var)

    def _get_expected_shape(self, dims):
        return var_mod.get_shape(self.__dimensions__, dims, local=rst.proc_rank != 0)

    def __repr__(self):
        return f"{self.__class__.__qualname__}(parent_state={self.__parent_state__}, local_variables={self.__local_variables__})"


class VerosState:
    """Holds all settings and model state for a given Veros run."""

    def __init__(self, var_meta, setting_meta, dimensions, diagnostics=None, plugin_interfaces=None):
        self._var_meta = var_meta
        self._variables = None

        self._settings = VerosSettings(setting_meta)
        self._dimensions = dimensions

        if diagnostics is not None:
            self._diagnostics = diagnostics
        else:
            self._diagnostics = {}

        if plugin_interfaces is not None:
            self._plugin_interfaces = plugin_interfaces
        else:
            self._plugin_interfaces = ()

        timer_factory = timer.Timer
        self.timers = defaultdict(timer_factory)
        self.profile_timers = defaultdict(timer_factory)

    def __repr__(self):
        from textwrap import indent

        attr_str = []

        for attr in ("settings", "dimensions", "variables", "diagnostics", "plugin_interfaces"):
            # indent all lines of attr repr except the first
            attr_val = indent(repr(getattr(self, f"_{attr}")), " " * 4)[4:]
            attr_str.append(f"    {attr} = {attr_val}")

        attr_str = ",\n".join(attr_str)

        return f"{self.__class__.__qualname__}(\n{attr_str}\n)"

    def initialize_variables(self):
        if self._variables is not None:
            raise RuntimeError("Variables are already initialized.")

        self._var_meta = var_mod.manifest_metadata(self._var_meta, self._settings)
        self._variables = VerosVariables(self._var_meta, self._manifest_dimensions())

    @property
    def var_meta(self):
        return self._var_meta

    @property
    def variables(self):
        if self._variables is None:
            raise RuntimeError("Variables have not been initialized yet.")

        return self._variables

    @property
    def settings(self):
        return self._settings

    def _manifest_dimensions(self):
        concrete_dimensions = {}
        for dim_name, dim_target in self._dimensions.items():
            if isinstance(dim_target, str):
                dim_size = getattr(self._settings, dim_target)
            else:
                dim_size = dim_target

            concrete_dimensions[dim_name] = int(dim_size)

        return concrete_dimensions

    @property
    def dimensions(self):
        concrete_dimensions = self._manifest_dimensions()
        return StaticDictProxy(concrete_dimensions, self._dimensions)

    @property
    def diagnostics(self):
        return self._diagnostics

    @property
    def plugin_interfaces(self):
        return self._plugin_interfaces

    def to_xarray(self):
        import xarray as xr

        vs = self.variables

        coords = {}
        data_vars = {}

        for var_name, var_meta in self.var_meta.items():
            if not var_meta.active:
                continue

            data = var_mod.remove_ghosts(vs.get(var_name), var_meta.dims)

            data_vars[var_name] = xr.DataArray(
                data,
                dims=var_meta.dims,
                name=var_name,
                attrs=dict(
                    long_description=var_meta.long_description,
                    units=var_meta.units,
                    scale=var_meta.scale,
                ),
            )

            if var_meta.dims is None:
                continue

            for dim in var_meta.dims:
                if dim not in coords:
                    coords[dim] = range(var_mod.get_shape(self.dimensions, (dim,), include_ghosts=False)[0])

        data_vars = {k: v for k, v in data_vars.items() if k not in coords}

        attrs = dict(self.settings.items())

        return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def get_default_state(use_plugins=None):
    if use_plugins is not None:
        plugin_interfaces = tuple(plugins.load_plugin(p) for p in use_plugins)
    else:
        plugin_interfaces = tuple()

    default_settings = deepcopy(settings_mod.SETTINGS)

    for plugin in plugin_interfaces:
        default_settings.update(plugin.settings)

    default_dimensions = deepcopy(var_mod.DIM_TO_SHAPE_VAR)
    var_meta = deepcopy(var_mod.VARIABLES)

    for plugin in plugin_interfaces:
        var_meta.update(plugin.variables)

    return VerosState(var_meta, default_settings, default_dimensions, plugin_interfaces=plugin_interfaces)


def veros_state_pytree_flatten(state):
    aux_data = tuple((k, v) for k, v in vars(state).items() if k != "_variables")

    # ensure that functions are re-traced when settings change
    with state.settings.unlock():
        pseudo_hash = hash(tuple(state.settings.items()))

    return ([state.variables], (aux_data, pseudo_hash))


def veros_state_pytree_unflatten(aux_data, leaves):
    assert len(leaves) == 1
    variables = leaves[0]

    # by-pass __init__ and set attributes manually
    state = VerosState.__new__(VerosState)
    state._variables = variables

    state_attrs, _ = aux_data
    for attr, val in state_attrs:
        setattr(state, attr, val)

    return state


def veros_variables_pytree_flatten(variables):
    aux_attrs = (
        "__dimensions__",
        "__metadata__",
        "__fields__",
        "__locked__",
    )
    leaves = list(variables.values())
    aux_data = (tuple(variables.fields()), tuple((attr, getattr(variables, attr)) for attr in aux_attrs))
    return (leaves, aux_data)


def veros_variables_pytree_unflatten(aux_data, leaves):
    keys, aux_attrs = aux_data

    # by-pass __init__ and set attributes manually
    variables = VerosVariables.__new__(VerosVariables)

    for key, val in aux_attrs:
        setattr(variables, key, val)

    with variables.unlock():
        for key, val in zip(keys, leaves):
            setattr(variables, key, val)

    return variables


def dist_safe_wrapper_pytree_flatten(variables):
    aux_attrs = (
        "__dimensions__",
        "__metadata__",
        "__fields__",
        "__locked__",
        "__local_variables__",
        "__parent_state__",
    )
    with variables.unlock():
        leaves = [getattr(variables, attr) for attr in variables.__local_variables__]

    aux_data = (tuple(variables.__local_variables__), tuple((attr, getattr(variables, attr)) for attr in aux_attrs))
    return (leaves, aux_data)


def dist_safe_wrapper_pytree_unflatten(aux_data, leaves):
    keys, aux_attrs = aux_data

    # by-pass __init__ and set attributes manually
    variables = DistSafeVariableWrapper.__new__(DistSafeVariableWrapper)

    for key, val in aux_attrs:
        setattr(variables, key, val)

    with variables.unlock():
        for key, val in zip(keys, leaves):
            setattr(variables, key, val)

    return variables


def resize_dimension(state, dimension, new_size):
    """Resize a dimension of an existing VerosState object.

    This re-allocates all variables using the dimension to 0.
    """
    state._dimensions[dimension] = new_size
    state.variables.__dimensions__[dimension] = new_size

    with state.variables.unlock():
        for var in state.variables.fields():
            var_meta = state.variables.__metadata__[var]
            var_dims = var_meta.dims

            if var_dims is None or dimension not in var_dims:
                continue

            setattr(state.variables, var, var_mod.allocate(state.dimensions, var_meta.dims, dtype=var_meta.dtype))
