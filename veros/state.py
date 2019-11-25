import abc
import math

from . import variables, settings, timer, plugins, diagnostics


class VerosStateBase(metaclass=abc.ABCMeta):
    pass


class VerosState(VerosStateBase):
    """Holds all settings and model state for a given Veros run."""
    # Constants
    pi = math.pi
    radius = 6370e3  # Earth radius in m
    degtom = radius / 180. * pi  # Conversion degrees latitude to meters
    mtodeg = 1. / degtom  # Conversion meters to degrees latitude
    omega = pi / 43082.  # Earth rotation frequency in 1/s
    rho_0 = 1024.  # Boussinesq reference density in :math:`kg/m^3`
    grav = 9.81  # Gravitational constant in :math:`m/s^2`

    def __init__(self, use_plugins=None):
        self.variables = {}
        self.diagnostics = {}
        self.poisson_solver = None
        self.nisle = 0 # to be overriden during streamfunction_init
        self.taum1, self.tau, self.taup1 = 0, 1, 2 # pointers to last, current, and next time step
        self.time, self.itt = 0., 0 # current time and iteration

        if use_plugins is not None:
            self._plugin_interfaces = tuple(plugins.load_plugin(p) for p in use_plugins)
        else:
            self._plugin_interfaces = tuple()

        settings.set_default_settings(self)

        for plugin in self._plugin_interfaces:
            settings.update_settings(self, plugin.settings)

        self.timers = {k: timer.Timer() for k in (
            'setup', 'main', 'momentum', 'temperature', 'eke', 'idemix',
            'tke', 'diagnostics', 'pressure', 'friction', 'isoneutral',
            'vmix', 'eq_of_state', 'plugins'
        )}

        for plugin in self._plugin_interfaces:
            self.timers[plugin.name] = timer.Timer()

    def allocate_variables(self):
        self.variables.update(variables.get_standard_variables(self))

        for plugin in self._plugin_interfaces:
            plugin_vars = variables.get_active_variables(self, plugin.variables, plugin.conditional_variables)
            self.variables.update(plugin_vars)

        for key, var in self.variables.items():
            setattr(self, key, variables.allocate(self, var.dims, dtype=var.dtype))

    def create_diagnostics(self):
        self.diagnostics.update(diagnostics.create_default_diagnostics(self))

        for plugin in self._plugin_interfaces:
            for diagnostic in plugin.diagnostics:
                self.diagnostics[diagnostic.name] = diagnostic(self)

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
