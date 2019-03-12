import math

from . import variables, settings


class VerosState(object):
    """Holds all settings and model state for a given Veros run."""
    # Constants
    pi = math.pi
    radius = 6370e3  # Earth radius in m
    degtom = radius / 180. * pi  # Conversion degrees latitude to meters
    mtodeg = 1. / degtom  # Conversion meters to degrees latitude
    omega = pi / 43082.  # Earth rotation frequency in 1/s
    rho_0 = 1024.  # Boussinesq reference density in :math:`kg/m^3`
    grav = 9.81  # Gravitational constant in :math:`m/s^2`

    def __init__(self):
        self.variables = {}
        self.poisson_solver = None
        self.nisle = 0 # to be overriden during streamfunction_init
        self.taum1, self.tau, self.taup1 = 0, 1, 2 # pointers to last, current, and next time step
        self.time, self.itt = 0., 1 # current time and iteration

        settings.set_default_settings(self)

    def allocate_variables(self):
        self.variables.update(variables.get_standard_variables(self))
        variables.allocate_variables(self)
