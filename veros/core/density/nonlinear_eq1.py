"""
==========================================================================
  non-linear equation of state from Vallis 2008
  input is Salinity sa in g/kg,
  pot. temperature ct in deg C, no pressure dependency
==========================================================================
"""
from veros import veros_kernel, runtime_settings

rho0 = 1024.0
theta0 = 283.0 - 273.15
S0 = 35.0
betaT = 1.67e-4
betaTs = 1e-5 / 2.0
betaS = 0.78e-3
grav = 9.81
z0 = 0.0

if runtime_settings.pyom_compatibility_mode:
    import numpy as onp

    def float32(val):
        return float(onp.float32(val))

    theta0 = float(onp.float32(283.0) - onp.float32(273.15))
    rho0 = float32(rho0)
    S0 = float32(S0)
    grav = float32(grav)
    z0 = float32(z0)


@veros_kernel
def nonlin1_eq_of_state_rho(sa, ct):
    thetas = ct - theta0
    return -(betaT * thetas + betaTs * thetas ** 2 - betaS * (sa - S0)) * rho0


@veros_kernel
def nonlin1_eq_of_state_dyn_enthalpy(sa, ct, p):
    zz = -p - z0
    thetas = ct - theta0
    return grav * zz * (-betaT * thetas - betaTs * thetas ** 2 + betaS * (sa - S0))


@veros_kernel
def nonlin1_eq_of_state_salt(rho, ct):
    thetas = ct - theta0
    return (rho + (betaT * thetas + betaTs * thetas ** 2) * rho0) / (betaS * rho0) + S0


@veros_kernel
def nonlin1_eq_of_state_drhodT(ct):
    thetas = ct - theta0
    return -(betaT + 2 * betaTs * thetas) * rho0


@veros_kernel
def nonlin1_eq_of_state_drhodS():
    return betaS * rho0


@veros_kernel
def nonlin1_eq_of_state_drhodp():
    return 0.0
