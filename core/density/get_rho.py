from . import (gsw, linear_eq as lq, nonlinear_eq1 as nq1,
               nonlinear_eq2 as nq2, nonlinear_eq3 as nq3)
from ... import veros_method


@veros_method
def get_rho(vs, salt_loc, temp_loc, press):
    """
    calculate density as a function of temperature, salinity and pressure
    """
    if vs.eq_of_state_type == 1:
        return lq.linear_eq_of_state_rho(salt_loc, temp_loc)
    elif vs.eq_of_state_type == 2:
        return nq1.nonlin1_eq_of_state_rho(salt_loc, temp_loc)
    elif vs.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_rho(salt_loc, temp_loc, press)
    elif vs.eq_of_state_type == 4:
        return nq3.nonlin3_eq_of_state_rho(salt_loc, temp_loc)
    elif vs.eq_of_state_type == 5:
        return gsw.gsw_rho(vs, salt_loc, temp_loc, press)
    else:
        raise ValueError('unknown equation of state')


@veros_method
def get_potential_rho(vs, salt_loc, temp_loc, press):
    """
    calculate potential density as a function of temperature, salinity
    and pressure

    NB! The potential density is computed for equation of state No.5
    according to TEOS-10 formulation, for other equations of state (No.1-4)
    it is in-situ density from get_rho
    """
    if vs.eq_of_state_type == 1:
        return lq.linear_eq_of_state_rho(salt_loc, temp_loc)
    elif vs.eq_of_state_type == 2:
        return nq1.nonlin1_eq_of_state_rho(salt_loc, temp_loc)
    elif vs.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_rho(salt_loc, temp_loc, press)
    elif vs.eq_of_state_type == 4:
        return nq3.nonlin3_eq_of_state_rho(salt_loc, temp_loc)
    elif vs.eq_of_state_type == 5:
        return gsw.gsw_pot_rho_ct(vs, salt_loc, temp_loc)
    else:
        raise ValueError('unknown equation of state')


@veros_method
def get_dyn_enthalpy(vs, salt_loc, temp_loc, press):
    """
    calculate dynamic enthalpy as a function of temperature, salinity and pressure
    """
    if vs.eq_of_state_type == 1:
        return lq.linear_eq_of_state_dyn_enthalpy(salt_loc, temp_loc, press)
    elif vs.eq_of_state_type == 2:
        return nq1.nonlin1_eq_of_state_dyn_enthalpy(salt_loc, temp_loc, press)
    elif vs.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_dyn_enthalpy(salt_loc, temp_loc, press)
    elif vs.eq_of_state_type == 4:
        return nq3.nonlin3_eq_of_state_dyn_enthalpy(salt_loc, temp_loc, press)
    elif vs.eq_of_state_type == 5:
        return gsw.gsw_dyn_enthalpy(vs, salt_loc, temp_loc, press)
    else:
        raise ValueError('unknown equation of state')


@veros_method
def get_salt(vs, rho_loc, temp_loc, press_loc):
    """
    calculate salinity as a function of density, temperature and pressure
    """
    if vs.eq_of_state_type == 1:
        return lq.linear_eq_of_state_salt(rho_loc, temp_loc)
    elif vs.eq_of_state_type == 2:
        return nq1.nonlin1_eq_of_state_salt(rho_loc, temp_loc)
    elif vs.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_salt(rho_loc, temp_loc, press_loc)
    elif vs.eq_of_state_type == 4:
        return nq3.nonlin3_eq_of_state_salt(rho_loc, temp_loc)
    else:
        raise ValueError('unknown equation of state')


@veros_method
def get_drhodT(vs, salt_loc, temp_loc, press_loc):
    """
    calculate drho/dT as a function of temperature, salinity and pressure
    """
    if vs.eq_of_state_type == 1:
        return lq.linear_eq_of_state_drhodT()
    elif vs.eq_of_state_type == 2:
        return nq1.nonlin1_eq_of_state_drhodT(temp_loc)
    elif vs.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_drhodT(temp_loc, press_loc)
    elif vs.eq_of_state_type == 4:
        return nq3.nonlin3_eq_of_state_drhodT(temp_loc)
    elif vs.eq_of_state_type == 5:
        return gsw.gsw_drhodT(vs, salt_loc, temp_loc, press_loc)
    else:
        raise ValueError('unknown equation of state')


@veros_method
def get_drhodS(vs, salt_loc, temp_loc, press_loc):
    """
    calculate drho/dS as a function of temperature, salinity and pressure
    """
    if vs.eq_of_state_type == 1:
        return lq.linear_eq_of_state_drhodS()
    elif vs.eq_of_state_type == 2:
        return nq1.nonlin1_eq_of_state_drhodS()
    elif vs.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_drhodS()
    elif vs.eq_of_state_type == 4:
        return nq3.nonlin3_eq_of_state_drhodS()
    elif vs.eq_of_state_type == 5:
        return gsw.gsw_drhodS(vs, salt_loc, temp_loc, press_loc)
    else:
        raise ValueError('unknown equation of state')


@veros_method
def get_drhodp(vs, salt_loc, temp_loc, press_loc):
    """
    calculate drho/dP as a function of temperature, salinity and pressure
    """
    if vs.eq_of_state_type == 1:
        return lq.linear_eq_of_state_drhodp()
    elif vs.eq_of_state_type == 2:
        return nq1.nonlin1_eq_of_state_drhodp()
    elif vs.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_drhodp(temp_loc)
    elif vs.eq_of_state_type == 4:
        return nq3.nonlin3_eq_of_state_drhodp()
    elif vs.eq_of_state_type == 5:
        return gsw.gsw_drhodp(vs, salt_loc, temp_loc, press_loc)
    else:
        raise ValueError('unknown equation of state')


@veros_method
def get_int_drhodT(vs, salt_loc, temp_loc, press_loc):
    """
    calculate int_z^0 drho/dT dz' as a function of temperature, salinity and pressure
    """
    if vs.eq_of_state_type == 1:
        return press_loc * lq.linear_eq_of_state_drhodT()  # int_z^0rho_T dz = - rho_T z
    elif vs.eq_of_state_type == 2:
        return press_loc * nq1.nonlin1_eq_of_state_drhodT(temp_loc)
    elif vs.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_int_drhodT(temp_loc, press_loc)
    elif vs.eq_of_state_type == 4:
        return press_loc * nq3.nonlin3_eq_of_state_drhodT(temp_loc)
    elif vs.eq_of_state_type == 5:
        return -(1024.0 / 9.81) * gsw.gsw_dHdT(vs, salt_loc, temp_loc, press_loc)
    else:
        raise ValueError('unknown equation of state')


@veros_method
def get_int_drhodS(vs, salt_loc, temp_loc, press_loc):
    """
    calculate int_z^0 drho/dS dz' as a function of temperature, salinity and pressure
    """
    if vs.eq_of_state_type == 1:
        return press_loc * lq.linear_eq_of_state_drhodS()  # int_z^0rho_T dz = - rho_T z
    elif vs.eq_of_state_type == 2:
        return press_loc * nq1.nonlin1_eq_of_state_drhodS()
    elif vs.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_int_drhodS(press_loc)
    elif vs.eq_of_state_type == 4:
        return press_loc * nq3.nonlin3_eq_of_state_drhodS()
    elif vs.eq_of_state_type == 5:
        return -(1024.0 / 9.81) * gsw.gsw_dHdS(vs, salt_loc, temp_loc, press_loc)
    else:
        raise ValueError('unknown equation of state')
