import gsw
import linear_eq as lq
import nonlinear_eq1 as nq1
import nonlinear_eq2 as nq2
import nonlinear_eq3 as nq3

def get_rho(salt_loc,temp_loc,press,pyom):
    """
    calculate density as a function of temperature, salinity and pressure
    """
    if pyom.eq_of_state_type == 1:
        return lq.linear_eq_of_state_rho(salt_loc,temp_loc)
    elif pyom.eq_of_state_type == 2:
        return nq1.nonlin1_eq_of_state_rho(salt_loc,temp_loc)
    elif pyom.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_rho(salt_loc,temp_loc,press)
    elif pyom.eq_of_state_type == 4:
        return nq3.nonlin3_eq_of_state_rho(salt_loc,temp_loc)
    elif pyom.eq_of_state_type == 5:
        return gsw.gsw_rho(salt_loc,temp_loc,press)
    else:
        halt_stop(' unknown equation of state in get_rho')


def get_dyn_enthalpy(salt_loc,temp_loc,press,pyom):
    """
    calculate dynamic enthalpy as a function of temperature, salinity and pressure
    """
    if pyom.eq_of_state_type == 1:
        return lq.linear_eq_of_state_dyn_enthalpy(salt_loc,temp_loc,press)
    elif pyom.eq_of_state_type == 2:
        return nq1.nonlin1_eq_of_state_dyn_enthalpy(salt_loc,temp_loc,press)
    elif pyom.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_dyn_enthalpy(salt_loc,temp_loc,press)
    elif pyom.eq_of_state_type == 4:
        return nq3.nonlin3_eq_of_state_dyn_enthalpy(salt_loc,temp_loc,press)
    elif pyom.eq_of_state_type == 5:
        return gsw.gsw_dyn_enthalpy(salt_loc,temp_loc,press)
    else:
        halt_stop(' unknown equation of state in get_dyn_enthalpy')


def get_salt(rho_loc,temp_loc,press_loc,pyom):
    """
    calculate salinity as a function of density, temperature and pressure
    """
    if pyom.eq_of_state_type == 1:
        return lq.linear_eq_of_state_salt(rho_loc,temp_loc)
    elif pyom.eq_of_state_type == 2:
        return nq1.nonlin1_eq_of_state_salt(rho_loc,temp_loc)
    elif pyom.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_salt(rho_loc,temp_loc,press_loc)
    elif pyom.eq_of_state_type == 4:
        return nq3.nonlin3_eq_of_state_salt(rho_loc,temp_loc)
    else:
        halt_stop(' unknown equation of state in get_salt')


def get_drhodT(salt_loc,temp_loc,press_loc,pyom):
    """
    calculate drho/dT as a function of temperature, salinity and pressure
    """
    if pyom.eq_of_state_type == 1:
        return lq.linear_eq_of_state_drhodT()
    elif pyom.eq_of_state_type == 2:
        return nq1.nonlin1_eq_of_state_drhodT(temp_loc)
    elif pyom.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_drhodT(temp_loc,press_loc)
    elif pyom.eq_of_state_type == 4:
        return nq3.nonlin3_eq_of_state_drhodT(temp_loc)
    elif pyom.eq_of_state_type == 5:
        return gsw.gsw_drhodT(salt_loc,temp_loc,press_loc)
    else:
        halt_stop(' unknown equation of state in get_drhodT')


def get_drhodS(salt_loc,temp_loc,press_loc,pyom):
    """
    calculate drho/dS as a function of temperature, salinity and pressure
    """
    if pyom.eq_of_state_type == 1:
        return lq.linear_eq_of_state_drhodS()
    elif pyom.eq_of_state_type == 2:
        return nq1.nonlin1_eq_of_state_drhodS()
    elif pyom.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_drhodS()
    elif pyom.eq_of_state_type == 4:
        return nq3.nonlin3_eq_of_state_drhodS()
    elif pyom.eq_of_state_type == 5:
        return gsw.gsw_drhodS(salt_loc,temp_loc,press_loc)
    else:
        halt_stop(' unknown equation of state in get_drhodS')


def get_drhodp(salt_loc,temp_loc,press_loc,pyom):
    """
    calculate drho/dP as a function of temperature, salinity and pressure
    """
    if pyom.eq_of_state_type == 1:
        return lq.linear_eq_of_state_drhodp()
    elif pyom.eq_of_state_type == 2:
        return nq1.nonlin1_eq_of_state_drhodp()
    elif pyom.eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_drhodp(temp_loc)
    elif pyom.eq_of_state_type == 4:
        return nq3.nonlin3_eq_of_state_drhodp()
    elif pyom.eq_of_state_type == 5:
        return gsw.gsw_drhodp(salt_loc,temp_loc,press_loc)
    else:
        halt_stop(' unknown equation of state in get_drhodp')


def get_int_drhodT(salt_loc,temp_loc,press_loc,pyom):
    """
    calculate int_z^0 drho/dT dz' as a function of temperature, salinity and pressure
    """
    if eq_of_state_type == 1:
        return press_loc*lq.linear_eq_of_state_drhodT() # int_z^0rho_T dz = - rho_T z
    elif eq_of_state_type == 2:
        return press_loc*nq1.nonlin1_eq_of_state_drhodT(temp_loc)
    elif eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_int_drhodT(temp_loc,press_loc)
    elif eq_of_state_type == 4:
        return press_loc*nq3.nonlin3_eq_of_state_drhodT(temp_loc)
    elif eq_of_state_type == 5:
        return -(1024.0/9.81)*gsw.gsw_dHdT(salt_loc,temp_loc,press_loc)
    else:
        halt_stop(' unknown equation of state in get_int_drhodT')


def get_int_drhodS(salt_loc,temp_loc,press_loc,pyom):
    """
    calculate int_z^0 drho/dS dz' as a function of temperature, salinity and pressure
    """
    if eq_of_state_type == 1:
        return press_loc*lq.linear_eq_of_state_drhodS() # int_z^0rho_T dz = - rho_T z
    elif eq_of_state_type == 2:
        return press_loc*nq1.nonlin1_eq_of_state_drhodS()
    elif eq_of_state_type == 3:
        return nq2.nonlin2_eq_of_state_int_drhodS(press_loc)
    elif eq_of_state_type == 4:
        return press_loc*nq3.nonlin3_eq_of_state_drhodS()
    elif eq_of_state_type == 5:
        return -(1024.0/9.81)*gsw.gsw_dHdS(salt_loc,temp_loc,press_loc)
    else:
        halt_stop(' unknown equation of state in get_int_drhodS')
