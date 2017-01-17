import numpy as np

def init_eke(pyom):
    """
    =======================================================================
    Initialize EKE
    =======================================================================
    """
    if pyom.enable_eke_leewave_dissipation:
        pyom.hrms_k0 = np.maximum(pyom.eke_hrms_k0_min, 2/np.pi*pyom.eke_topo_hrms**2/np.maximum(1e-12,pyom.eke_topo_lam)**1.5  )
