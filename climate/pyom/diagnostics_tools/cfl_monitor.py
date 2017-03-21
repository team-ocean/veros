from .. import pyom_method

def initialize(pyom):
    pass

@pyom_method
def diagnose(pyom):
    """
    check for CFL violation
    """
    cfl = max(
        np.max(np.abs(pyom.u[2:-2,2:-2,:,pyom.tau]) * pyom.maskU[2:-2,2:-2,:] \
                / (pyom.cost[np.newaxis, 2:-2, np.newaxis] * pyom.dxt[2:-2, np.newaxis, np.newaxis]) \
                * pyom.dt_tracer),
        np.max(np.abs(pyom.v[2:-2,2:-2,:,pyom.tau]) * pyom.maskV[2:-2,2:-2,:] \
                / pyom.dyt[np.newaxis, 2:-2, np.newaxis] * pyom.dt_tracer)
    )
    wcfl = np.max(np.abs(pyom.w[2:-2, 2:-2, :, pyom.tau]) * pyom.maskW[2:-2, 2:-2, :] \
                  / pyom.dzt[np.newaxis, np.newaxis, :] * pyom.dt_tracer)

    if np.isnan(cfl) or np.isnan(wcfl):
        raise RuntimeError("CFL number is NaN at iteration {}".format(pyom.itt))

    logging.warning("maximal hor. CFL number = {}".format(cfl))
    logging.warning("maximal ver. CFL number = {}".format(wcfl))

    if pyom.enable_eke or pyom.enable_tke or pyom.enable_idemix:
        cfl = max(
            np.max(np.abs(pyom.u_wgrid[2:-2,2:-2,:]) * pyom.maskU[2:-2,2:-2,:] \
                    / (pyom.cost[np.newaxis, 2:-2, np.newaxis] * pyom.dxt[2:-2, np.newaxis, np.newaxis]) \
                    * pyom.dt_tracer),
            np.max(np.abs(pyom.v_wgrid[2:-2,2:-2,:]) * pyom.maskV[2:-2,2:-2,:] \
                    / pyom.dyt[np.newaxis, 2:-2, np.newaxis] * pyom.dt_tracer)
        )
        wcfl = np.max(np.abs(pyom.w_wgrid[2:-2, 2:-2, :]) * pyom.maskW[2:-2, 2:-2, :] \
                      / pyom.dzt[np.newaxis, np.newaxis, :] * pyom.dt_tracer)
        logging.warning("maximal hor. CFL number on w grid = {}".format(cfl))
        logging.warning("maximal ver. CFL number on w grid = {}".format(wcfl))

def output(pyom):
    pass
