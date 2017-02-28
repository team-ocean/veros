import numpy as np

import climate
from climate.pyom import cyclic
from climate.pyom.external import utilities


def congrad_streamfunction(forc,sol,pyom):
    """
    conjugate gradient solver with preconditioner from MOM
    """
    # congrad_streamfunction.first is basically like a static variable
    if congrad_streamfunction.first:
        congrad_streamfunction.cf = np.zeros((pyom.nx+4, pyom.ny+4, 3, 3))
        _make_coeff_streamfunction(congrad_streamfunction.cf,pyom)
        congrad_streamfunction.first = False

    Z    = np.zeros((pyom.nx+4, pyom.ny+4))
    Zres = np.zeros((pyom.nx+4, pyom.ny+4))
    ss   = np.zeros((pyom.nx+4, pyom.ny+4))
    As   = np.zeros((pyom.nx+4, pyom.ny+4))
    res  = np.zeros((pyom.nx+4, pyom.ny+4))
    """
    make approximate inverse operator Z (always even symmetry)
    """
    utilities.make_inv_sfc(congrad_streamfunction.cf, Z, pyom)
    """
    impose boundary conditions on guess
    sol(0) = guess
    """
    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(sol)
    """
    res(0)  = forc - A * eta(0)
    """
    utilities.apply_op(congrad_streamfunction.cf, sol, res, pyom)
    res[2:-2, 2:-2] = forc[2:-2, 2:-2] - res[2:-2, 2:-2]

    """
    Zres(k-1) = Z * res(k-1)
    see if guess is a solution, bail out to avoid division by zero
    """
    n = np.int(0)
    utilities.inv_op_sfc(Z, res, Zres, pyom)
    Zresmax = utilities.absmax_sfc(Zres, pyom)
    # Assume convergence rate of 0.99 to extrapolate error
    if 100.0 * Zresmax < pyom.congr_epsilon:
        estimated_error = 100.0 * Zresmax
        _print_info(n, estimated_error, pyom)
        return True #Converged
    """
    beta(0) = 1
    ss(0)    = zerovector()
    """
    betakm1 = np.float(1.0)
    ss[...] = 0.
    """
    begin iteration loop
    """
    n = 1
    cont = True
    while n < pyom.congr_max_iterations and cont:
        """
        Zres(k-1) = Z * res(k-1)
        """
        utilities.inv_op_sfc(Z, res, Zres, pyom)
        """
        beta(k)   = res(k-1) * Zres(k-1)
        """
        betak = utilities.dot_sfc(Zres, res, pyom)
        if n == 1:
            betak_min = np.abs(betak)
        elif n > 2:
            betak_min = np.minimum(betak_min, np.abs(betak))
            if np.abs(betak) > 100.0*betak_min:
                print("WARNING: solver diverging at itt={:d}".format(pyom.congr_itts))
                _fail(n, estimated_error, pyom)
                #cont = False
                #converged = False #Converged
        """
        ss(k)      = Zres(k-1) + (beta(k)/beta(k-1)) * ss(k-1)
        """
        betaquot = betak/betakm1
        ss[2:-2,2:-2] = Zres[2:-2,2:-2] + betaquot*ss[2:-2,2:-2]

        if pyom.enable_cyclic_x:
            cyclic.setcyclic_x(ss)
        """
        As(k)     = A * ss(k)
        """
        utilities.apply_op(congrad_streamfunction.cf, ss, As, pyom)
        """
        If ss=0 then the division for alpha(k) gives a float exception.
        Assume convergence rate of 0.99 to extrapolate error.
        Also assume alpha(k) ~ 1.
        """
        s_dot_As = utilities.dot_sfc(ss, As, pyom)
        if np.abs(s_dot_As) < np.abs(betak)*np.float(1.e-10):
            smax = utilities.absmax_sfc(ss,pyom)
            estimated_error = 100.0 * smax
            _print_info(n, estimated_error, pyom)
            cont = False
            #converged = True #Converged
        """
        alpha(k)  = beta(k) / (ss(k) * As(k))
        """
        alpha = betak / s_dot_As
        """
        update values:
        eta(k)   = eta(k-1) + alpha(k) * ss(k)
        res(k)    = res(k-1) - alpha(k) * As(k)
        """
        if cont:
            sol[2:-2, 2:-2] += alpha * ss[2:-2, 2:-2]
            res[2:-2, 2:-2] += -alpha * As[2:-2, 2:-2]

        smax = utilities.absmax_sfc(ss, pyom)
        """
        test for convergence
        if (estimated_error) < congr_epsilon) exit
        """
        step = np.abs(alpha) * smax
        if n == 1:
            step1 = step
            estimated_error = step
            if step < pyom.congr_epsilon:
                _print_info(n, estimated_error, pyom)
                cont = False
                #converged = True #Converged
        elif step < pyom.congr_epsilon:
            convergence_rate = np.exp(np.log(step/step1)/(n-1))
            estimated_error = step*convergence_rate/(1.0-convergence_rate)
            if estimated_error < pyom.congr_epsilon:
                _print_info(n, estimated_error, pyom)
                cont = False
                #converged = True #Converged
        betakm1 = betak
        if cont:
            n += 1
        if climate.is_bohrium:
            np.flush()
    """
    end of iteration loop
    """
    if cont:
        print(" WARNING: max iterations exceeded at itt="),n
        _fail(n, estimated_error, pyom)
        #return False #Converged
congrad_streamfunction.first = True


def _make_coeff_streamfunction(cf, pyom):
    """
    A * p = forc
    res = A * p
    res = res +  cf(...,ii,jj,kk) * p(i+ii,j+jj,k+kk)
    """
    cf[2:-2, 2:-2, 1, 1] -= pyom.hvr[3:-1, 2:-2] / pyom.dxu[2:-2, np.newaxis] / pyom.dxt[3:-1, np.newaxis] / pyom.cosu[np.newaxis, 2:-2]**2
    cf[2:-2, 2:-2, 2, 1] += pyom.hvr[3:-1, 2:-2] / pyom.dxu[2:-2, np.newaxis] / pyom.dxt[3:-1, np.newaxis] / pyom.cosu[np.newaxis, 2:-2]**2
    cf[2:-2, 2:-2, 1, 1] -= pyom.hvr[2:-2, 2:-2] / pyom.dxu[2:-2, np.newaxis] / pyom.dxt[2:-2, np.newaxis] / pyom.cosu[np.newaxis, 2:-2]**2
    cf[2:-2, 2:-2, 0, 1] += pyom.hvr[2:-2, 2:-2] / pyom.dxu[2:-2, np.newaxis] / pyom.dxt[2:-2, np.newaxis] / pyom.cosu[np.newaxis, 2:-2]**2

    cf[2:-2, 2:-2, 1, 1] -= pyom.hur[2:-2, 3:-1] / pyom.dyu[np.newaxis, 2:-2] / pyom.dyt[np.newaxis, 3:-1] * pyom.cost[np.newaxis, 3:-1] / pyom.cosu[np.newaxis, 2:-2]
    cf[2:-2, 2:-2, 1, 2] += pyom.hur[2:-2, 3:-1] / pyom.dyu[np.newaxis, 2:-2] / pyom.dyt[np.newaxis, 3:-1] * pyom.cost[np.newaxis, 3:-1] / pyom.cosu[np.newaxis, 2:-2]
    cf[2:-2, 2:-2, 1, 1] -= pyom.hur[2:-2, 2:-2] / pyom.dyu[np.newaxis, 2:-2] / pyom.dyt[np.newaxis, 2:-2] * pyom.cost[np.newaxis, 2:-2] / pyom.cosu[np.newaxis, 2:-2]
    cf[2:-2, 2:-2, 1, 0] += pyom.hur[2:-2, 2:-2] / pyom.dyu[np.newaxis, 2:-2] / pyom.dyt[np.newaxis, 2:-2] * pyom.cost[np.newaxis, 2:-2] / pyom.cosu[np.newaxis, 2:-2]


def _print_info(n, estimated_error, pyom):
    pyom.congr_itts = n
    #if pyom.enable_congrad_verbose:
    #    print(" estimated error="),estimated_error,"/",pyom.congr_epsilon
    #    print(" iterations="),n


def _fail(n, estimated_error, pyom):
    pyom.congr_itts = n
    #print(" estimated error="),estimated_error,"/",pyom.congr_epsilon
    #print(" iterations="),n
    # check for NaN
    if np.isnan(estimated_error):
        raise RuntimeError("error is NaN, stopping integration")
