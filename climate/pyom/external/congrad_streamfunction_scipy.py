import numpy as np
import scipy.sparse
import scipy.sparse.linalg as spalg

import climate
from climate.pyom import cyclic


def congrad_streamfunction(forc,sol,pyom):
    # congrad_streamfunction.first is basically like a static variable
    if congrad_streamfunction.first:
        congrad_streamfunction.matrix = _assemble_matrix(pyom)
        congrad_streamfunction.first = False

    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(sol)
        def make_cyclic(x):
            pass#cyclic.setcyclic_x(x.reshape(pyom.nx+4, pyom.ny+4))
    else:
        make_cyclic = lambda x: None

    preconditioner = _make_inv_sfc(pyom)
    solution, info = spalg.bicg(congrad_streamfunction.matrix, forc.flatten(),
                                    x0=sol.flatten(), callback=make_cyclic, tol=pyom.congr_epsilon,
                                    maxiter=pyom.congr_max_iterations, M=preconditioner)
    sol[...] = solution.reshape(pyom.nx+4,pyom.ny+4)
congrad_streamfunction.first = True


def _make_inv_sfc(pyom):
    """
    construct an approximate inverse Z to A
    """
    # copy diagonal coefficients of A to Z
    Z = np.zeros((pyom.nx+4, pyom.ny+4))
    Z[2:-2, 2:-2] = congrad_streamfunction.matrix.diagonal().copy().reshape(pyom.nx+4, pyom.ny+4)[2:-2, 2:-2]

    # now invert Z
    Y = Z[2:-2, 2:-2]
    if climate.is_bohrium:
        Y[...] = (1. / (Y+(Y==0)))*(Y!=0)
    else:
        Y[Y != 0] = 1./Y[Y != 0]

    # make inverse zero on island perimeters that are not integrated
    if pyom.nisle:
        Z *= np.prod(np.invert(pyom.boundary_mask), axis=2)
    return scipy.sparse.dia_matrix((Z.flatten(),0), shape=(Z.size, Z.size))


def _assemble_matrix(pyom):
    # assemble diagonals
    main_diag = np.ones((pyom.nx+4, pyom.ny+4))
    east_diag, west_diag, north_diag, south_diag = (np.zeros((pyom.nx+4, pyom.ny+4)) for _ in range(4))
    main_diag[2:-2, 2:-2] = -pyom.hvr[3:-1, 2:-2] / pyom.dxu[2:-2, np.newaxis] / pyom.dxt[3:-1, np.newaxis] / pyom.cosu[np.newaxis, 2:-2]**2 \
                           - pyom.hvr[2:-2, 2:-2] / pyom.dxu[2:-2, np.newaxis] / pyom.dxt[2:-2, np.newaxis] / pyom.cosu[np.newaxis, 2:-2]**2 \
                           - pyom.hur[2:-2, 2:-2] / pyom.dyu[np.newaxis, 2:-2] / pyom.dyt[np.newaxis, 2:-2] * pyom.cost[np.newaxis, 2:-2] / pyom.cosu[np.newaxis, 2:-2] \
                           - pyom.hur[2:-2, 3:-1] / pyom.dyu[np.newaxis, 2:-2] / pyom.dyt[np.newaxis, 3:-1] * pyom.cost[np.newaxis, 3:-1] / pyom.cosu[np.newaxis, 2:-2]
    east_diag[2:-2, 2:-2] = pyom.hvr[3:-1, 2:-2] / pyom.dxu[2:-2, np.newaxis] / pyom.dxt[3:-1, np.newaxis] / pyom.cosu[np.newaxis, 2:-2]**2
    west_diag[2:-2, 2:-2] = pyom.hvr[2:-2, 2:-2] / pyom.dxu[2:-2, np.newaxis] / pyom.dxt[2:-2, np.newaxis] / pyom.cosu[np.newaxis, 2:-2]**2
    north_diag[2:-2, 2:-2] = pyom.hur[2:-2, 3:-1] / pyom.dyu[np.newaxis, 2:-2] / pyom.dyt[np.newaxis, 3:-1] * pyom.cost[np.newaxis, 3:-1] / pyom.cosu[np.newaxis, 2:-2]
    south_diag[2:-2, 2:-2] = pyom.hur[2:-2, 2:-2] / pyom.dyu[np.newaxis, 2:-2] / pyom.dyt[np.newaxis, 2:-2] * pyom.cost[np.newaxis, 2:-2] / pyom.cosu[np.newaxis, 2:-2]

    # fix boundaries
    #east_diag[:2, :] = 0.
    #west_diag[-2:, :] = 0.
    #north_diag[:, -1] = 0.
    #south_diag[:, 0] = 0.

    # assemble dia_matrix input
    offsets = (0, -main_diag.shape[1], main_diag.shape[1], -1, 1)
    cf = np.zeros((5, main_diag.size))
    cf[0] = main_diag.flatten()
    cf[1] = east_diag.flatten()
    cf[2] = west_diag.flatten()
    cf[3] = north_diag.flatten()
    cf[4] = south_diag.flatten()
    return scipy.sparse.dia_matrix((cf, offsets), shape=(main_diag.size, main_diag.size)).T


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
