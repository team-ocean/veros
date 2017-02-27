import numpy as np
import scipy.sparse
import scipy.sparse.linalg as spalg

import climate
from climate.pyom import cyclic


def congrad_streamfunction(forc,sol,pyom):
    # congrad_streamfunction.first is basically like a static variable
    if congrad_streamfunction.first:
        congrad_streamfunction.matrix = _assemble_matrix(pyom)
        #congrad_streamfunction.solve = spalg.factorized(_assemble_matrix(pyom))
        congrad_streamfunction.first = False

    #residuals = []
    #ml = pyamg.smoothed_aggregation_solver(congrad_streamfunction.poisson_matrix,numpy.ones((congrad_streamfunction.poisson_matrix.shape[0],1)),max_coarse=10)
    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(sol)
        make_cyclic = cyclic.setcyclic_x
    else:
        make_cyclic = lambda x: x

    #preconditioner = _make_inv_sfc(pyom)
    solution, info = spalg.bicgstab(congrad_streamfunction.matrix, forc[2:-2, 2:-2].flatten(),
                                    x0=sol[2:-2, 2:-2].flatten(), callback=make_cyclic, tol=pyom.congr_epsilon,
                                    maxiter=pyom.congr_max_iterations)#, M=preconditioner)
    print(info)
    sol[2:-2, 2:-2] = solution.reshape(pyom.nx,pyom.ny)
    #congrad_streamfunction.solve(forc[2:-2, 2:-2].reshape(-1, order="F")).reshape((pyom.nx, pyom.ny), order="F")
    #ml.solve(b=b,x0=x0,tol=1e-10,residuals=residuals,accel='cg')
congrad_streamfunction.first = True

def _make_inv_sfc(pyom):
    """
    construct an approximate inverse Z to A
    """
    # copy diagonal coefficients of A to Z
    Z = congrad_streamfunction.matrix.diagonal().copy().reshape(pyom.nx+4, pyom.ny+4)

    # now invert Z
    if climate.is_bohrium:
        Z[...] = (1. / (Z+(Z==0)))*(Z!=0)
    else:
        Z[Z != 0] = 1./Z[Z != 0]
    # make inverse zero on island perimeters that are not integrated
    for isle in xrange(pyom.nisle): #isle=1,nisle
        Z *= np.invert(pyom.boundary_mask[...,isle]).astype(np.float)
    return scipy.sparse.dia_matrix((Z.flatten(),0), shape=(Z.size, Z.size))

def _assemble_matrix(pyom):
    # assemble diagonals
    main_diag = -pyom.hvr[3:-1, 2:-2] / pyom.dxu[2:-2, np.newaxis] / pyom.dxt[3:-1, np.newaxis] / pyom.cosu[np.newaxis, 2:-2]**2 \
           - pyom.hvr[2:-2, 2:-2] / pyom.dxu[2:-2, np.newaxis] / pyom.dxt[2:-2, np.newaxis] / pyom.cosu[np.newaxis, 2:-2]**2 \
           - pyom.hur[2:-2, 2:-2] / pyom.dyu[np.newaxis, 2:-2] / pyom.dyt[np.newaxis, 2:-2] * pyom.cost[np.newaxis, 2:-2] / pyom.cosu[np.newaxis, 2:-2] \
           - pyom.hur[2:-2, 3:-1] / pyom.dyu[np.newaxis, 2:-2] / pyom.dyt[np.newaxis, 3:-1] * pyom.cost[np.newaxis, 3:-1] / pyom.cosu[np.newaxis, 2:-2]
    east_diag = pyom.hvr[3:-1, 2:-2] / pyom.dxu[2:-2, np.newaxis] / pyom.dxt[3:-1, np.newaxis] / pyom.cosu[np.newaxis, 2:-2]**2
    west_diag = pyom.hvr[2:-2, 2:-2] / pyom.dxu[2:-2, np.newaxis] / pyom.dxt[2:-2, np.newaxis] / pyom.cosu[np.newaxis, 2:-2]**2
    north_diag = pyom.hur[2:-2, 3:-1] / pyom.dyu[np.newaxis, 2:-2] / pyom.dyt[np.newaxis, 3:-1] * pyom.cost[np.newaxis, 3:-1] / pyom.cosu[np.newaxis, 2:-2]
    south_diag = pyom.hur[2:-2, 2:-2] / pyom.dyu[np.newaxis, 2:-2] / pyom.dyt[np.newaxis, 2:-2] * pyom.cost[np.newaxis, 2:-2] / pyom.cosu[np.newaxis, 2:-2]

    # fix boundaries
    east_diag[-1, :] = 0.
    west_diag[0, :] = 0.
    north_diag[:, -1] = 0.
    south_diag[:, 0] = 0.

    # assemble dia_matrix input
    offsets = (0, -main_diag.shape[0], main_diag.shape[0], -1, 1)
    cf = np.zeros((5, main_diag.size))
    cf[0] = main_diag.flatten()
    cf[1] = east_diag.flatten()
    cf[2] = west_diag.flatten()
    cf[3] = north_diag.flatten()
    cf[4] = south_diag.flatten()
    #cf[0] = main_diag.flatten(order="F")
    #cf[1][1:] = east_diag.flatten(order="F")[:-1]
    #cf[2][:-1] = west_diag.flatten(order="F")[1:]
    #cf[3][pyom.ny:] = north_diag.flatten(order="F")[:-pyom.ny]
    #cf[4][:-pyom.ny] = south_diag.flatten(order="F")[pyom.ny:]

    return scipy.sparse.dia_matrix((cf, offsets), shape=(main_diag.size, main_diag.size))


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
