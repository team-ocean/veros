import numpy as np
import scipy.sparse
import scipy.sparse.linalg as spalg

import climate
from climate.pyom import cyclic


def solve(rhs, sol, pyom):
    """
    Main solver for streamfunction. Solves a 2D Poisson equation.
    """
    # assemble static matrices on first call only
    if solve.first:
        solve.matrix = _assemble_poisson_matrix(pyom)
        solve.preconditioner = _assemble_preconditioner(solve.matrix, pyom)
        sparse_preconditioner = scipy.sparse.dia_matrix((solve.preconditioner.flatten(),0), shape=(sol.size, sol.size))
        solve.matrix = sparse_preconditioner * solve.matrix
        solve.first = False

    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(sol)

    rhs *= solve.preconditioner
    solution, info = spalg.bicgstab(solve.matrix, rhs.flatten(),
                                    x0=sol.flatten(), tol=pyom.congr_epsilon,
                                    maxiter=pyom.congr_max_iterations)
    if info > 0:
        print("WARNING: streamfunction solver did not converge after {} iterations".format(info))
    sol[...] = solution.reshape(pyom.nx+4,pyom.ny+4)
solve.first = True


def _assemble_preconditioner(matrix, pyom):
    """
    Construct a simple Jacobi preconditioner

    .. note::
        This preconditioner is generally singular. Thus, it will modify the solution of the
        system to satisfy the island boundary conditions.
    """
    # copy diagonal coefficients of A to Z
    Z = np.zeros((pyom.nx+4, pyom.ny+4))
    Z[2:-2, 2:-2] = matrix.diagonal().copy().reshape(pyom.nx+4, pyom.ny+4)[2:-2, 2:-2]

    # now invert Z
    Y = Z[2:-2, 2:-2]
    if climate.is_bohrium:
        Y[...] = (1. / (Y+(Y==0)))*(Y!=0)
    else:
        Y[Y != 0] = 1./Y[Y != 0]

    # make inverse zero on island perimeters that are not integrated
    if pyom.nisle:
        Z[...] *= np.prod(np.invert(pyom.boundary_mask), axis=2)
    return Z


def _assemble_poisson_matrix(pyom):
    """
    Construct a sparse matrix based on the stencil for the 2D Poisson equation.
    """
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

    if pyom.enable_cyclic_x:
        # couple edges of the domain
        wrap_diag_east, wrap_diag_west = (np.zeros((pyom.nx+4, pyom.ny+4)) for _ in range(2))
        wrap_diag_east[2, 2:-2] = west_diag[2, 2:-2]
        wrap_diag_west[-3, 2:-2] = east_diag[-3, 2:-2]

    # construct sparse matrix
    cf = tuple(diag.flatten() for diag in (main_diag, east_diag, west_diag, north_diag, south_diag))
    offsets = (0, -main_diag.shape[1], main_diag.shape[1], -1, 1)

    if pyom.enable_cyclic_x:
        offsets += (-main_diag.shape[1] * (pyom.nx-1), main_diag.shape[1] * (pyom.nx-1))
        cf += (wrap_diag_east.flatten(), wrap_diag_west.flatten())

    return scipy.sparse.dia_matrix((np.array(cf), offsets), shape=(main_diag.size, main_diag.size)).T
