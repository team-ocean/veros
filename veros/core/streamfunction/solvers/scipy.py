from loguru import logger
from veros.core.operators import numpy as np
import numpy as onp
import scipy.sparse
import scipy.sparse.linalg as spalg

from veros import distributed
from veros.variables import allocate
from veros.core import utilities
from veros.core.operators import update, at
from veros.core.streamfunction.solvers.base import LinearSolver


class SciPySolver(LinearSolver):
    # @veros_method(dist_safe=False, local_variables=[
    #     'hvr', 'hur',
    #     'dxu', 'dxt', 'dyu', 'dyt',
    #     'cosu', 'cost',
    #     'boundary_mask'
    # ])
    def __init__(self, vs):
        self._extra_args = {}
        self._matrix = self._assemble_poisson_matrix(vs)
        self._preconditioner = self._jacobi_preconditioner(vs, self._matrix)
        self._matrix = self._preconditioner * self._matrix

    # @veros_method(dist_safe=False, local_variables=['boundary_mask'])
    def _scipy_solver(self, vs, rhs, sol, boundary_val):
        sol = utilities.enforce_boundaries(sol, vs.enable_cyclic_x)

        boundary_mask = np.prod(1 - vs.boundary_mask, axis=2)
        rhs = utilities.where(boundary_mask, rhs, boundary_val) # set right hand side on boundaries

        rhs = onp.asarray(rhs.flatten()) * self._preconditioner.diagonal()
        x0 = onp.asarray(sol.flatten())

        linear_solution, info = spalg.bicgstab(
            self._matrix, rhs,
            x0=x0, atol=0, tol=vs.congr_epsilon,
            maxiter=vs.congr_max_iterations,
            **self._extra_args
        )

        if info > 0:
            logger.warning('Streamfunction solver did not converge after {} iterations', info)

        return np.asarray(linear_solution.reshape(vs.nx + 4, vs.ny + 4))

    # @veros_method
    def solve(self, vs, rhs, sol, boundary_val=None):
        """
        Main solver for streamfunction. Solves a 2D Poisson equation. Uses scipy.sparse.linalg
        linear solvers.

        Arguments:
            rhs: Right-hand side vector
            sol: Initial guess, gets overwritten with solution
            boundary_val: Array containing values to set on boundary elements. Defaults to `sol`.
        """
        rhs_global = distributed.gather(rhs, ('xt', 'yt'))
        sol_global = distributed.gather(sol, ('xt', 'yt'))

        if boundary_val is None:
            boundary_val = sol_global
        else:
            boundary_val = distributed.gather(boundary_val, ('xt', 'yt'))

        linear_solution = self._scipy_solver(vs, rhs_global, sol_global, boundary_val=boundary_val)
        return distributed.scatter(linear_solution, ('xt', 'yt'))

    @staticmethod
    # @veros_method(dist_safe=False, local_variables=[])
    def _jacobi_preconditioner(vs, matrix):
        """
        Construct a simple Jacobi preconditioner
        """
        eps = 1e-20
        Z = allocate(vs, ('xu', 'yu'), fill=1, local=False)
        Y = np.reshape(matrix.diagonal().copy(), (vs.nx + 4, vs.ny + 4))[2:-2, 2:-2]
        Z = update(Z, at[2:-2, 2:-2], utilities.where(np.abs(Y) > eps, 1. / (Y + eps), 1.))
        return scipy.sparse.dia_matrix((Z.flatten(), 0), shape=(Z.size, Z.size)).tocsr()

    @staticmethod
    # @veros_method(dist_safe=False, local_variables=['boundary_mask'])
    def _assemble_poisson_matrix(vs):
        """
        Construct a sparse matrix based on the stencil for the 2D Poisson equation.
        """
        boundary_mask = np.prod(1 - vs.boundary_mask, axis=2)

        # assemble diagonals
        main_diag = allocate(vs, ('xu', 'yu'), fill=1, local=False)
        east_diag, west_diag, north_diag, south_diag = (allocate(vs, ('xu', 'yu'), local=False) for _ in range(4))
        main_diag = update(main_diag, at[2:-2, 2:-2], -vs.hvr[3:-1, 2:-2] / vs.dxu[2:-2, np.newaxis] / vs.dxt[3:-1, np.newaxis] / vs.cosu[np.newaxis, 2:-2]**2 \
            - vs.hvr[2:-2, 2:-2] / vs.dxu[2:-2, np.newaxis] / vs.dxt[2:-2, np.newaxis] / vs.cosu[np.newaxis, 2:-2]**2 \
            - vs.hur[2:-2, 2:-2] / vs.dyu[np.newaxis, 2:-2] / vs.dyt[np.newaxis, 2:-2] * vs.cost[np.newaxis, 2:-2] / vs.cosu[np.newaxis, 2:-2] \
            - vs.hur[2:-2, 3:-1] / vs.dyu[np.newaxis, 2:-2] / vs.dyt[np.newaxis, 3:-1] * vs.cost[np.newaxis, 3:-1] / vs.cosu[np.newaxis, 2:-2])
        east_diag = update(east_diag, at[2:-2, 2:-2], vs.hvr[3:-1, 2:-2] / vs.dxu[2:-2, np.newaxis] / \
            vs.dxt[3:-1, np.newaxis] / vs.cosu[np.newaxis, 2:-2]**2)
        west_diag = update(west_diag, at[2:-2, 2:-2], vs.hvr[2:-2, 2:-2] / vs.dxu[2:-2, np.newaxis] / \
            vs.dxt[2:-2, np.newaxis] / vs.cosu[np.newaxis, 2:-2]**2)
        north_diag = update(north_diag, at[2:-2, 2:-2], vs.hur[2:-2, 3:-1] / vs.dyu[np.newaxis, 2:-2] / \
            vs.dyt[np.newaxis, 3:-1] * vs.cost[np.newaxis, 3:-1] / vs.cosu[np.newaxis, 2:-2])
        south_diag = update(south_diag, at[2:-2, 2:-2], vs.hur[2:-2, 2:-2] / vs.dyu[np.newaxis, 2:-2] / \
            vs.dyt[np.newaxis, 2:-2] * vs.cost[np.newaxis, 2:-2] / vs.cosu[np.newaxis, 2:-2])

        if vs.enable_cyclic_x:
            # couple edges of the domain
            wrap_diag_east, wrap_diag_west = (allocate(vs, ('xu', 'yu'), local=False) for _ in range(2))
            wrap_diag_east = update(wrap_diag_east, at[2, 2:-2], west_diag[2, 2:-2] * boundary_mask[2, 2:-2])
            wrap_diag_west = update(wrap_diag_west, at[-3, 2:-2], east_diag[-3, 2:-2] * boundary_mask[-3, 2:-2])
            west_diag = update(west_diag, at[2, 2:-2], 0.)
            east_diag = update(east_diag, at[-3, 2:-2], 0.)

        # construct sparse matrix
        cf = tuple(diag.flatten() for diag in (
            boundary_mask * main_diag + (1 - boundary_mask),
            boundary_mask * east_diag,
            boundary_mask * west_diag,
            boundary_mask * north_diag,
            boundary_mask * south_diag
        ))
        offsets = (0, -main_diag.shape[1], main_diag.shape[1], -1, 1)

        if vs.enable_cyclic_x:
            offsets += (-main_diag.shape[1] * (vs.nx - 1), main_diag.shape[1] * (vs.nx - 1))
            cf += (wrap_diag_east.flatten(), wrap_diag_west.flatten())

        cf = onp.asarray(cf)
        return scipy.sparse.dia_matrix((cf, offsets), shape=(main_diag.size, main_diag.size)).T.tocsr()
