import numpy as onp
import scipy.sparse
import scipy.sparse.linalg as spalg

from veros import logger, veros_kernel, veros_routine, distributed, runtime_state as rst
from veros.variables import allocate
from veros.core.operators import update, at, numpy as npx
from veros.core.external.solvers.base import LinearSolver
from veros.core.external.poisson_matrix import assemble_poisson_matrix


class SciPySolver(LinearSolver):
    @veros_routine(
        local_variables=(
            "hu",
            "hv",
            "hvr",
            "hur",
            "dxu",
            "dxt",
            "dyu",
            "dyt",
            "cosu",
            "cost",
            "isle_boundary_mask",
            "maskT",
        ),
        dist_safe=False,
    )
    def __init__(self, state):
        self._matrix, self._boundary_mask = self._assemble_poisson_matrix(state)

        jacobi_precon = self._jacobi_preconditioner(state, self._matrix)
        self._matrix = jacobi_precon * self._matrix
        self._rhs_scale = jacobi_precon.diagonal()
        self._extra_args = {}

        logger.info("Computing ILU preconditioner...")
        ilu_preconditioner = spalg.spilu(self._matrix.tocsc(), drop_tol=1e-6, fill_factor=100)
        self._extra_args["M"] = spalg.LinearOperator(self._matrix.shape, ilu_preconditioner.solve)

    def _scipy_solver(self, state, rhs, x0, boundary_val):
        orig_shape = x0.shape
        orig_dtype = x0.dtype

        rhs = npx.where(self._boundary_mask, rhs, boundary_val)  # set right hand side on boundaries

        rhs = onp.asarray(rhs.reshape(-1) * self._rhs_scale, dtype="float64")
        x0 = onp.asarray(x0.reshape(-1), dtype="float64")

        linear_solution, info = spalg.bicgstab(
            self._matrix,
            rhs,
            x0=x0,
            atol=1e-8,
            rtol=0,
            maxiter=1000,
            **self._extra_args,
        )

        if info > 0:
            logger.warning("Streamfunction solver did not converge after {} iterations", info)

        return npx.asarray(linear_solution, dtype=orig_dtype).reshape(orig_shape)

    def solve(self, state, rhs, x0, boundary_val=None):
        """
        Main solver for streamfunction. Solves a 2D Poisson equation. Uses scipy.sparse.linalg
        linear solvers.

        Arguments:
            rhs: Right-hand side vector
            x0: Initial guess
            boundary_val: Array containing values to set on boundary elements. Defaults to `x0`.

        """
        rhs_global, x0_global, boundary_val = gather_variables(state, rhs, x0, boundary_val)

        if rst.proc_rank == 0:
            linear_solution = self._scipy_solver(state, rhs_global, x0_global, boundary_val=boundary_val)
        else:
            linear_solution = npx.empty_like(rhs)

        return scatter_variables(state, linear_solution)

    @staticmethod
    def _jacobi_preconditioner(state, matrix):
        """
        Construct a simple Jacobi preconditioner
        """
        settings = state.settings

        eps = 1e-20
        precon = allocate(state.dimensions, ("xu", "yu"), fill=1, local=False)
        diag = npx.reshape(matrix.diagonal().copy(), (settings.nx + 4, settings.ny + 4))[2:-2, 2:-2]
        precon = update(precon, at[2:-2, 2:-2], npx.where(npx.abs(diag) > eps, 1.0 / (diag + eps), 1.0))
        precon = onp.asarray(precon)
        return scipy.sparse.dia_matrix((precon.reshape(-1), 0), shape=(precon.size, precon.size)).tocsr()

    @staticmethod
    def _assemble_poisson_matrix(state):
        settings = state.settings

        diags, offsets, boundary_mask = assemble_poisson_matrix(state)

        # flatten offsets (as expected by scipy.sparse)
        offsets = tuple(-dx * diags[0].shape[1] - dy for dx, dy in offsets)

        if settings.enable_cyclic_x:
            # add cyclic boundary conditions as additional matrix diagonals
            # (only works in single-process mode)
            wrap_diag_east, wrap_diag_west = (allocate(state.dimensions, ("xu", "yu"), local=False) for _ in range(2))
            wrap_diag_east = update(wrap_diag_east, at[2, 2:-2], diags[2][2, 2:-2] * boundary_mask[2, 2:-2])
            wrap_diag_west = update(wrap_diag_west, at[-3, 2:-2], diags[1][-3, 2:-2] * boundary_mask[-3, 2:-2])
            diags[2] = update(diags[2], at[2, 2:-2], 0.0)
            diags[1] = update(diags[1], at[-3, 2:-2], 0.0)

            offsets += (-diags[0].shape[1] * (settings.nx - 1), diags[0].shape[1] * (settings.nx - 1))
            diags += (wrap_diag_east, wrap_diag_west)

        diags = tuple(onp.asarray(diag.reshape(-1)) for diag in (diags))

        matrix = scipy.sparse.dia_matrix(
            (diags, offsets),
            shape=(diags[0].size, diags[0].size),
            dtype="float64",
        ).T.tocsr()

        return matrix, boundary_mask


@veros_kernel
def gather_variables(state, rhs, x0, boundary_val):
    rhs_global = distributed.gather(rhs, state.dimensions, ("xt", "yt"))
    x0_global = distributed.gather(x0, state.dimensions, ("xt", "yt"))

    if boundary_val is None:
        boundary_val = x0_global
    else:
        boundary_val = distributed.gather(boundary_val, state.dimensions, ("xt", "yt"))

    return rhs_global, x0_global, boundary_val


@veros_kernel
def scatter_variables(state, linear_solution):
    return distributed.scatter(linear_solution, state.dimensions, ("xt", "yt"))
