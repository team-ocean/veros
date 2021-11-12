import numpy as onp
import scipy.sparse
import scipy.sparse.linalg as spalg

from veros import logger, veros_kernel, veros_routine, distributed, runtime_settings, runtime_state as rst
from veros.variables import allocate
from veros.core.operators import update, update_add, at, numpy as npx
from veros.core.external.solvers.base import LinearSolver


class SciPyPressureSolver(LinearSolver):
    @veros_routine(
        local_variables=("hv", "hu", "dxu", "dxt", "dyu", "dyt", "cosu", "cost", "maskT"),
        dist_safe=False,
    )
    def __init__(self, state):
        self._matrix = self._assemble_poisson_matrix(state)
        jacobi_precon = self._jacobi_preconditioner(state, self._matrix)
        self._matrix = jacobi_precon @ self._matrix
        self._rhs_scale = jacobi_precon.diagonal()
        self._extra_args = {}

        logger.info("Computing ILU preconditioner...")
        ilu_preconditioner = spalg.spilu(self._matrix.tocsc(), drop_tol=1e-6, fill_factor=100)
        self._extra_args["M"] = spalg.LinearOperator(self._matrix.shape, ilu_preconditioner.solve)

    def _scipy_solver(self, state, rhs, x0, boundary_mask, boundary_val):
        orig_shape = x0.shape
        orig_dtype = x0.dtype

        rhs = npx.where(boundary_mask, rhs, boundary_val)
        x0 = onp.asarray(x0.reshape(-1), dtype="float64")
        rhs = onp.asarray(rhs.reshape(-1) * self._rhs_scale, dtype="float64")

        linear_solution, info = spalg.bicgstab(
            self._matrix,
            rhs,
            x0=x0,
            atol=1e-8,
            tol=0,
            maxiter=10000,
            **self._extra_args,
        )

        if info > 0:
            logger.warning("Pressure solver did not converge after {} iterations", info)

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
        rhs_global, x0_global, boundary_mask, boundary_val = gather_variables(state, rhs, x0, boundary_val)

        if rst.proc_rank == 0:
            linear_solution = self._scipy_solver(
                state, rhs_global, x0_global, boundary_mask=boundary_mask, boundary_val=boundary_val
            )
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
        precon = onp.array(precon)
        return scipy.sparse.dia_matrix((precon.reshape(-1), 0), shape=(precon.size, precon.size)).tocsr()

    @staticmethod
    def _assemble_poisson_matrix(state):
        main_diag = allocate(state.dimensions, ("xu", "yu"), local=False)
        east_diag, west_diag, north_diag, south_diag = (
            allocate(state.dimensions, ("xu", "yu"), local=False) for _ in range(4)
        )

        vs = state.variables
        settings = state.settings

        maskM = allocate(state.dimensions, ("xu", "yu"), local=False)
        mp_i, mm_i, mp_j, mm_j = (
            allocate(state.dimensions, ("xu", "yu"), local=False, include_ghosts=False) for _ in range(4)
        )

        maskM = update(maskM, at[:, :], vs.maskT[:, :, -1])

        mp_i = update(mp_i, at[:, :], maskM[2:-2, 2:-2] * maskM[3:-1, 2:-2])
        mm_i = update(mm_i, at[:, :], maskM[2:-2, 2:-2] * maskM[1:-3, 2:-2])

        mp_j = update(mp_j, at[:, :], maskM[2:-2, 2:-2] * maskM[2:-2, 3:-1])
        mm_j = update(mm_j, at[:, :], maskM[2:-2, 2:-2] * maskM[2:-2, 1:-3])

        main_diag = update(
            main_diag,
            at[2:-2, 2:-2],
            -mp_i
            * vs.hu[2:-2, 2:-2]
            / vs.dxu[2:-2, npx.newaxis]
            / vs.dxt[2:-2, npx.newaxis]
            / vs.cost[npx.newaxis, 2:-2] ** 2
            - mm_i
            * vs.hu[1:-3, 2:-2]
            / vs.dxu[1:-3, npx.newaxis]
            / vs.dxt[2:-2, npx.newaxis]
            / vs.cost[npx.newaxis, 2:-2] ** 2
            - mp_j
            * vs.hv[2:-2, 2:-2]
            / vs.dyu[npx.newaxis, 2:-2]
            / vs.dyt[npx.newaxis, 2:-2]
            * vs.cosu[npx.newaxis, 2:-2]
            / vs.cost[npx.newaxis, 2:-2]
            - mm_j
            * vs.hv[2:-2, 1:-3]
            / vs.dyu[npx.newaxis, 1:-3]
            / vs.dyt[npx.newaxis, 2:-2]
            * vs.cosu[npx.newaxis, 1:-3]
            / vs.cost[npx.newaxis, 2:-2],
        )

        if settings.enable_free_surface:
            if runtime_settings.pyom_compatibility_mode:
                dt = settings.dt_mom
            else:
                dt = settings.dt_tracer

        main_diag = update_add(
            main_diag,
            at[2:-2, 2:-2],
            -1.0 / (settings.grav * settings.dt_mom * dt) * maskM[2:-2, 2:-2],
        )

        east_diag = update(
            east_diag,
            at[2:-2, 2:-2],
            mp_i
            * vs.hu[2:-2, 2:-2]
            / vs.dxu[2:-2, npx.newaxis]
            / vs.dxt[2:-2, npx.newaxis]
            / vs.cost[npx.newaxis, 2:-2] ** 2,
        )

        west_diag = update(
            west_diag,
            at[2:-2, 2:-2],
            mm_i
            * vs.hu[1:-3, 2:-2]
            / vs.dxu[1:-3, npx.newaxis]
            / vs.dxt[2:-2, npx.newaxis]
            / vs.cost[npx.newaxis, 2:-2] ** 2,
        )

        north_diag = update(
            north_diag,
            at[2:-2, 2:-2],
            mp_j
            * vs.hv[2:-2, 2:-2]
            / vs.dyu[npx.newaxis, 2:-2]
            / vs.dyt[npx.newaxis, 2:-2]
            * vs.cosu[npx.newaxis, 2:-2]
            / vs.cost[npx.newaxis, 2:-2],
        )

        south_diag = update(
            south_diag,
            at[2:-2, 2:-2],
            mm_j
            * vs.hv[2:-2, 1:-3]
            / vs.dyu[npx.newaxis, 1:-3]
            / vs.dyt[npx.newaxis, 2:-2]
            * vs.cosu[npx.newaxis, 1:-3]
            / vs.cost[npx.newaxis, 2:-2],
        )
        main_diag = main_diag * maskM
        main_diag = npx.where(main_diag == 0.0, 1, main_diag)

        offsets = (0, -main_diag.shape[1], main_diag.shape[1], -1, 1)

        if settings.enable_cyclic_x:
            wrap_diag_east, wrap_diag_west = (allocate(state.dimensions, ("xu", "yu"), local=False) for _ in range(2))
            wrap_diag_east = update(wrap_diag_east, at[2, 2:-2], west_diag[2, 2:-2] * maskM[2, 2:-2])
            wrap_diag_west = update(wrap_diag_west, at[-3, 2:-2], east_diag[-3, 2:-2] * maskM[-3, 2:-2])
            west_diag = update(west_diag, at[2, 2:-2], 0.0)
            east_diag = update(east_diag, at[-3, 2:-2], 0.0)

        cf = tuple(
            diag.reshape(-1)
            for diag in (
                main_diag,
                east_diag,
                west_diag,
                north_diag,
                south_diag,
            )
        )

        if settings.enable_cyclic_x:
            offsets += (-main_diag.shape[1] * (settings.nx - 1), main_diag.shape[1] * (settings.nx - 1))
            cf += (wrap_diag_east.reshape(-1), wrap_diag_west.reshape(-1))

        cf = onp.asarray(cf, dtype="float64")
        return scipy.sparse.dia_matrix((cf, offsets), shape=(main_diag.size, main_diag.size)).T.tocsr()


@veros_kernel
def gather_variables(state, rhs, x0, boundary_val):
    rhs_global = distributed.gather(rhs, state.dimensions, ("xt", "yt"))
    x0_global = distributed.gather(x0, state.dimensions, ("xt", "yt"))

    boundary_mask = distributed.gather(state.variables.maskT[..., -1], state.dimensions, ("xt", "yt"))

    if boundary_val is None:
        boundary_val = x0_global
    else:
        boundary_val = distributed.gather(boundary_val, state.dimensions, ("xt", "yt"))

    return rhs_global, x0_global, boundary_mask, boundary_val


@veros_kernel
def scatter_variables(state, linear_solution):
    return distributed.scatter(linear_solution, state.dimensions, ("xt", "yt"))
