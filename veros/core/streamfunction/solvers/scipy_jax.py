from veros import distributed, veros_routine, veros_kernel, runtime_state as rst
from veros.variables import allocate

from veros.core import utilities
from veros.core.operators import update, update_add, at, numpy as npx
from veros.core.streamfunction.solvers.base import LinearSolver


@veros_kernel(static_args=("solve_fun",))
def solve_kernel(state, rhs, x0, boundary_val, solve_fun):
    vs = state.variables

    rhs_global = distributed.gather(rhs, state.dimensions, ("xt", "yt"))
    x0_global = distributed.gather(x0, state.dimensions, ("xt", "yt"))

    boundary_mask = ~npx.any(vs.boundary_mask, axis=2)
    boundary_mask_global = distributed.gather(boundary_mask, state.dimensions, ("xt", "yt"))

    if boundary_val is None:
        boundary_val = x0_global
    else:
        boundary_val = distributed.gather(boundary_val, state.dimensions, ("xt", "yt"))

    if rst.proc_rank == 0:
        linear_solution = solve_fun(rhs_global, x0_global, boundary_mask_global, boundary_val)
    else:
        linear_solution = npx.empty_like(rhs)

    return distributed.scatter(linear_solution, state.dimensions, ("xt", "yt"))


class JAXSciPySolver(LinearSolver):
    @veros_routine(
        local_variables=("hvr", "hur", "dxu", "dxt", "dyu", "dyt", "cosu", "cost", "boundary_mask"),
        dist_safe=False,
    )
    def __init__(self, state):
        from jax.scipy.sparse.linalg import bicgstab

        settings = state.settings

        matrix_diags, offsets = self._assemble_poisson_matrix(state)
        jacobi_precon = self._jacobi_preconditioner(state, matrix_diags)
        matrix_diags = tuple(jacobi_precon * diag for diag in matrix_diags)

        @veros_kernel
        def linear_solve(rhs, x0, boundary_mask, boundary_val):
            x0 = utilities.enforce_boundaries(x0, settings.enable_cyclic_x, local=True)
            rhs = npx.where(boundary_mask, rhs, boundary_val)  # set right hand side on boundaries

            def matmul(rhs):
                nx, ny = rhs.shape

                res = npx.zeros_like(rhs)
                for diag, (di, dj) in zip(matrix_diags, offsets):
                    assert diag.shape == (nx, ny)
                    i_s = min(max(di, 0), nx - 1)
                    i_e = min(max(nx + di, 1), nx)
                    j_s = min(max(dj, 0), ny - 1)
                    j_e = min(max(ny + dj, 1), ny)

                    i_s_inv = nx - i_e
                    i_e_inv = nx - i_s
                    j_s_inv = ny - j_e
                    j_e_inv = ny - j_s

                    res = update_add(
                        res,
                        at[i_s_inv:i_e_inv, j_s_inv:j_e_inv],
                        diag[i_s_inv:i_e_inv, j_s_inv:j_e_inv] * rhs[i_s:i_e, j_s:j_e],
                    )

                return res

            linear_solution, _ = bicgstab(
                matmul,
                rhs[2:-2, 2:-2] * self._rhs_scale,
                x0=x0[2:-2, 2:-2],
                tol=0,
                atol=1e-8,
                maxiter=10_000,
            )

            return update(rhs, at[2:-2, 2:-2], linear_solution)

        self._linear_solve = linear_solve
        self._rhs_scale = jacobi_precon

    def solve(self, state, rhs, x0, boundary_val=None):
        """
        Main solver for streamfunction. Solves a 2D Poisson equation. Uses jax.scipy.sparse.linalg
        linear solvers.

        Arguments:
            rhs: Right-hand side vector
            x0: Initial guess
            boundary_val: Array containing values to set on boundary elements. Defaults to `x0`.

        """
        if rst.proc_rank == 0:
            linear_solve = self._linear_solve
        else:
            linear_solve = None

        return solve_kernel(state, rhs, x0, boundary_val, linear_solve)

    @staticmethod
    def _jacobi_preconditioner(state, matrix_diags):
        """
        Construct a simple Jacobi preconditioner
        """
        eps = 1e-20
        main_diag = matrix_diags[0]
        precon = npx.where(npx.abs(main_diag) > eps, 1.0 / main_diag, 1.0)
        return precon

    @staticmethod
    def _assemble_poisson_matrix(state):
        """
        Construct a sparse matrix based on the stencil for the 2D Poisson equation.
        """
        vs = state.variables
        settings = state.settings

        boundary_mask = ~npx.any(vs.boundary_mask[2:-2, 2:-2], axis=2)

        # assemble diagonals
        main_diag = allocate(state.dimensions, ("xu", "yu"), fill=1, local=False, include_ghosts=False)
        east_diag, west_diag, north_diag, south_diag = (
            allocate(state.dimensions, ("xu", "yu"), local=False, include_ghosts=False) for _ in range(4)
        )
        main_diag = (
            -vs.hvr[3:-1, 2:-2]
            / vs.dxu[2:-2, npx.newaxis]
            / vs.dxt[3:-1, npx.newaxis]
            / vs.cosu[npx.newaxis, 2:-2] ** 2
            - vs.hvr[2:-2, 2:-2]
            / vs.dxu[2:-2, npx.newaxis]
            / vs.dxt[2:-2, npx.newaxis]
            / vs.cosu[npx.newaxis, 2:-2] ** 2
            - vs.hur[2:-2, 2:-2]
            / vs.dyu[npx.newaxis, 2:-2]
            / vs.dyt[npx.newaxis, 2:-2]
            * vs.cost[npx.newaxis, 2:-2]
            / vs.cosu[npx.newaxis, 2:-2]
            - vs.hur[2:-2, 3:-1]
            / vs.dyu[npx.newaxis, 2:-2]
            / vs.dyt[npx.newaxis, 3:-1]
            * vs.cost[npx.newaxis, 3:-1]
            / vs.cosu[npx.newaxis, 2:-2]
        )
        east_diag = (
            vs.hvr[3:-1, 2:-2] / vs.dxu[2:-2, npx.newaxis] / vs.dxt[3:-1, npx.newaxis] / vs.cosu[npx.newaxis, 2:-2] ** 2
        )
        west_diag = (
            vs.hvr[2:-2, 2:-2] / vs.dxu[2:-2, npx.newaxis] / vs.dxt[2:-2, npx.newaxis] / vs.cosu[npx.newaxis, 2:-2] ** 2
        )
        north_diag = (
            vs.hur[2:-2, 3:-1]
            / vs.dyu[npx.newaxis, 2:-2]
            / vs.dyt[npx.newaxis, 3:-1]
            * vs.cost[npx.newaxis, 3:-1]
            / vs.cosu[npx.newaxis, 2:-2]
        )
        south_diag = (
            vs.hur[2:-2, 2:-2]
            / vs.dyu[npx.newaxis, 2:-2]
            / vs.dyt[npx.newaxis, 2:-2]
            * vs.cost[npx.newaxis, 2:-2]
            / vs.cosu[npx.newaxis, 2:-2]
        )

        if settings.enable_cyclic_x:
            # couple edges of the domain
            wrap_diag_east, wrap_diag_west = (
                allocate(state.dimensions, ("xu", "yu"), local=False, include_ghosts=False) for _ in range(2)
            )
            wrap_diag_east = update(wrap_diag_east, at[0, :], west_diag[0, :] * boundary_mask[0, :])
            wrap_diag_west = update(wrap_diag_west, at[-1, :], east_diag[-1, :] * boundary_mask[-1, :])
            west_diag = update(west_diag, at[0, :], 0.0)
            east_diag = update(east_diag, at[-1, :], 0.0)

        main_diag = main_diag * boundary_mask
        main_diag = npx.where(main_diag == 0.0, 1.0, main_diag)

        # construct sparse matrix diagonals
        cf = (
            main_diag,
            boundary_mask * east_diag,
            boundary_mask * west_diag,
            boundary_mask * north_diag,
            boundary_mask * south_diag,
        )
        offsets = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
        if settings.enable_cyclic_x:
            cf += (wrap_diag_east, wrap_diag_west)
            offsets += ((settings.nx, 0), (-settings.nx, 0))

        return cf, offsets
