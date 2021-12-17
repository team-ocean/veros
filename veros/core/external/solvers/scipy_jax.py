from veros import distributed, veros_routine, veros_kernel, runtime_state as rst
from veros.variables import allocate

from veros.core.operators import update, update_add, at, numpy as npx
from veros.core.external.solvers.base import LinearSolver
from veros.core.external.poisson_matrix import assemble_poisson_matrix


@veros_kernel(static_args=("solve_fun",))
def solve_kernel(state, rhs, x0, boundary_val, solve_fun):
    rhs_global = distributed.gather(rhs, state.dimensions, ("xt", "yt"))
    x0_global = distributed.gather(x0, state.dimensions, ("xt", "yt"))

    if boundary_val is None:
        boundary_val_global = x0_global
    else:
        boundary_val_global = distributed.gather(boundary_val, state.dimensions, ("xt", "yt"))

    if rst.proc_rank == 0:
        linear_solution = solve_fun(rhs_global, x0_global, boundary_val_global)
    else:
        linear_solution = npx.empty_like(rhs)

    return distributed.scatter(linear_solution, state.dimensions, ("xt", "yt"))


class JAXSciPySolver(LinearSolver):
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
        from jax.scipy.sparse.linalg import bicgstab

        matrix_diags, offsets, boundary_mask = self._assemble_poisson_matrix(state)
        jacobi_precon = self._jacobi_preconditioner(state, matrix_diags)
        matrix_diags = tuple(jacobi_precon * diag for diag in matrix_diags)

        @veros_kernel
        def linear_solve(rhs, x0, boundary_val):
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
                rhs * self._rhs_scale,
                x0=x0,
                tol=0,
                atol=1e-8,
                maxiter=10_000,
            )

            return linear_solution

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
        precon = allocate(state.dimensions, ("xu", "yu"), fill=1, local=False)
        main_diag = matrix_diags[0][2:-2, 2:-2]
        precon = update(precon, at[2:-2, 2:-2], npx.where(npx.abs(main_diag) > eps, 1.0 / (main_diag + eps), 1.0))
        return precon

    @staticmethod
    def _assemble_poisson_matrix(state):
        settings = state.settings

        matrix_diags, offsets, boundary_mask = assemble_poisson_matrix(state)

        if settings.enable_cyclic_x:
            wrap_diag_east, wrap_diag_west = (allocate(state.dimensions, ("xu", "yu"), local=False) for _ in range(2))
            wrap_diag_east = update(wrap_diag_east, at[2, 2:-2], matrix_diags[2][2, 2:-2] * boundary_mask[2, 2:-2])
            wrap_diag_west = update(wrap_diag_west, at[-3, 2:-2], matrix_diags[1][-3, 2:-2] * boundary_mask[-3, 2:-2])
            matrix_diags[2] = update(matrix_diags[2], at[2, 2:-2], 0.0)
            matrix_diags[1] = update(matrix_diags[1], at[-3, 2:-2], 0.0)

            offsets += ((settings.nx - 1, 0), (-settings.nx + 1, 0))
            matrix_diags += (wrap_diag_east, wrap_diag_west)

        return matrix_diags, offsets, boundary_mask
