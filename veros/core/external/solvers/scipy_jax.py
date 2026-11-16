from veros import distributed, veros_routine, veros_kernel, runtime_state as rst, runtime_settings as rs
from veros.variables import allocate

from veros.core.operators import update, update_add, at, numpy as npx
from veros.core.external.solvers.base import LinearSolver
from veros.core.external.poisson_matrix import assemble_poisson_matrix


@veros_kernel(static_args=("solve_fun",))
def solve_kernel(state, rhs, boundary_val, coefficients, solve_fun):
    from veros.routines import CURRENT_CONTEXT

    if rst.proc_num > 1 and getattr(CURRENT_CONTEXT, "tracing", False):
        from veros._distributed_jax import gather, local_slice

        rhs_global = gather(rhs, state.dimensions, ("xt", "yt"))
        boundary_global = gather(boundary_val, state.dimensions, ("xt", "yt"))
        solution, converged = solve_fun(rhs_global, boundary_global, coefficients)
        return local_slice(solution, state.dimensions, ("xt", "yt")), converged

    rhs_global = distributed.gather(rhs, state.dimensions, ("xt", "yt"))
    boundary_val_global = distributed.gather(boundary_val, state.dimensions, ("xt", "yt"))

    if rst.proc_rank == 0:
        linear_solution, converged = solve_fun(rhs_global, boundary_val_global, coefficients)
    else:
        linear_solution = npx.empty_like(rhs)
        converged = npx.asarray(True)

    solution = distributed.scatter(linear_solution, state.dimensions, ("xt", "yt"))
    if rst.proc_num > 1:
        converged = distributed.bcast(converged, comm=rs.mpi_comm)
    return solution, converged


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
        matrix_diags, offsets, boundary_mask = self._assemble_poisson_matrix(state)
        jacobi_precon = self._jacobi_preconditioner(state, matrix_diags)
        matrix_diags = tuple(jacobi_precon * diag for diag in matrix_diags)

        self._set_coefficients(matrix_diags, offsets, boundary_mask, jacobi_precon)

    def _set_coefficients(self, matrix_diags, offsets, boundary_mask, rhs_scale):
        import jax
        from jax.scipy.sparse.linalg import bicgstab, gmres

        self._coefficients = (matrix_diags, offsets, boundary_mask, rhs_scale)

        @veros_kernel
        def linear_solve(rhs, boundary_val, coefficients):
            matrix_diags, boundary_mask, rhs_scale = coefficients
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

            def solve(operator, value):
                # Recompute tolerances for each primal, tangent, or adjoint RHS.
                # Reusing primal tolerances can discard small sensitivities.
                # Avoid norm underflow and GMRES's sub-epsilon cutoff.
                magnitude = npx.max(npx.abs(value))
                scale = npx.where(magnitude > 0, magnitude, 1)
                normalized = value / scale
                tolerance = 10 * npx.finfo(value.dtype).eps
                guess = npx.zeros(value.shape, dtype=value.dtype)
                solution, _ = bicgstab(operator, normalized, x0=guess, tol=tolerance, atol=0, maxiter=10_000)
                # Rescaling can move the true residual across the convergence limit.
                # Meet the physical gate without discarding small sensitivities.
                residual = (operator(scale * solution) - value) / scale
                normalized_norm = npx.linalg.norm(normalized)
                relative_tolerance = npx.maximum(1e-8, 100 * npx.finfo(value.dtype).eps) * normalized_norm
                physical_tolerance = npx.maximum(1e-8 / scale, 100 * npx.finfo(value.dtype).eps * normalized_norm)
                converged = npx.all(npx.isfinite(solution)) & (
                    npx.linalg.norm(residual) <= npx.minimum(relative_tolerance, physical_tolerance)
                )

                def fallback(_):
                    # Identity boundary rows can cause BiCGSTAB breakdown.
                    return gmres(
                        operator,
                        normalized,
                        x0=guess,
                        tol=tolerance,
                        atol=0,
                        restart=40,
                        maxiter=250,
                        solve_method="incremental",
                    )[0]

                return scale * jax.lax.cond(converged, lambda _: solution, fallback, None)

            scaled_rhs = rhs * rhs_scale
            linear_solution = jax.lax.custom_linear_solve(matmul, scaled_rhs, solve=solve, transpose_solve=solve)

            residual = matmul(linear_solution) - scaled_rhs
            tolerance = npx.maximum(1e-8, 100 * npx.finfo(rhs.dtype).eps * npx.linalg.norm(scaled_rhs))
            converged = npx.all(npx.isfinite(linear_solution)) & (npx.linalg.norm(residual) <= tolerance)
            return linear_solution, converged

        self._linear_solve = linear_solve

    def prepare_compiled(self):
        """Replicate fixed solver coefficients before distributed tracing."""
        if rst.proc_num == 1:
            return
        import jax
        import numpy as np

        coefficients = jax.tree.map(np.asarray, self._coefficients) if rst.proc_rank == 0 else None
        coefficients = rs.mpi_comm.bcast(coefficients, root=0)
        matrix_diags, offsets, boundary_mask, rhs_scale = coefficients
        offsets = tuple(tuple(int(value) for value in offset) for offset in offsets)
        self._set_coefficients(
            tuple(npx.asarray(diag) for diag in matrix_diags),
            offsets,
            npx.asarray(boundary_mask),
            npx.asarray(rhs_scale),
        )

    def solve(self, state, rhs, x0, boundary_val=None):
        """
        Main solver for streamfunction. Solves a 2D Poisson equation. Uses jax.scipy.sparse.linalg
        linear solvers.

        Arguments:
            rhs: Right-hand side vector
            x0: Default boundary values. Iterative solves start from zero.
            boundary_val: Array containing values to set on boundary elements. Defaults to `x0`.

        """
        from veros.routines import CURRENT_CONTEXT

        if rst.proc_rank == 0 or getattr(CURRENT_CONTEXT, "tracing", False):
            linear_solve = self._linear_solve
            if state.settings.enable_partial_cells:
                coefficients = self.geometry_coefficients(state)
            else:
                matrix_diags, _, mask, scale = self._coefficients
                coefficients = (matrix_diags, mask, scale)
        else:
            linear_solve = None
            coefficients = None
        boundary_val = x0 if boundary_val is None else boundary_val
        solution, converged = solve_kernel(state, rhs, boundary_val, coefficients, linear_solve)
        if hasattr(state, "_solver_residual_ok"):
            state._solver_residual_ok = state._solver_residual_ok & converged
        return solution

    def geometry_coefficients(self, state):
        """Assemble the current numerical operator with differentiable coefficients."""
        matrix_diags, _, boundary_mask = self._assemble_poisson_matrix(state)
        preconditioner = self._jacobi_preconditioner(state, matrix_diags)
        return tuple(preconditioner * diag for diag in matrix_diags), boundary_mask, preconditioner

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
