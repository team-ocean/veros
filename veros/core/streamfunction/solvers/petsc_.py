import os

from petsc4py import PETSc
import numpy as onp

from veros import logger, veros_kernel, runtime_settings as rs, runtime_state as rst
from veros.core import utilities
from veros.core.streamfunction.solvers.base import LinearSolver
from veros.core.operators import numpy as npx, update, update_add, at, flush


class PETScSolver(LinearSolver):
    def __init__(self, state):
        if rst.proc_num > 1 and rs.device == "cpu" and "OMP_NUM_THREADS" not in os.environ:
            logger.warning(
                "Environment variable OMP_NUM_THREADS is not set, which can lead to severely "
                "degraded performance when MPI is used."
            )

        settings = state.settings

        if settings.enable_cyclic_x:
            boundary_type = ("periodic", "ghosted")
        else:
            boundary_type = ("ghosted", "ghosted")

        self._da = PETSc.DMDA().create(
            [settings.nx, settings.ny],
            stencil_width=1,
            stencil_type="star",
            comm=rs.mpi_comm,
            proc_sizes=rs.num_proc,
            boundary_type=boundary_type,
            ownership_ranges=[
                (settings.nx // rs.num_proc[0],) * rs.num_proc[0],
                (settings.ny // rs.num_proc[1],) * rs.num_proc[1],
            ],
        )

        if rs.device == "gpu":
            self._da.setVecType("cuda")
            self._da.setMatType("aijcusparse")

        self._matrix, self._boundary_fac = self._assemble_poisson_matrix(state)

        petsc_options = PETSc.Options()

        # setup krylov method
        self._ksp = PETSc.KSP()
        self._ksp.create(self._da.comm)
        self._ksp.setOperators(self._matrix)

        self._ksp.setType("bcgs")
        self._ksp.setTolerances(atol=1e-24, rtol=1e-14, max_it=1000)

        # preconditioner
        self._ksp.getPC().setType("gamg")
        petsc_options["pc_gamg_type"] = "agg"
        petsc_options["pc_gamg_reuse_interpolation"] = True
        petsc_options["pc_gamg_threshold"] = 1e-4
        petsc_options["pc_gamg_sym_graph"] = True
        petsc_options["pc_gamg_agg_nsmooths"] = 2
        petsc_options["mg_levels_pc_type"] = "jacobi"

        if rs.petsc_options:
            petsc_options.insertString(rs.petsc_options)

        self._ksp.setFromOptions()
        self._ksp.getPC().setFromOptions()

        self._rhs_petsc = self._da.createGlobalVec()
        self._sol_petsc = self._da.createGlobalVec()

    def _petsc_solver(self, rhs, x0):
        # hangs on multi-GPU without this
        flush()

        self._da.getVecArray(self._rhs_petsc)[...] = rhs[2:-2, 2:-2]
        self._da.getVecArray(self._sol_petsc)[...] = x0[2:-2, 2:-2]

        self._ksp.solve(self._rhs_petsc, self._sol_petsc)

        info = self._ksp.getConvergedReason()
        iterations = self._ksp.getIterationNumber()

        if info < 0:
            logger.warning(f"Streamfunction solver did not converge after {iterations} iterations (error code: {info})")

        if rs.monitor_streamfunction_residual:
            # re-use rhs vector to store residual
            rhs_norm = self._rhs_petsc.norm(PETSc.NormType.NORM_2)
            self._matrix.multAdd(self._sol_petsc, -self._rhs_petsc, self._rhs_petsc)
            residual_norm = self._rhs_petsc.norm(PETSc.NormType.NORM_2)
            rel_residual = residual_norm / rhs_norm

            if rel_residual > 1e-8:
                logger.warning(
                    f"Streamfunction solver did not achieve required precision (rel. residual: {rel_residual:.2e})"
                )

        return npx.asarray(self._da.getVecArray(self._sol_petsc)[...])

    def solve(self, state, rhs, x0, boundary_val=None):
        """
        Arguments:
            rhs: Right-hand side vector
            x0: Initial guess
            boundary_val: Array containing values to set on boundary elements. Defaults to `x0`.
        """
        rhs, x0 = prepare_solver_inputs(state, rhs, x0, boundary_val, self._boundary_fac)
        linear_solution = self._petsc_solver(rhs, x0)
        return update(rhs, at[2:-2, 2:-2], linear_solution)

    def _assemble_poisson_matrix(self, state):
        """
        Construct a sparse matrix based on the stencil for the 2D Poisson equation.
        """
        vs = state.variables
        settings = state.settings

        matrix = self._da.getMatrix()

        boundary_mask = ~npx.any(vs.boundary_mask[2:-2, 2:-2], axis=2)

        # assemble diagonals
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

        main_diag = npx.where(boundary_mask, main_diag, 0.0)
        main_diag = npx.where(main_diag == 0.0, 1.0, main_diag)

        # construct sparse matrix
        cf = tuple(
            # copy to NumPy for speed
            onp.asarray(diag)
            for diag in (
                main_diag,
                boundary_mask * east_diag,
                boundary_mask * west_diag,
                boundary_mask * north_diag,
                boundary_mask * south_diag,
            )
        )

        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()

        ij_offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

        (i0, i1), (j0, j1) = self._da.getRanges()
        for j in range(j0, j1):
            for i in range(i0, i1):
                iloc, jloc = i % (settings.nx // rs.num_proc[0]), j % (settings.ny // rs.num_proc[1])
                row.index = (i, j)

                for diag, offset in zip(cf, ij_offsets):
                    io, jo = (i + offset[0], j + offset[1])
                    col.index = (io, jo)
                    matrix.setValueStencil(row, col, diag[iloc, jloc])

        matrix.assemble()

        boundary_scale = {
            "east": npx.asarray(cf[1][-1, :]),
            "west": npx.asarray(cf[2][0, :]),
            "north": npx.asarray(cf[3][:, -1]),
            "south": npx.asarray(cf[4][:, 0]),
        }

        return matrix, boundary_scale


@veros_kernel
def prepare_solver_inputs(state, rhs, x0, boundary_val, boundary_fac):
    vs = state.variables
    settings = state.settings

    if boundary_val is None:
        boundary_val = x0

    x0 = utilities.enforce_boundaries(x0, settings.enable_cyclic_x)

    boundary_mask = ~npx.any(vs.boundary_mask, axis=2)
    rhs = npx.where(boundary_mask, rhs, boundary_val)  # set right hand side on boundaries

    # add dirichlet BC to rhs
    if not settings.enable_cyclic_x:
        if rst.proc_idx[0] == rs.num_proc[0] - 1:
            rhs = update_add(rhs, at[-3, 2:-2], -rhs[-2, 2:-2] * boundary_fac["east"])

        if rst.proc_idx[0] == 0:
            rhs = update_add(rhs, at[2, 2:-2], -rhs[1, 2:-2] * boundary_fac["west"])

    if rst.proc_idx[1] == rs.num_proc[1] - 1:
        rhs = update_add(rhs, at[2:-2, -3], -rhs[2:-2, -2] * boundary_fac["north"])

    if rst.proc_idx[1] == 0:
        rhs = update_add(rhs, at[2:-2, 2], -rhs[2:-2, 1] * boundary_fac["south"])

    return rhs, x0
