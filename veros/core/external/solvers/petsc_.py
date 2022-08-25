import os

from petsc4py import PETSc
import numpy as onp

from veros import logger, veros_kernel, runtime_settings as rs, runtime_state as rst
from veros.core import utilities
from veros.core.external.solvers.base import LinearSolver
from veros.core.operators import numpy as npx, update, update_add, at, flush
from veros.core.external.poisson_matrix import assemble_poisson_matrix

STREAM_OPTIONS = {
    "solver_type": "bcgs",
    "atol": 1e-24,
    "rtol": 1e-14,
    "max_it": 1000,
    "PC_type": "gamg",
    "pc_options": {
        "pc_gamg_type": "agg",
        "pc_gamg_reuse_interpolation": True,
        "pc_gamg_threshold": 1e-4,
        "pc_gamg_sym_graph": True,
        "pc_gamg_agg_nsmooths": 2,
        "mg_levels_pc_type": "jacobi",
    },
}

PRESSURE_OPTIONS = {
    "solver_type": "bcgs",
    "atol": 1e-24,
    "rtol": 1e-14,
    "max_it": 1000,
    "PC_type": "gamg",
    "pc_options": {
        "pc_gamg_type": "agg",
        "pc_gamg_reuse_interpolation": True,
        "pc_gamg_threshold": 1e-4,
        "pc_gamg_sym_graph": True,
        "pc_gamg_agg_nsmooths": 2,
        "mg_levels_pc_type": "jacobi",
    },
}


class PETScSolver(LinearSolver):
    def __init__(self, state):
        if rst.proc_num > 1 and rs.device == "cpu" and "OMP_NUM_THREADS" not in os.environ:
            logger.warning(
                "Environment variable OMP_NUM_THREADS is not set, which can lead to severely "
                "degraded performance when MPI is used."
            )

        settings = state.settings
        if settings.enable_streamfunction:
            options = STREAM_OPTIONS
        else:
            options = PRESSURE_OPTIONS

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

        self._matrix, self._boundary_mask = self._assemble_poisson_matrix(state)

        petsc_options = PETSc.Options()

        # setup krylov method
        self._ksp = PETSc.KSP()
        self._ksp.create(self._da.comm)
        self._ksp.setOperators(self._matrix)

        self._ksp.setType(options["solver_type"])
        self._ksp.setTolerances(atol=options["atol"], rtol=options["rtol"], max_it=options["max_it"])

        # preconditioner
        self._ksp.getPC().setType(options["PC_type"])

        for key in options["pc_options"]:
            petsc_options[key] = options["pc_options"][key]

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
            rel_residual = residual_norm / max(rhs_norm, 1e-22)

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
        rhs, x0 = prepare_solver_inputs(state, rhs, x0, boundary_val, self._boundary_mask, self._boundary_fac)
        linear_solution = self._petsc_solver(rhs, x0)
        return update(rhs, at[2:-2, 2:-2], linear_solution)

    def _assemble_poisson_matrix(self, state):
        diags, offsets, boundary_mask = assemble_poisson_matrix(state)
        diags = onp.asarray(diags, dtype=onp.float64)
        diags = diags[:, 2:-2, 2:-2]

        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()

        (i0, i1), (j0, j1) = self._da.getRanges()
        matrix = self._da.getMatrix()

        for j in range(j0, j1):
            for i in range(i0, i1):
                iloc, jloc = i % (state.settings.nx // rs.num_proc[0]), j % (state.settings.ny // rs.num_proc[1])
                row.index = (i, j)
                for diag, offset in zip(diags, offsets):
                    io, jo = (i + offset[0], j + offset[1])
                    col.index = (io, jo)
                    matrix.setValueStencil(row, col, diag[iloc, jloc])

        matrix.assemble()

        self._boundary_fac = {
            "east": npx.asarray(diags[1][-1, :]),
            "west": npx.asarray(diags[2][0, :]),
            "north": npx.asarray(diags[3][:, -1]),
            "south": npx.asarray(diags[4][:, 0]),
        }

        return matrix, boundary_mask


@veros_kernel
def prepare_solver_inputs(state, rhs, x0, boundary_val, boundary_mask, boundary_fac):
    settings = state.settings

    if boundary_val is None:
        boundary_val = x0

    x0 = utilities.enforce_boundaries(x0, settings.enable_cyclic_x)

    rhs = npx.where(boundary_mask, rhs, boundary_val)  # set right hand side on boundaries

    if settings.enable_streamfunction:
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
