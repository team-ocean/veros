from petsc4py import PETSc
from loguru import logger

from .base import LinearSolver
from ... import utilities
from .... import veros_method, runtime_settings as rs, runtime_state as rst


class PETScSolver(LinearSolver):
    @veros_method
    def __init__(self, vs):
        if vs.enable_cyclic_x:
            boundary_type = ('periodic', 'ghosted')
        else:
            boundary_type = ('ghosted', 'ghosted')

        self._da = PETSc.DMDA().create(
            [vs.nx, vs.ny],
            stencil_width=1,
            stencil_type='star',
            comm=rs.mpi_comm,
            proc_sizes=rs.num_proc,
            boundary_type=boundary_type,
            ownership_ranges=[
                (vs.nx // rs.num_proc[0],) * rs.num_proc[0],
                (vs.ny // rs.num_proc[1],) * rs.num_proc[1],
            ]
        )

        self._matrix, self._boundary_fac = self._assemble_poisson_matrix(vs)

        petsc_options = PETSc.Options()

        # setup krylov method
        self._ksp = PETSc.KSP()
        self._ksp.create(rs.mpi_comm)
        self._ksp.setOperators(self._matrix)
        self._ksp.setType('bcgs')
        self._ksp.setTolerances(atol=0, rtol=vs.congr_epsilon, max_it=vs.congr_max_iterations)

        # preconditioner
        self._ksp.getPC().setType('hypre')
        petsc_options['pc_hypre_type'] = 'boomeramg'
        self._ksp.getPC().setFromOptions()

        self._rhs_petsc = self._da.createGlobalVec()
        self._sol_petsc = self._da.createGlobalVec()

    @veros_method
    def _petsc_solver(self, vs, rhs, x0):
        # add dirichlet BC to rhs
        if not vs.enable_cyclic_x:
            if rst.proc_idx[0] == rs.num_proc[0] - 1:
                rhs[-3, 2:-2] -= rhs[-2, 2:-2] * self._boundary_fac['east']

            if rst.proc_idx[0] == 0:
                rhs[2, 2:-2] -= rhs[1, 2:-2] * self._boundary_fac['west']

        if rst.proc_idx[1] == rs.num_proc[1] - 1:
            rhs[2:-2, -3] -= rhs[2:-2, -2] * self._boundary_fac['north']

        if rst.proc_idx[1] == 0:
            rhs[2:-2, 2] -= rhs[2:-2, 1] * self._boundary_fac['south']

        try:
            rhs = rhs.copy2numpy()
        except AttributeError:
            pass

        try:
            x0 = x0.copy2numpy()
        except AttributeError:
            pass
        
        self._da.getVecArray(self._rhs_petsc)[...] = rhs[2:-2, 2:-2]
        self._da.getVecArray(self._sol_petsc)[...] = x0[2:-2, 2:-2]

        self._ksp.solve(self._rhs_petsc, self._sol_petsc)

        info = self._ksp.getConvergedReason()
        iterations = self._ksp.getIterationNumber()

        if info < 0:
            logger.warning('Streamfunction solver did not converge after {} iterations (error code: {})', iterations, info)

        return np.array(self._da.getVecArray(self._sol_petsc)[...])

    @veros_method
    def solve(self, vs, rhs, sol, boundary_val=None):
        """
        Arguments:
            rhs: Right-hand side vector
            sol: Initial guess, gets overwritten with solution
            boundary_val: Array containing values to set on boundary elements. Defaults to `sol`.
        """
        if boundary_val is None:
            boundary_val = sol

        utilities.enforce_boundaries(vs, sol)

        boundary_mask = np.logical_and.reduce(~vs.boundary_mask, axis=2)
        rhs = utilities.where(vs, boundary_mask, rhs, boundary_val) # set right hand side on boundaries

        sol[...] = rhs
        sol[2:-2, 2:-2] = self._petsc_solver(vs, rhs, sol)

    @veros_method
    def _assemble_poisson_matrix(self, vs):
        """
        Construct a sparse matrix based on the stencil for the 2D Poisson equation.
        """

        matrix = self._da.getMatrix()

        boundary_mask = np.logical_and.reduce(~vs.boundary_mask[2:-2, 2:-2], axis=2)

        # assemble diagonals
        main_diag = -vs.hvr[3:-1, 2:-2] / vs.dxu[2:-2, np.newaxis] / vs.dxt[3:-1, np.newaxis] / vs.cosu[np.newaxis, 2:-2]**2 \
            - vs.hvr[2:-2, 2:-2] / vs.dxu[2:-2, np.newaxis] / vs.dxt[2:-2, np.newaxis] / vs.cosu[np.newaxis, 2:-2]**2 \
            - vs.hur[2:-2, 2:-2] / vs.dyu[np.newaxis, 2:-2] / vs.dyt[np.newaxis, 2:-2] * vs.cost[np.newaxis, 2:-2] / vs.cosu[np.newaxis, 2:-2] \
            - vs.hur[2:-2, 3:-1] / vs.dyu[np.newaxis, 2:-2] / vs.dyt[np.newaxis, 3:-1] * \
            vs.cost[np.newaxis, 3:-1] / vs.cosu[np.newaxis, 2:-2]
        east_diag = vs.hvr[3:-1, 2:-2] / vs.dxu[2:-2, np.newaxis] / \
            vs.dxt[3:-1, np.newaxis] / vs.cosu[np.newaxis, 2:-2]**2
        west_diag = vs.hvr[2:-2, 2:-2] / vs.dxu[2:-2, np.newaxis] / \
            vs.dxt[2:-2, np.newaxis] / vs.cosu[np.newaxis, 2:-2]**2
        north_diag = vs.hur[2:-2, 3:-1] / vs.dyu[np.newaxis, 2:-2] / \
            vs.dyt[np.newaxis, 3:-1] * vs.cost[np.newaxis, 3:-1] / vs.cosu[np.newaxis, 2:-2]
        south_diag = vs.hur[2:-2, 2:-2] / vs.dyu[np.newaxis, 2:-2] / \
            vs.dyt[np.newaxis, 2:-2] * vs.cost[np.newaxis, 2:-2] / vs.cosu[np.newaxis, 2:-2]

        # construct sparse matrix
        cf = tuple(diag for diag in (
            boundary_mask * main_diag + (1 - boundary_mask),
            boundary_mask * east_diag,
            boundary_mask * west_diag,
            boundary_mask * north_diag,
            boundary_mask * south_diag
        ))

        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()

        ij_offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

        (i0, i1), (j0, j1) = self._da.getRanges()
        for j in range(j0, j1):
            for i in range(i0, i1):
                iloc, jloc = i % (vs.nx // rs.num_proc[0]), j % (vs.ny // rs.num_proc[1])
                row.index = (i, j)

                for diag, offset in zip(cf, ij_offsets):
                    io, jo = (i + offset[0], j + offset[1])
                    col.index = (io, jo)
                    matrix.setValueStencil(row, col, diag[iloc, jloc])

        matrix.assemble()

        boundary_scale = {
            'east': cf[1][-1, :],
            'west': cf[2][0, :],
            'north': cf[3][:, -1],
            'south': cf[4][:, 0]
        }

        return matrix, boundary_scale
