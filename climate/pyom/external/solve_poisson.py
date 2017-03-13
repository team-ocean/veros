import warnings
import scipy.sparse

try:
    import pyamg
    has_pyamg = True
except ImportError:
    warnings.warn("pyamg was not found, falling back to SciPy CG solver")
    import scipy.sparse.linalg as spalg
    has_pyamg = False

from climate.pyom import cyclic, pyom_method

@pyom_method
def solve(pyom, rhs, sol, boundary_val=None):
    """
    Main solver for streamfunction. Solves a 2D Poisson equation. Uses either pyamg
    or scipy.sparse.linalg linear solvers.

    :param rhs: Right-hand side vector
    :param sol: Initial guess, gets overwritten with solution
    :param boundary_val: Array containing values to set on boundary elements. Defaults to `sol`.
    """
    pyom.flush()
    if not solve.pyom or solve.pyom != id(pyom): # only initialize solver if parent object changes
        if has_pyamg:
            solve.linear_solver = _get_amg_solver(pyom)
        else:
            solve.linear_solver = _get_scipy_solver(pyom)
        solve.pyom = id(pyom)

    if pyom.enable_cyclic_x:
        cyclic.setcyclic_x(sol)

    if boundary_val is None:
        boundary_val = sol

    z = np.prod(~pyom.boundary_mask, axis=2).astype(np.bool)
    rhs[...] = np.where(z, rhs, boundary_val) # set right hand side on boundaries
    linear_solution = solve.linear_solver(rhs,sol)
    sol[...] = boundary_val
    sol[2:-2,2:-2] = linear_solution.reshape(pyom.nx+4,pyom.ny+4)[2:-2,2:-2]
solve.pyom = None

@pyom_method
def _get_scipy_solver(pyom):
    matrix = _assemble_poisson_matrix(pyom)
    preconditioner = _jacobi_preconditioner(pyom, matrix)
    preconditioner.diagonal()[...] *= np.prod(~pyom.boundary_mask, axis=2).astype(np.int).flatten()
    matrix = preconditioner * matrix
    def scipy_solver(rhs,x0):
        rhs = rhs.flatten() * preconditioner.diagonal()
        solution, info = spalg.bicgstab(matrix, rhs,
                                        x0=x0.flatten(), tol=pyom.congr_epsilon,
                                        maxiter=pyom.congr_max_iterations)
        if info > 0:
            warnings.warn("Streamfunction solver did not converge after {} iterations".format(info))
        return solution
    return scipy_solver

@pyom_method
def _get_amg_solver(pyom):
    matrix = _assemble_poisson_matrix(pyom)
    if pyom.backend_name == "bohrium":
        near_null_space = np.ones(matrix.shape[0], bohrium=False)
    else:
        near_null_space = np.ones(matrix.shape[0])
    ml = pyamg.smoothed_aggregation_solver(matrix, near_null_space)
    def amg_solver(rhs,x0):
        if pyom.backend_name == "bohrium":
            rhs = rhs.copy2numpy()
            x0 = x0.copy2numpy()
        residuals = []
        tolerance = pyom.congr_epsilon * 1e-8 # to achieve the same precision as the preconditioned scipy solver
        solution = ml.solve(b=rhs.flatten(), x0=x0.flatten(), tol=tolerance,
                                         residuals=residuals, accel="bicgstab")
        rel_res = residuals[-1] / residuals[0]
        if rel_res > tolerance:
            warnings.warn("Streamfunction solver did not converge - residual: {:.2e}".format(rel_res))
        return np.asarray(solution)
    return amg_solver

@pyom_method
def _jacobi_preconditioner(pyom, matrix):
    """
    Construct a simple Jacobi preconditioner
    """
    Z = np.ones((pyom.nx+4, pyom.ny+4))
    Y = matrix.diagonal().copy().reshape(pyom.nx+4, pyom.ny+4)[2:-2, 2:-2]
    Z[2:-2, 2:-2] = np.where(Y != 0., 1. / Y, 1.)
    return scipy.sparse.dia_matrix((Z.flatten(),0), shape=(Z.size,Z.size)).tocsr()

@pyom_method
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
    z = np.prod(np.invert(pyom.boundary_mask), axis=2) # used to enforce boundary conditions
    if pyom.enable_cyclic_x:
        # couple edges of the domain
        wrap_diag_east, wrap_diag_west = (np.zeros((pyom.nx+4, pyom.ny+4)) for _ in range(2))
        wrap_diag_east[2, 2:-2] = west_diag[2, 2:-2] * z[2, 2:-2]
        wrap_diag_west[-3, 2:-2] = east_diag[-3, 2:-2] * z[-3, 2:-2]
        west_diag[2, 2:-2] = 0.
        east_diag[-3, 2:-2] = 0.

    # construct sparse matrix
    cf = tuple(diag.flatten() for diag in (z * main_diag + (1-z), z * east_diag, z * west_diag, z * north_diag, z * south_diag))
    offsets = (0, -main_diag.shape[1], main_diag.shape[1], -1, 1)

    if pyom.enable_cyclic_x:
        offsets += (-main_diag.shape[1] * (pyom.nx-1), main_diag.shape[1] * (pyom.nx-1))
        cf += (wrap_diag_east.flatten(), wrap_diag_west.flatten())

    if pyom.backend_name == "bohrium":
        cf = np.array(cf, bohrium=False)
    else:
        cf = np.array(cf)
    return scipy.sparse.dia_matrix((cf, offsets), shape=(main_diag.size, main_diag.size)).T.tocsr()
