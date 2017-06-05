import warnings
import scipy.sparse
import scipy.sparse.linalg as spalg

try:
    import pyamg
    has_pyamg = True
except ImportError:
    warnings.warn("pyamg was not found, falling back to SciPy CG solver")
    has_pyamg = False

from .. import cyclic
from ... import veros_method


@veros_method
def solve(veros, rhs, sol, boundary_val=None):
    """
    Main solver for streamfunction. Solves a 2D Poisson equation. Uses either pyamg
    or scipy.sparse.linalg linear solvers.

    Arguments:
        rhs: Right-hand side vector
        sol: Initial guess, gets overwritten with solution
        boundary_val: Array containing values to set on boundary elements. Defaults to `sol`.
    """
    veros.flush()
    # only initialize solver if parent object changes
    if not solve.veros or solve.veros != id(veros):
        solve.linear_solver = _get_scipy_solver(veros)
        solve.veros = id(veros)

    if veros.enable_cyclic_x:
        cyclic.setcyclic_x(sol)

    if boundary_val is None:
        boundary_val = sol

    z = np.prod(~veros.boundary_mask, axis=2).astype(np.bool)
    rhs[...] = np.where(z, rhs, boundary_val) # set right hand side on boundaries
    linear_solution = solve.linear_solver(rhs, sol)
    sol[...] = boundary_val
    sol[2:-2, 2:-2] = linear_solution.reshape(veros.nx + 4, veros.ny + 4)[2:-2, 2:-2]


solve.veros = None


@veros_method
def _get_scipy_solver(veros):
    matrix = _assemble_poisson_matrix(veros)
    preconditioner = _jacobi_preconditioner(veros, matrix)
    preconditioner.diagonal()[...] *= np.prod(~veros.boundary_mask, axis=2).astype(np.int).flatten()
    matrix = preconditioner * matrix
    extra_args = {}

    if has_pyamg:
        if veros.backend_name == "bohrium":
            near_null_space = np.ones(matrix.shape[0], bohrium=False)
        else:
            near_null_space = np.ones(matrix.shape[0])
        ml = pyamg.smoothed_aggregation_solver(matrix, near_null_space[:, np.newaxis])
        extra_args["M"] = ml.aspreconditioner()

    def scipy_solver(rhs, x0):
        #if veros.backend_name == "bohrium":
        #    rhs = rhs.copy2numpy()
        #    x0 = x0.copy2numpy()
        rhs = rhs.flatten() * preconditioner.diagonal()
        solution, info = spalg.bicgstab(matrix, rhs,
                                        x0=x0.flatten(), tol=veros.congr_epsilon,
                                        maxiter=veros.congr_max_iterations,
                                        **extra_args)
        if info > 0:
            warnings.warn("Streamfunction solver did not converge after {} iterations".format(info))
        return solution
    return scipy_solver


@veros_method
def _jacobi_preconditioner(veros, matrix):
    """
    Construct a simple Jacobi preconditioner
    """
    Z = np.ones((veros.nx + 4, veros.ny + 4))
    Y = matrix.diagonal().copy().reshape(veros.nx + 4, veros.ny + 4)[2:-2, 2:-2]
    Z[2:-2, 2:-2] = np.where(Y != 0., 1. / Y, 1.)
    return scipy.sparse.dia_matrix((Z.flatten(), 0), shape=(Z.size, Z.size)).tocsr()


@veros_method
def _assemble_poisson_matrix(veros):
    """
    Construct a sparse matrix based on the stencil for the 2D Poisson equation.
    """
    # assemble diagonals
    main_diag = np.ones((veros.nx + 4, veros.ny + 4))
    east_diag, west_diag, north_diag, south_diag = (
        np.zeros((veros.nx + 4, veros.ny + 4)) for _ in range(4))
    main_diag[2:-2, 2:-2] = -veros.hvr[3:-1, 2:-2] / veros.dxu[2:-2, np.newaxis] / veros.dxt[3:-1, np.newaxis] / veros.cosu[np.newaxis, 2:-2]**2 \
        - veros.hvr[2:-2, 2:-2] / veros.dxu[2:-2, np.newaxis] / veros.dxt[2:-2, np.newaxis] / veros.cosu[np.newaxis, 2:-2]**2 \
        - veros.hur[2:-2, 2:-2] / veros.dyu[np.newaxis, 2:-2] / veros.dyt[np.newaxis, 2:-2] * veros.cost[np.newaxis, 2:-2] / veros.cosu[np.newaxis, 2:-2] \
        - veros.hur[2:-2, 3:-1] / veros.dyu[np.newaxis, 2:-2] / veros.dyt[np.newaxis, 3:-1] * veros.cost[np.newaxis, 3:-1] / veros.cosu[np.newaxis, 2:-2]
    east_diag[2:-2, 2:-2] = veros.hvr[3:-1, 2:-2] / veros.dxu[2:-2, np.newaxis] / \
        veros.dxt[3:-1, np.newaxis] / veros.cosu[np.newaxis, 2:-2]**2
    west_diag[2:-2, 2:-2] = veros.hvr[2:-2, 2:-2] / veros.dxu[2:-2, np.newaxis] / \
        veros.dxt[2:-2, np.newaxis] / veros.cosu[np.newaxis, 2:-2]**2
    north_diag[2:-2, 2:-2] = veros.hur[2:-2, 3:-1] / veros.dyu[np.newaxis, 2:-2] / \
        veros.dyt[np.newaxis, 3:-1] * veros.cost[np.newaxis, 3:-1] / veros.cosu[np.newaxis, 2:-2]
    south_diag[2:-2, 2:-2] = veros.hur[2:-2, 2:-2] / veros.dyu[np.newaxis, 2:-2] / \
        veros.dyt[np.newaxis, 2:-2] * veros.cost[np.newaxis, 2:-2] / veros.cosu[np.newaxis, 2:-2]
    z = np.prod(np.invert(veros.boundary_mask), axis=2)  # used to enforce boundary conditions
    if veros.enable_cyclic_x:
        # couple edges of the domain
        wrap_diag_east, wrap_diag_west = (np.zeros((veros.nx + 4, veros.ny + 4)) for _ in range(2))
        wrap_diag_east[2, 2:-2] = west_diag[2, 2:-2] * z[2, 2:-2]
        wrap_diag_west[-3, 2:-2] = east_diag[-3, 2:-2] * z[-3, 2:-2]
        west_diag[2, 2:-2] = 0.
        east_diag[-3, 2:-2] = 0.

    # construct sparse matrix
    cf = tuple(diag.flatten() for diag in (z * main_diag + (1 - z), z *
                                           east_diag, z * west_diag, z * north_diag, z * south_diag))
    offsets = (0, -main_diag.shape[1], main_diag.shape[1], -1, 1)

    if veros.enable_cyclic_x:
        offsets += (-main_diag.shape[1] * (veros.nx - 1), main_diag.shape[1] * (veros.nx - 1))
        cf += (wrap_diag_east.flatten(), wrap_diag_west.flatten())

    if veros.backend_name == "bohrium":
        cf = np.array(cf, bohrium=False)
    else:
        cf = np.array(cf)
    return scipy.sparse.dia_matrix((cf, offsets), shape=(main_diag.size, main_diag.size)).T.tocsr()
