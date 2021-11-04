import functools

from veros import runtime_settings as rs, runtime_state as rst, logger


def memoize(func):
    func.cache = {}

    @functools.wraps(func)
    def inner(*args):
        if args not in func.cache:
            func.cache[args] = func(*args)

        return func.cache[args]

    return inner


def _get_solver_class():
    ls = rs.linear_solver

    def _get_best_solver():
        if rst.proc_num > 1:
            try:
                from veros.core.external.solvers.petsc_ import PETScSolver
            except ImportError:
                logger.warning("PETSc linear solver not available, falling back to SciPy")
            else:
                return PETScSolver

        if rs.backend == "jax" and rs.device == "gpu" and rs.float_type == "float64":
            from veros.core.external.solvers.scipy_jax import JAXSciPySolver

            return JAXSciPySolver

        from veros.core.external.solvers.scipy import SciPySolver

        return SciPySolver

    if ls == "best":
        return _get_best_solver()
    elif ls == "petsc":
        from veros.core.external.solvers.petsc_ import PETScSolver

        return PETScSolver
    elif ls == "scipy":
        from veros.core.external.solvers.scipy import SciPySolver

        return SciPySolver
    elif ls == "scipy_jax":
        from veros.core.external.solvers.scipy_jax import JAXSciPySolver

        return JAXSciPySolver

    raise ValueError(f"unrecognized linear solver {ls}")


@memoize
def get_linear_solver(state):
    logger.debug("Initializing linear solver")
    SolverClass = _get_solver_class()
    return SolverClass(state)
