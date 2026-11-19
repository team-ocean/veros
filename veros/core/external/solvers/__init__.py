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


def get_linear_solver(state):
    prepared = getattr(state, "_linear_solver", None)
    if prepared is not None:
        return prepared
    return _get_linear_solver(state)


def get_compiled_solver(state, linear_solver=None):
    if linear_solver is None:
        linear_solver = rs.linear_solver
    if linear_solver not in ("best", "petsc", "scipy_jax"):
        raise ValueError("Compiled linear_solver must be 'best', 'petsc', or 'scipy_jax'")
    if state.settings.enable_partial_cells and linear_solver == "petsc":
        raise ValueError("Partial cells require the variable-coefficient scipy_jax solver")
    if linear_solver == "petsc" or (linear_solver == "best" and rst.proc_num > 1):
        try:
            from veros.core.external.solvers.petsc_jax import CompiledPETScSolver
        except ImportError:
            if linear_solver == "petsc":
                raise
            logger.warning("PETSc linear solver not available, falling back to replicated JAX solves")
        else:
            return CompiledPETScSolver(state)
    from veros.core.external.solvers.scipy_jax import JAXSciPySolver

    solver = JAXSciPySolver(state)
    solver.prepare_compiled()
    return solver


@memoize
def _get_linear_solver(state):
    logger.debug("Initializing linear solver")
    if state.settings.enable_partial_cells:
        if rs.linear_solver not in ("best", "scipy_jax"):
            raise ValueError("Partial cells require the variable-coefficient scipy_jax solver")
        from veros.core.external.solvers.scipy_jax import JAXSciPySolver

        SolverClass = JAXSciPySolver
    else:
        SolverClass = _get_solver_class()
    return SolverClass(state)
