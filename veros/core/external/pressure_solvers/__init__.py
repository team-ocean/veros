import functools

from veros import logger


def memoize(func):
    func.cache = {}

    @functools.wraps(func)
    def inner(*args):
        if args not in func.cache:
            func.cache[args] = func(*args)

        return func.cache[args]

    return inner


def _get_solver_class():
    # TODO: implement other solvers
    from veros.core.external.pressure_solvers.scipy_pressure import SciPyPressureSolver

    return SciPyPressureSolver


@memoize
def get_linear_solver(state):
    logger.debug("Initializing linear solver")
    SolverClass = _get_solver_class()
    return SolverClass(state)
