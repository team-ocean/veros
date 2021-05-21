import pytest

import numpy as np

from veros.state import get_default_state, resize_dimension


@pytest.fixture
def solver_state(cyclic):
    state = get_default_state()
    settings = state.settings

    with settings.unlock():
        settings.nx = 400
        settings.ny = 200
        settings.nz = 1

        settings.congr_epsilon = 1e-12
        settings.congr_max_iterations = 10000

        settings.enable_cyclic_x = cyclic

    state.initialize_variables()
    resize_dimension(state, "isle", 1)

    vs = state.variables

    with vs.unlock():
        vs.dxt = 1e-12 * np.ones(settings.nx + 4)
        vs.dxu = 1e-12 * np.ones(settings.nx + 4)

        vs.dyt = 1e-12 * np.ones(settings.ny + 4)
        vs.dyu = 1e-12 * np.ones(settings.ny + 4)

        vs.hur = np.linspace(500, 2000, settings.nx + 4)[:, None] * np.ones((settings.nx + 4, settings.ny + 4))
        vs.hvr = np.linspace(500, 2000, settings.ny + 4)[None, :] * np.ones((settings.nx + 4, settings.ny + 4))

        vs.cosu = np.ones(settings.ny + 4)
        vs.cost = np.ones(settings.ny + 4)

        boundary_mask = np.zeros((settings.nx + 4, settings.ny + 4, settings.nz), dtype="bool")
        boundary_mask[:, :2] = 1
        boundary_mask[50:100, 50:100] = 1
        vs.boundary_mask = boundary_mask

    return state


def get_residual(state, rhs, sol, boundary_val=None):
    from veros.core.streamfunction.solvers.scipy import SciPySolver

    scipy_solver = SciPySolver(state)

    if boundary_val is None:
        boundary_val = sol

    boundary_mask = ~np.any(state.variables.boundary_mask, axis=2)
    rhs = np.where(boundary_mask, rhs, boundary_val)
    print(scipy_solver._rhs_scale.max(), scipy_solver._rhs_scale.min())
    residual = scipy_solver._matrix @ sol.reshape(-1) - rhs.reshape(-1) * scipy_solver._rhs_scale
    return residual


@pytest.mark.parametrize("cyclic", [True, False])
@pytest.mark.parametrize("solver", ["scipy", "petsc"])
def test_solver(solver, solver_state, cyclic):
    if solver == "scipy":
        from veros.core.streamfunction.solvers.scipy import SciPySolver

        solver_class = SciPySolver
    elif solver == "petsc":
        petsc_mod = pytest.importorskip("veros.core.streamfunction.solvers.petsc_")
        solver_class = petsc_mod.PETScSolver
    else:
        raise ValueError("unknown solver")

    settings = solver_state.settings

    rhs = np.ones((settings.nx + 4, settings.ny + 4))
    x0 = np.random.rand(settings.nx + 4, settings.ny + 4)

    sol = solver_class(solver_state).solve(solver_state, rhs, x0, boundary_val=10)

    residual = get_residual(solver_state, rhs, sol, boundary_val=10)

    # set tolerance may apply in preconditioned space,
    # so let's allow for some wiggle room
    assert np.max(np.abs(residual)) < settings.congr_epsilon * 1e2
