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

        settings.enable_cyclic_x = cyclic

    state.initialize_variables()
    resize_dimension(state, "isle", 1)

    vs = state.variables

    with vs.unlock():
        vs.dxt = 10e3 * np.ones(settings.nx + 4)
        vs.dxu = 10e3 * np.ones(settings.nx + 4)

        vs.dyt = 10e3 * np.ones(settings.ny + 4)
        vs.dyu = 10e3 * np.ones(settings.ny + 4)

        vs.hur = 1.0 / np.linspace(500, 2000, settings.nx + 4)[:, None] * np.ones((settings.nx + 4, settings.ny + 4))
        vs.hvr = 1.0 / np.linspace(500, 2000, settings.ny + 4)[None, :] * np.ones((settings.nx + 4, settings.ny + 4))

        vs.cosu = np.ones(settings.ny + 4)
        vs.cost = np.ones(settings.ny + 4)

        boundary_mask = np.zeros((settings.nx + 4, settings.ny + 4, settings.nz), dtype="bool")
        boundary_mask[:, :2] = 1
        boundary_mask[50:100, 50:100] = 1
        vs.boundary_mask = boundary_mask

    return state


def assert_solution(state, rhs, sol, boundary_val=None, tol=1e-8):
    from veros.core.external.solvers.scipy import SciPySolver

    matrix = SciPySolver._assemble_poisson_matrix(state)

    if boundary_val is None:
        boundary_val = sol
    
    if state.settings.enable_streamfunction:
        boundary_mask = ~np.any(state.variables.boundary_mask, axis=2)
        rhs = np.where(boundary_mask, rhs, boundary_val)

    rhs_sol = matrix @ sol.reshape(-1)

    np.testing.assert_allclose(rhs_sol, rhs.flatten(), atol=0, rtol=tol)


@pytest.mark.parametrize("cyclic", [True, False])
@pytest.mark.parametrize("solver", ["scipy", "scipy_jax", "petsc"])
@pytest.mark.parametrize("streamfunction", [True, False])
def test_solver(solver, solver_state, cyclic, streamfunction):
    from veros import runtime_settings
    from veros.core.operators import numpy as npx
    with solver_state.settings.unlock():
        solver_state.settings.enable_streamfunction = streamfunction

    if solver == "scipy":
        from veros.core.external.solvers.scipy import SciPySolver

        solver_class = SciPySolver
    elif solver == "scipy_jax":
        if runtime_settings.backend != "jax":
            pytest.skip("scipy_jax solver requires JAX")

        from veros.core.external.solvers.scipy_jax import JAXSciPySolver

        solver_class = JAXSciPySolver
    elif solver == "petsc":
        petsc_mod = pytest.importorskip("veros.core.external.solvers.petsc_")
        solver_class = petsc_mod.PETScSolver
    else:
        raise ValueError("unknown solver")

    settings = solver_state.settings

    rhs = npx.ones((settings.nx + 4, settings.ny + 4))
    x0 = npx.asarray(np.random.rand(settings.nx + 4, settings.ny + 4))

    sol = solver_class(solver_state).solve(solver_state, rhs, x0, boundary_val=10)

    assert_solution(solver_state, rhs, sol, boundary_val=10, tol=1e-8)
