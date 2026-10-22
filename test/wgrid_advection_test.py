"""Energy transport uses occupied volumes and donor-volume Courant scaling."""

from copy import copy

import numpy as np
import pytest

from wgrid_geometry_test import basin as basin, candidate, reference_width  # noqa: F401
from wgrid_energy_test import energy_state


@pytest.fixture(scope="module")
def flow_basin(basin):  # noqa: F811
    import jax.numpy as jnp

    _, initial = basin
    vs = initial.variables
    nx, ny = vs.kbot[2:-2, 2:-2].shape
    x, y = np.arange(nx + 1), np.arange(ny + 1)
    if basin[0]._template.settings.enable_cyclic_x:
        psi = 1e5 * np.cos(2 * np.pi * x[:, None] / nx) * np.sin(np.pi * y[None, :] / ny)
        psi[-1] = psi[0]
    else:
        psi = 1e5 * np.sin(np.pi * x[:, None] / nx) * np.sin(np.pi * y[None, :] / ny)
        psi[[0, -1]] = 0
    psi[:, [0, -1]] = 0
    for i, j in np.argwhere(np.asarray(vs.kbot[2:-2, 2:-2]) == 0):
        psi[i : i + 2, j : j + 2] = 0
    return basin, jnp.asarray(psi)


def advected_state(base, psi, scheme, constant=False):
    import jax.numpy as jnp
    from veros.core import advection, momentum, utilities
    from veros.routines import tracing_context

    state = energy_state(base)
    with state.settings.unlock():
        state.settings.c_eps = state.settings.eke_c_eps = 0
        state.settings.alpha_tke = state.settings.alpha_eke = 0
        state.settings.enable_tke_hor_diffusion = False
        for name in ("tke", "eke"):
            setattr(state.settings, "enable_" + name + "_superbee_advection", scheme == "superbee")
            setattr(state.settings, "enable_" + name + "_upwind_advection", scheme == "upwind")
    vs = state.variables
    with vs.unlock(), tracing_context():
        for name in vs.fields():
            if name.startswith(("K_diss_", "P_diss_")) or name == "forc_tke_surface":
                setattr(vs, name, jnp.zeros_like(getattr(vs, name)))
        vs.kappaM = jnp.zeros_like(vs.kappaM)
        if constant:
            for name in ("tke", "eke"):
                setattr(vs, name, jnp.broadcast_to(vs.maskW[..., None], getattr(vs, name).shape).astype(vs.temp.dtype))
        hu = jnp.sum(vs.dzt * vs.hfacU, axis=2)[2:-2, 2:-2]
        hv = jnp.sum(vs.dzt * vs.hfacV, axis=2)[2:-2, 2:-2]
        u = (psi[1:, 1:] - psi[1:, :-1]) / (vs.dyt[None, 2:-2] * jnp.where(hu > 0, hu, 1))
        v = (-psi[1:, 1:] + psi[:-1, 1:]) / (vs.dxt[2:-2, None] * jnp.where(hv > 0, hv, 1))
        vs.u = jnp.zeros_like(vs.u).at[2:-2, 2:-2].set(u[..., None, None] * vs.maskU[2:-2, 2:-2, :, None])
        vs.v = jnp.zeros_like(vs.v).at[2:-2, 2:-2].set(v[..., None, None] * vs.maskV[2:-2, 2:-2, :, None])
        vs.u = utilities.enforce_boundaries(vs.u, state.settings.enable_cyclic_x)
        vs.v = utilities.enforce_boundaries(vs.v, state.settings.enable_cyclic_x)
        vs.update(momentum.vertical_velocity_kernel(state))
        vs.w = jnp.broadcast_to(vs.w[..., vs.taup1, None], vs.w.shape)
        vs.update(advection.calculate_velocity_on_wgrid_kernel(state))
    return state


def integrate(state, name):
    from veros.core import tke, eke
    from veros.routines import tracing_context

    with tracing_context():
        return getattr(tke if name == "tke" else eke, "integrate_" + name + "_kernel")(state)


def face_transfer_scale(state, values, scheme):
    """Normalize cancelling budgets by physical face transfers, not the zero tendency."""
    import jax.numpy as jnp
    from veros.core import advection
    from veros.routines import tracing_context

    vs = state.variables
    with tracing_context():
        east, north, top = getattr(advection, "adv_flux_" + scheme + "_wgrid")(state, values)
    nominal = jnp.broadcast_to(vs.dzw, vs.maskW.shape).at[..., -1].multiply(0.5)[2:-2, 2:-2]
    return float(
        jnp.sum(abs(east[2:-2, 2:-2]) * vs.dyt[None, 2:-2, None] * nominal)
        + jnp.sum(abs(north[2:-2, 2:-2]) * vs.dxt[2:-2, None, None] * nominal)
        + jnp.sum(abs(top[2:-2, 2:-2]) * vs.area_t[2:-2, 2:-2, None])
    )


@pytest.mark.parametrize("scheme", ("upwind", "superbee"))
@pytest.mark.parametrize("name", ("tke", "eke"))
def test_energy_advection_constant_and_inventory(flow_basin, scheme, name):
    import jax.numpy as jnp

    geometry, psi = flow_basin
    for constant in (True, False):
        state = advected_state(candidate(geometry, 0.3), psi, scheme, constant)
        vs = state.variables
        np.testing.assert_allclose(vs.w_wgrid[2:-2, 2:-2, -1], 0, atol=1e-17)
        output = integrate(state, name)
        before = np.asarray(getattr(vs, name)[2:-2, 2:-2, :, vs.tau])
        after = np.asarray(getattr(output, name)[2:-2, 2:-2, :, vs.taup1])
        volume = reference_width(state)[2:-2, 2:-2] * np.asarray(vs.area_t[2:-2, 2:-2, None])
        np.testing.assert_allclose(np.sum(volume * after), np.sum(volume * before), rtol=5e-15, atol=1e-7)
        history = "dtke" if name == "tke" else "deke"
        rate = getattr(output, history)[2:-2, 2:-2, :, vs.tau]
        activity = float(jnp.sum(abs(rate) * volume))
        scale = face_transfer_scale(state, getattr(vs, name)[..., vs.tau], scheme)
        assert abs(float(jnp.sum(rate * volume))) <= 1e-13 * max(scale, 1)
        if constant:
            np.testing.assert_allclose(after, before, rtol=0, atol=1e-14)
        else:
            assert activity > 1e-3


@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.parametrize("velocity", (-2.0, 2.0))
def test_superbee_courant_term_uses_donor_occupation(flow_basin, axis, velocity):
    import jax.numpy as jnp
    from veros.core.advection import adv_flux_superbee_wgrid, wgrid_cfl_spacings
    from veros.routines import tracing_context

    geometry, psi = flow_basin
    state = advected_state(candidate(geometry, 0.3), psi, "superbee")
    vs = state.variables
    shape = vs.maskW.shape
    with vs.unlock():
        setattr(vs, "u_wgrid" if axis == 0 else "v_wgrid", jnp.full(shape, velocity))
    values = jnp.broadcast_to(
        jnp.arange(shape[axis], dtype=vs.temp.dtype).reshape((-1, 1, 1) if axis == 0 else (1, -1, 1)), shape
    )
    index = (4, 3, 3) if axis == 0 else (3, 4, 3)
    donor = list(index)
    if velocity < 0:
        donor[axis] += 1
    width = reference_width(state)
    nominal = np.broadcast_to(np.asarray(vs.dzw), shape).copy()
    nominal[..., -1] *= 0.5
    fraction = width[tuple(donor)] / nominal[tuple(donor)]
    base_distance = float(vs.cost[index[1]] * (vs.dxt[index[0]] if axis == 0 else vs.dyt[index[1]]))
    effective = base_distance * fraction
    np.testing.assert_allclose(wgrid_cfl_spacings(state)[axis][index], effective, rtol=1e-13)
    normal_velocity = velocity * (1 if axis == 0 else float(vs.cosu[index[1]]))
    courant = abs(normal_velocity) * state.settings.dt_tracer / effective
    upper = list(index)
    upper[axis] += 1
    lower_value, upper_value = float(values[index]), float(values[tuple(upper)])
    expected = (
        normal_velocity * (lower_value + upper_value) / 2
        - abs(normal_velocity) * courant * (upper_value - lower_value) / 2
    )
    with tracing_context():
        actual = adv_flux_superbee_wgrid(state, values)[axis]
    np.testing.assert_allclose(actual[index], expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("scheme", ("upwind", "superbee"))
def test_advection_full_cell_limit_and_geometry_products(flow_basin, scheme):
    import jax
    import jax.numpy as jnp

    geometry, psi = flow_basin
    partial = advected_state(candidate(geometry, 1.0), psi, scheme)
    full = copy(partial)
    full._settings = copy(partial.settings)
    with full.settings.unlock():
        full.settings.enable_partial_cells = False
    for name in ("tke", "eke"):
        first, second = integrate(partial, name), integrate(full, name)
        for field in (name, "dtke" if name == "tke" else "deke"):
            np.testing.assert_allclose(getattr(first, field), getattr(second, field), rtol=1e-12, atol=1e-15)
    rng = np.random.default_rng(2026)
    fractions = jnp.asarray(rng.uniform(0.22, 0.28, size=(8, 10)))
    patterns = jnp.asarray(rng.uniform(-0.01, 0.01, size=(2, 8, 10)))

    def function(z):
        state = advected_state(candidate(geometry, fractions + jnp.einsum("i,ijk->jk", z, patterns)), psi, scheme)
        result = integrate(state, "tke")
        return result.tke[2:-2, 2:-2, :, state.variables.taup1]

    point, direction = jnp.asarray([0.1, -0.1]), jnp.asarray([0.3, -0.7])
    tangent = jax.jvp(function, (point,), (direction,))[1]
    seed = jnp.cos(jnp.arange(320)).reshape(tangent.shape)
    transpose = jax.vjp(function, point)[1](seed)[0]
    assert np.linalg.norm(tangent) > 1e-8
    np.testing.assert_allclose(jnp.vdot(seed, tangent), jnp.vdot(transpose, direction), rtol=1e-8, atol=1e-12)
    for epsilon in (1e-3, 1e-4):
        difference = (function(point + epsilon * direction) - function(point - epsilon * direction)) / (2 * epsilon)
        np.testing.assert_allclose(tangent, difference, rtol=2e-3, atol=1e-10)


def test_wgrid_cfl_diagnostic_matches_reconstruction(flow_basin):
    import jax.numpy as jnp
    from veros.core.advection import wgrid_cfl_spacings
    from veros.diagnostics.cfl_monitor import CFLMonitor

    geometry, psi = flow_basin
    state = advected_state(candidate(geometry, 0.3), psi, "superbee")
    vs = state.variables
    x, y, z = wgrid_cfl_spacings(state)
    values = CFLMonitor.finalize(None, state)
    expected_h = (
        jnp.maximum(
            jnp.max(abs(vs.u_wgrid[2:-2, 2:-2]) * vs.maskU[2:-2, 2:-2] / x[2:-2, 2:-2]),
            jnp.max(abs(vs.v_wgrid[2:-2, 2:-2] * vs.cosu[None, 2:-2, None]) * vs.maskV[2:-2, 2:-2] / y[2:-2, 2:-2]),
        )
        * state.settings.dt_tracer
    )
    expected_v = jnp.max(abs(vs.w_wgrid[2:-2, 2:-2]) * vs.maskW[2:-2, 2:-2] / z[2:-2, 2:-2]) * state.settings.dt_tracer
    np.testing.assert_allclose(values["cfl_wgrid"], expected_h, rtol=1e-13)
    np.testing.assert_allclose(values["wcfl_wgrid"], expected_v, rtol=1e-13)
    assert 0 < float(expected_h) < 1


@pytest.mark.parametrize("bad", (np.nan, np.inf))
def test_compiled_and_host_cfl_reject_nonfinite_energy_values(monkeypatch, bad):
    import jax
    import jax.numpy as jnp
    from bathymetry_helpers import make_simulation
    from veros.diagnostics.cfl_monitor import CFLMonitor
    from veros.routines import tracing_context

    def faulty(self, state):
        return dict(
            cfl=jnp.asarray(0.0), wcfl=jnp.asarray(0.0), cfl_wgrid=jnp.asarray(bad), wcfl_wgrid=jnp.asarray(0.0)
        )

    monkeypatch.setattr(CFLMonitor, "finalize", faulty)
    simulation = make_simulation(partial_cells=False, wind=0)
    simulation.state.diagnostics["cfl_monitor"].output_frequency = simulation.state.settings.dt_tracer
    prepared = simulation.prepare_integrator(record_diagnostics=("cfl_monitor",))
    initial = prepared.pack(simulation.state)

    def check(variables):
        with tracing_context():
            return prepared._update_diagnostics(prepared._make_state(variables), initial.diagnostics)[2]

    assert not bool(jax.jit(check)(initial.variables))
    with pytest.raises(RuntimeError, match="not finite"):
        CFLMonitor.write_finalized(None, simulation.state, faulty(None, simulation.state))
