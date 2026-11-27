"""Kernel-level partial-cell TKE and EKE energy budgets."""

from copy import copy

import numpy as np
import pytest

from wgrid_geometry_test import basin as basin, candidate  # noqa: F401

pytestmark = pytest.mark.usefixtures("jax_backend")


def energy_state(base):
    """Allocate native energy fields without declaring full-setup compatibility."""
    import jax.numpy as jnp
    from veros.state import get_default_state
    from veros.core.utilities import enforce_boundaries
    from veros.routines import tracing_context

    state = get_default_state()
    state._settings = copy(base.settings)
    with state.settings.unlock():
        state.settings.enable_tke = state.settings.enable_eke = True
        state.settings.enable_conserve_energy = True
        state.settings.enable_tke_hor_diffusion = True
        state.settings.K_h_tke = 20
        state.settings.alpha_tke = 30
        state.settings.alpha_eke = 20
    state.initialize_variables()
    vs = state.variables
    rng = np.random.default_rng(1925)
    with vs.unlock():
        for name, value in base.variables.items():
            if name in vs:
                setattr(vs, name, value)
        for name, scale in (("tke", 1e-3), ("eke", 0.02)):
            values = scale * jnp.asarray(rng.uniform(0.8, 1.2, size=vs.maskW.shape)) * vs.maskW
            with tracing_context():
                values = enforce_boundaries(values, state.settings.enable_cyclic_x)
            setattr(vs, name, jnp.broadcast_to(values[..., None], getattr(vs, name).shape))
        vs.sqrttke = jnp.sqrt(vs.tke[..., vs.tau])
        vs.sqrteke = jnp.sqrt(vs.eke[..., vs.tau])
        vs.mxl = jnp.ones_like(vs.mxl) * 50
        vs.eke_len = jnp.ones_like(vs.eke_len) * 1000
        vs.kappaM = jnp.ones_like(vs.kappaM) * 1e-3
        for name, scale in (
            ("K_diss_v", 2e-7),
            ("K_diss_h", 3e-7),
            ("K_diss_gm", 1e-7),
            ("K_diss_bot", 4e-8),
            ("P_diss_v", 1e-8),
            ("P_diss_adv", 2e-8),
            ("P_diss_nonlin", 1e-9),
            ("P_diss_skew", 1e-9),
            ("P_diss_hmix", 2e-9),
            ("P_diss_iso", 3e-9),
        ):
            setattr(vs, name, scale * vs.maskW)
        vs.forc_tke_surface = jnp.ones_like(vs.forc_tke_surface) * 1e-5
    return state


def advance_energies(state):
    from veros.core import tke, eke
    from veros.routines import tracing_context

    current = copy(state)
    current._variables = copy(state.variables)
    with current.variables.unlock(), tracing_context():
        current.variables.update(eke.integrate_eke_kernel(current))
        current.variables.update(tke.integrate_tke_kernel(current))
    return current


@pytest.mark.parametrize("surface_flux", (1e-5, -1e-3))
def test_native_energy_sources_sinks_and_mixing_close_in_occupied_volumes(basin, surface_flux):  # noqa: F811
    import jax.numpy as jnp
    from veros.core.geometry import wgrid_thickness
    from veros.diagnostics.energy import diagnose_kernel
    from veros.routines import tracing_context

    initial = energy_state(candidate(basin, 0.3))
    with initial.variables.unlock():
        initial.variables.forc_tke_surface = jnp.ones_like(initial.variables.forc_tke_surface) * surface_flux
    current = advance_energies(initial)
    vs, old = current.variables, initial.variables
    area = vs.area_t[2:-2, 2:-2]
    occupied = wgrid_thickness(current)[2:-2, 2:-2]
    volume = area[..., None] * occupied
    nominal = jnp.broadcast_to(vs.dzw, vs.maskW.shape).at[..., -1].multiply(0.5)
    work_volume = area[..., None] * nominal[2:-2, 2:-2] * vs.maskW[2:-2, 2:-2]

    def work(field):
        return jnp.sum(work_volume * field[2:-2, 2:-2])

    eke_rate = (
        jnp.sum(volume * (vs.eke[2:-2, 2:-2, :, vs.taup1] - old.eke[2:-2, 2:-2, :, old.tau]))
        / current.settings.dt_tracer
    )
    eke_source = work(vs.K_diss_gm + vs.K_diss_h - vs.P_diss_skew - vs.P_diss_hmix - vs.P_diss_iso)
    np.testing.assert_allclose(eke_rate, eke_source - work(vs.eke_diss_iw), rtol=1e-10, atol=1e-7)
    tke_rate = (
        jnp.sum(volume * (vs.tke[2:-2, 2:-2, :, vs.taup1] - old.tke[2:-2, 2:-2, :, old.tau])) / current.settings.dt_mom
    )
    tke_source = work(
        vs.K_diss_v - vs.P_diss_v - vs.P_diss_adv - vs.P_diss_nonlin + vs.eke_diss_iw + vs.eke_diss_tke + vs.K_diss_bot
    )
    surface = jnp.sum(
        area * vs.maskW[2:-2, 2:-2, -1] * (vs.forc_tke_surface[2:-2, 2:-2] + vs.tke_surf_corr[2:-2, 2:-2])
    )
    np.testing.assert_allclose(tke_rate, tke_source + surface - work(vs.tke_diss), rtol=1e-10, atol=1e-7)
    if surface_flux >= 0:
        assert np.min(vs.tke[2:-2, 2:-2, :, vs.taup1]) >= 0
    if surface_flux < 0:
        assert np.max(vs.tke_surf_corr) > 0
    with tracing_context():
        record = diagnose_kernel(current)
    np.testing.assert_allclose(record.tke_m, jnp.sum(volume * vs.tke[2:-2, 2:-2, :, vs.tau]), rtol=1e-12)
    np.testing.assert_allclose(record.eke_m, jnp.sum(volume * vs.eke[2:-2, 2:-2, :, vs.tau]), rtol=1e-12)
    np.testing.assert_allclose(record.dtke_m, tke_rate, rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(record.deke_m, eke_rate, rtol=1e-12, atol=1e-8)


def test_native_energy_kernel_geometry_products(basin):  # noqa: F811
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(1926)
    fraction = jnp.asarray(rng.uniform(0.22, 0.28, size=(8, 10)))
    patterns = jnp.asarray(rng.uniform(-0.01, 0.01, size=(2, 8, 10)))

    def function(z):
        base = candidate(basin, fraction + jnp.einsum("i,ijk->jk", z, patterns))
        current = advance_energies(energy_state(base))
        vs = current.variables
        return jnp.concatenate((vs.tke[2:-2, 2:-2, :, vs.taup1].ravel(), vs.eke[2:-2, 2:-2, :, vs.taup1].ravel()))

    point = jnp.asarray([0.1, -0.1])
    direction = jnp.asarray([0.3, -0.7])
    seed = jnp.cos(jnp.arange(640))
    tangent = jax.jvp(function, (point,), (direction,))[1]
    transpose = jax.vjp(function, point)[1](seed)[0]
    assert np.linalg.norm(tangent) > 1e-7
    np.testing.assert_allclose(jnp.vdot(seed, tangent), jnp.vdot(transpose, direction), rtol=1e-8, atol=1e-12)
    for epsilon in (1e-3, 1e-4):
        difference = (function(point + epsilon * direction) - function(point - epsilon * direction)) / (2 * epsilon)
        np.testing.assert_allclose(tangent, difference, rtol=2e-3, atol=1e-10)


@pytest.mark.parametrize("surface_flux", (1e-5, -1e-3))
def test_energy_full_cell_limit_retains_native_kernel_values(basin, surface_flux):  # noqa: F811
    import jax.numpy as jnp

    partial = energy_state(candidate(basin, 1.0))
    with partial.variables.unlock():
        partial.variables.forc_tke_surface = jnp.full_like(partial.variables.forc_tke_surface, surface_flux)
    full = copy(partial)
    full._settings = copy(partial.settings)
    with full.settings.unlock():
        full.settings.enable_partial_cells = False
    first, second = advance_energies(partial), advance_energies(full)
    for name in ("tke", "eke", "tke_diss", "eke_diss_iw", "tke_surf_corr"):
        np.testing.assert_allclose(
            getattr(first.variables, name), getattr(second.variables, name), rtol=1e-12, atol=1e-15
        )
