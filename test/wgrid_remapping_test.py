"""Energy inventories and lagged work use their respective W-grid measures."""

from copy import copy

import numpy as np
import pytest

from wgrid_geometry_test import basin as basin, candidate, reference_width  # noqa: F401
from wgrid_energy_test import energy_state

pytestmark = pytest.mark.usefixtures("jax_backend")


@pytest.fixture(scope="module")
def energy_remap(basin):  # noqa: F811
    """Remap allocated turbulence fields and their tendency histories."""
    import jax.numpy as jnp
    from veros.core.utilities import enforce_boundaries
    from veros.routines import tracing_context

    prepared = copy(basin[0])
    state = energy_state(candidate(basin, 0.3))
    prepared._template = copy(prepared._template)
    prepared._template._settings = copy(state.settings)
    prepared._template._variables = copy(state.variables)
    rng = np.random.default_rng(1951)
    with state.variables.unlock():
        for name, scale in (("tke", 1e-3), ("eke", 0.02), ("dtke", 1e-7), ("deke", 2e-7)):
            field = getattr(state.variables, name)
            values = rng.uniform(0.5, 1.5, size=field.shape) if name in ("tke", "eke") else rng.normal(size=field.shape)
            with tracing_context():
                values = enforce_boundaries(
                    jnp.asarray(scale * values) * state.variables.maskW[..., None], state.settings.enable_cyclic_x
                )
            setattr(state.variables, name, values)
    return prepared, prepared.pack(state)


def work_width(state):
    vs = state.variables
    width = np.broadcast_to(np.asarray(vs.dzw), vs.maskW.shape).copy()
    width[..., -1] *= 0.5
    return width * np.asarray(vs.maskW)


def inventory(prepared, carry, name, *, global_sum=False):
    state = prepared._make_state(carry.variables)
    width = work_width(state) if name == "K_diss_v" else reference_width(state)
    volume = width[2:-2, 2:-2] * np.asarray(carry.variables.area_t[2:-2, 2:-2, None])
    values = np.asarray(getattr(carry.variables, name)[2:-2, 2:-2])
    if values.ndim == 4:
        volume = volume[..., None]
    return np.sum(volume * values, axis=(0, 1, 2) if global_sum else 2)


def changed(prepared, carry, kind, z):
    import jax.numpy as jnp
    from veros.core.geometry import bathymetry_geometry
    from veros.core.numerics import calc_topo_kernel
    from veros.remapping import remap_carry, vertical_grid
    from veros.routines import tracing_context

    vs = carry.variables
    if kind == "levels":
        return vertical_grid(
            prepared, carry, vs.dzt + z * jnp.array([2.0, -1.0, -3.0, 2.0]), conserve=("temp", "salt", "momentum")
        )
    if kind == "bed":
        pattern = 1 + jnp.sin(jnp.arange(vs.ht.shape[0])[:, None] + jnp.arange(vs.ht.shape[1])[None, :])
        depth = vs.ht + z * pattern * (vs.kbot > 0)
        with tracing_context():
            geometry = bathymetry_geometry(prepared._make_state(vs), depth)
        target = prepared.replace_variables(carry, **geometry._asdict())
        return remap_carry(prepared, carry, target, stretch=True)
    if kind == "opening":
        bottom = vs.kbot.at[5, 6].set(vs.maskW.shape[-1])
        depth = vs.ht.at[5, 6].set(60 + z)
    elif kind == "removal":
        bottom = vs.kbot.at[4, 5].set(0)
        depth = vs.ht.at[4, 5].set(0)
    else:
        raise ValueError(kind)
    target = prepared.replace_variables(carry, kbot=bottom)
    with tracing_context():
        geometry = calc_topo_kernel(prepared._make_state(target.variables))
        target = prepared.replace_variables(target, **geometry._asdict())
        geometry = bathymetry_geometry(prepared._make_state(target.variables), depth)
    target = prepared.replace_variables(target, **geometry._asdict())
    return remap_carry(prepared, carry, target, redistribute=True)


@pytest.mark.parametrize("kind", ("levels", "bed", "opening", "removal"))
def test_energy_and_all_history_inventories_survive_geometry_changes(energy_remap, kind):
    import jax
    import jax.numpy as jnp

    prepared, initial = energy_remap
    updated = jax.jit(lambda z: changed(prepared, initial, kind, z))(jnp.asarray(0.2))
    prepared.check_status(updated)
    for name in ("tke", "eke", "dtke", "deke", "K_diss_v"):
        np.testing.assert_allclose(
            inventory(prepared, updated, name, global_sum=kind in ("opening", "removal")),
            inventory(prepared, initial, name, global_sum=kind in ("opening", "removal")),
            rtol=2e-12,
            atol=1e-7,
        )
        field = np.asarray(getattr(updated.variables, name))
        dry = ~np.asarray(updated.variables.maskW).astype(bool)
        np.testing.assert_array_equal(field[dry], 0)
        if name in ("tke", "eke"):
            assert np.min(field) >= 0
    if kind == "opening":
        assert int(updated.variables.kbot[5, 6]) > 0
        state = prepared._make_state(updated.variables)
        for name in ("tke", "eke", "dtke", "deke", "K_diss_v"):
            width = work_width(state) if name == "K_diss_v" else reference_width(state)
            volume = width[2:-2, 2:-2] * np.asarray(updated.variables.area_t[2:-2, 2:-2, None])
            expected = inventory(prepared, initial, name, global_sum=True) / np.sum(volume)
            np.testing.assert_allclose(getattr(updated.variables, name)[5, 6, -1], expected, rtol=1e-12, atol=1e-14)


def test_constant_energies_dilute_with_changed_covered_w_volume(energy_remap):
    import jax.numpy as jnp
    from veros.core.geometry import bathymetry_geometry
    from veros.remapping import vertical_grid
    from veros.routines import tracing_context

    prepared, initial = energy_remap
    with tracing_context():
        geometry = bathymetry_geometry(
            prepared._make_state(initial.variables), jnp.sum(initial.variables.dzt * initial.variables.maskT, axis=2)
        )
    initial = prepared.replace_variables(initial, **geometry._asdict())
    values = initial.variables.maskW[..., None] * jnp.arange(1.0, 4.0)
    original = prepared.replace_variables(initial, tke=values, eke=values)
    updated = vertical_grid(
        prepared,
        original,
        original.variables.dzt + 0.2 * jnp.array([-2.0, -1.0, 1.0, 2.0]),
        conserve=("temp", "salt", "momentum"),
    )
    prepared.check_status(updated)
    old = reference_width(prepared._make_state(original.variables))
    new = reference_width(prepared._make_state(updated.variables))
    old_depth, new_depth = np.sum(old, axis=2), np.sum(new, axis=2)
    ratio = old_depth / np.where(new_depth > 0, new_depth, 1)
    assert np.max(abs(old_depth - new_depth)) > 0.1
    expected = ratio[..., None, None] * np.asarray(updated.variables.maskW[..., None]) * np.arange(1.0, 4.0)
    np.testing.assert_allclose(updated.variables.tke[2:-2, 2:-2], expected[2:-2, 2:-2], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("kind", ("levels", "bed", "opening"))
def test_energy_remap_geometry_jvp_vjp_and_finite_differences(energy_remap, kind):
    import jax
    import jax.numpy as jnp

    prepared, initial = energy_remap

    def function(z):
        result = changed(prepared, initial, kind, z)
        return jnp.concatenate(
            [
                getattr(result.variables, name)[2:-2, 2:-2].ravel()
                for name in ("tke", "eke", "dtke", "deke", "K_diss_v", "kappaH", "K_gm")
            ]
        )

    point = jnp.asarray(0.2)
    direction = jnp.asarray(0.7)
    tangent = jax.jvp(function, (point,), (direction,))[1]
    seed = jnp.cos(jnp.arange(tangent.size))
    transpose = jax.vjp(function, point)[1](seed)[0]
    assert np.linalg.norm(tangent) > 1e-6
    np.testing.assert_allclose(jnp.vdot(seed, tangent), transpose * direction, rtol=1e-9, atol=1e-10)
    for epsilon in (1e-3, 1e-4):
        difference = (function(point + epsilon * direction) - function(point - epsilon * direction)) / (2 * epsilon)
        np.testing.assert_allclose(difference, tangent, rtol=2e-3, atol=1e-8)


def test_energy_remap_identity_and_rebuilt_diffusivities(energy_remap):
    from veros.core import eke, tke
    from veros.remapping import remap_carry
    from veros.routines import tracing_context

    prepared, initial = energy_remap
    updated = remap_carry(prepared, initial, initial)
    for name in ("tke", "eke", "dtke", "deke", "K_diss_v"):
        np.testing.assert_allclose(
            getattr(updated.variables, name), getattr(initial.variables, name), rtol=1e-12, atol=1e-15
        )
    state = prepared._make_state(updated.variables)
    with tracing_context():
        first = eke.set_eke_diffusivities_kernel(state)
        second = tke.set_tke_diffusivities_kernel(state)
    for result in (first, second):
        for name in result._fields:
            np.testing.assert_array_equal(getattr(updated.variables, name), getattr(result, name))


def test_idemix_energy_remap_remains_guarded(energy_remap):
    from veros.remapping import remap_carry

    prepared, initial = energy_remap
    blocked = copy(prepared)
    blocked._template = copy(prepared._template)
    blocked._template._settings = copy(prepared._template.settings)
    with blocked._template.settings.unlock():
        blocked._template.settings.enable_idemix = True
    with pytest.raises(ValueError, match="IDEMIX"):
        remap_carry(blocked, initial, initial)
