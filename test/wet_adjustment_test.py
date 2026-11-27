"""Wet topology and connected-basin pressure constraints in adjusted coordinates."""

import numpy as np
import pytest

from wet_geometry_test import channel_basin

pytestmark = pytest.mark.usefixtures("jax_backend")


@pytest.mark.parametrize("host_inputs", (False, True))
def test_adjusted_coordinates_retain_new_wet_topology_and_prognostic_fields(host_inputs):
    import jax.numpy as jnp
    from veros.ocean_response import OceanMap, canonical

    sim, prepared, initial, channel = channel_basin()
    opened = prepared.with_wet_geometry(initial, opening_m=75.0, channel=channel)
    opened = canonical(prepared, opened)
    adapter = OceanMap(prepared, {}, lambda c: c.variables.temp[6, 7, -1, c.variables.tau])
    inputs = adapter.inputs.pack((opened, {}))
    if host_inputs:
        inputs = np.asarray(inputs)
    recovered, _ = adapter.inputs.unpack(inputs)
    active = adapter.active_state_mask(inputs)
    closed_active = adapter.active_state_mask(adapter.inputs.pack((canonical(prepared, initial), {})))
    assert active.dtype == bool and active.shape == (adapter.state.size,)
    assert np.count_nonzero(active) > np.count_nonzero(closed_active)
    for name in ("kbot", "maskT", "maskU", "maskV", "maskW", "maskZ"):
        np.testing.assert_array_equal(getattr(recovered.variables, name), getattr(opened.variables, name), err_msg=name)
    coordinates = adapter.initial(inputs)
    reconstructed, _ = adapter.construct(coordinates, inputs)
    np.testing.assert_allclose(reconstructed.variables.temp, opened.variables.temp, atol=1e-12)
    expected = adapter.normalize(prepared.step(opened)[0], opened)
    np.testing.assert_allclose(adapter.mapping(coordinates, inputs), adapter.state.pack(expected), atol=1e-12)
    # Dry allocated coordinates are constrained to zero rather than neutral directions.
    vector = coordinates + jnp.ones_like(coordinates)
    constrained = adapter.state.pack(adapter.state.unpack(vector, opened))
    assert int(jnp.sum(constrained == 0)) > 0
    dry_product = adapter.product(coordinates, inputs, (~active).astype(float), np.zeros_like(inputs))
    np.testing.assert_array_equal(dry_product, 0)


def test_pressure_constraints_preserve_each_connected_basin_mean():
    import jax
    import jax.numpy as jnp
    from veros.ocean_response import PrognosticCoordinates

    sim, prepared, initial, channel = channel_basin()
    mask = initial.variables.maskT[..., -1]
    left = jnp.zeros_like(mask).at[2:6, 2:-2].set(True)
    right = jnp.zeros_like(mask).at[7:-2, 2:-2].set(True)
    reference = prepared.replace_variables(
        initial,
        psi=jnp.broadcast_to(
            (2 * left - right).astype(initial.variables.psi.dtype)[..., None], initial.variables.psi.shape
        ),
    )
    candidate = prepared.replace_variables(reference, psi=reference.variables.psi + (4 * left - 3 * right)[..., None])
    coordinates = PrognosticCoordinates(prepared, initial)
    result = coordinates.constrain(candidate, reference)
    for region, expected in ((left, 2.0), (right, -1.0)):
        mean = jnp.sum(result.variables.psi * region[..., None], axis=(0, 1)) / jnp.sum(region)
        np.testing.assert_allclose(mean, expected, atol=1e-12)

    def function(psi):
        constrained = coordinates.constrain(prepared.replace_variables(candidate, psi=psi), reference)
        return jnp.sum(constrained.variables.psi**2)

    point = candidate.variables.psi
    tangent = jnp.sin(jnp.arange(point.size)).reshape(point.shape)
    forward = jax.jvp(jax.jit(function), (point,), (tangent,))[1]
    reverse = jnp.vdot(jax.grad(function)(point), tangent)
    np.testing.assert_allclose(forward, reverse, rtol=1e-10, atol=1e-10)


def test_connected_components_follow_cyclic_faces_and_open_passages():
    from veros.core.geometry import connected_components

    for cyclic, expected in ((False, 2), (True, 1)):
        sim, prepared, initial, channel = channel_basin(cyclic=cyclic)
        labels = np.asarray(connected_components(prepared._make_state(initial.variables)))
        wet = np.asarray(initial.variables.maskT[2:-2, 2:-2, -1])
        assert len(np.unique(labels[wet])) == expected
        opened = prepared.with_wet_geometry(initial, opening_m=75.0, channel=channel)
        labels = np.asarray(connected_components(prepared._make_state(opened.variables)))
        wet = np.asarray(opened.variables.maskT[2:-2, 2:-2, -1])
        assert len(np.unique(labels[wet])) == 1
