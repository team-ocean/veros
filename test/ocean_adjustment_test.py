"""Native adjusted ocean products and changed starting-state validation."""

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("jax_backend")


def test_adjusted_ocean_rejects_distributed_state_before_constructing_a_map(monkeypatch):
    from types import SimpleNamespace
    import veros
    from veros.integration import CompiledCompatibilityError
    from veros.ocean_response import response

    monkeypatch.setattr(veros, "_runtime_state", SimpleNamespace(proc_num=2))
    with pytest.raises(CompiledCompatibilityError, match="serial"):
        response(None, None, parameters={}, observe=lambda c: 0, tolerance=1e-7, max_iterations=1)


@pytest.mark.parametrize("energy", (False, True))
def test_enthalpy_advection_history_has_prognostic_coordinates(energy):
    import jax.numpy as jnp
    from types import SimpleNamespace
    from veros.state import get_default_state
    from veros.integration import IntegrationState, PreparedIntegrator
    from veros.ocean_response import PrognosticCoordinates

    state = get_default_state()
    with state.settings.unlock():
        state.settings.update(
            nx=2,
            ny=3,
            nz=2,
            dt_mom=3600.0,
            dt_tracer=3600.0,
            enable_streamfunction=False,
            enable_conserve_energy=energy,
            enable_tke=energy,
        )
    state.initialize_variables()
    with state.variables.unlock():
        vs = state.variables
        for name in ("maskT", "maskU", "maskV", "maskW"):
            setattr(vs, name, jnp.ones_like(getattr(vs, name)))
        vs.area_t = jnp.ones_like(vs.area_t)
        vs.dzt = jnp.ones_like(vs.dzt)
        vs.dHd = vs.dHd.at[2, 2, 0, 1].set(0.3)
        if energy:
            vs.K_diss_v = jnp.full_like(vs.K_diss_v, 1e-3 / state.settings.dt_tracer)
    prepared = SimpleNamespace(_template=state, replace_variables=PreparedIntegrator.replace_variables)
    carry = IntegrationState(vs, {}, jnp.int32(0), jnp.int32(-1))
    coordinates = PrognosticCoordinates(prepared, carry)
    assert ("dHd" in {name for name, _, _ in coordinates.fields}) == energy
    assert "dHd" not in coordinates.algebraic_fields
    if energy:
        assert "K_diss_v" in {name for name, _, _ in coordinates.fields}
        vector = coordinates.pack(carry)
        offset = 0
        for name, indices, _ in coordinates.fields:
            if name == "K_diss_v":
                np.testing.assert_allclose(vector[offset : offset + indices.size], 1)
            offset += indices.size
        restored = coordinates.unpack(vector, carry)
        np.testing.assert_allclose(restored.variables.dHd[2, 2, 0, 1], 0.3)


def test_native_equilibrium_observation_products_and_initial_state_independence(tmp_path):
    import jax.numpy as jnp
    from benchmarks.ad_demos.common import Checks, gradient_checks
    from benchmarks.ad_demos.matrix import CASES
    from benchmarks.ad_demos.research_ocean import ResearchOcean
    from benchmarks.ad_demos.responses import Response
    from benchmarks.ad_demos.__main__ import write_result

    config = dict(
        setup="adjusted_basin", physics="fixed", mode="study", steps=None, spinup_steps=0, neutral_policy="nonstable"
    )
    ocean = ResearchOcean(config, CASES["sensitivity_equilibrium"])
    response = Response(ocean, (2, 4, 6), config)
    point, checks = jnp.zeros(3), Checks()
    checks.on_update = lambda: write_result(
        tmp_path / "native-equilibrium-checks.json", dict(checks=checks.items, metrics=checks.metrics)
    )
    response.validate(point, checks)
    function = response.function(lambda values, z: values[ocean.heat_index] + values[ocean.amoc_index])
    gradient_checks(function, point, checks)
    write_result(tmp_path / "native-equilibrium-checks.json", dict(checks=checks.items, metrics=checks.metrics))
    assert all(check["passed"] for check in checks.items), [check for check in checks.items if not check["passed"]]
    assert checks.metrics["adjustment"]["pressure_gauge"] == "fixed initial wet-area mean sea level"
    assert "rho" not in checks.metrics["adjustment"]["state_fields"]
    assert "rho" in checks.metrics["adjustment"]["algebraic_fields"]
    assert "dHd" in checks.metrics["adjustment"]["state_fields"]
    assert "dHd" not in checks.metrics["adjustment"]["algebraic_fields"]
    runs = list(ocean.prepared._adjusted_responses.values())
    assert len(runs) == 2 and runs[0][0] is runs[1][0] and runs[0][1] is not runs[1][1]
    assert np.isfinite(function.value_and_grad(point)[1]).all()


@pytest.mark.parametrize("period_steps", (1, 7, 32))
def test_native_cycle_checkpoint_budget_preserves_transpose(period_steps):
    import jax.numpy as jnp
    from benchmarks.ad_demos.matrix import CASES
    from benchmarks.ad_demos.research_ocean import ResearchOcean
    from veros.ocean_response import OceanMap, canonical

    config = dict(
        setup="adjusted_basin", physics="fixed", mode="study", steps=None, spinup_steps=0, neutral_policy="nonstable"
    )
    ocean = ResearchOcean(config, CASES["sensitivity_equilibrium"])
    carry, parameters = ocean.controls(jnp.zeros(16))
    adapter = OceanMap(ocean.prepared, parameters, ocean.observe, period_steps=period_steps, ram_checkpoints=64)
    inputs = adapter.inputs.pack((canonical(ocean.prepared, carry), parameters))
    state = adapter.initial(inputs)
    direction = 1e-3 * np.sin(np.arange(state.size) + 1)
    seed = np.cos(np.arange(state.size) + 1)
    product = adapter.product(state, inputs, direction, np.zeros_like(inputs))
    high_state, high_inputs = adapter.transpose(state, inputs, seed)
    assert adapter.checkpoint_statistics["hierarchical_evaluations"] == int(period_steps == 32)
    assert adapter.checkpoint_statistics["peak_ram_checkpoints"] <= 64
    np.testing.assert_allclose(seed @ product, high_state @ direction, rtol=1e-7, atol=1e-8)
    epsilon = 1e-3
    difference = (
        adapter.mapping(state + epsilon * direction, inputs) - adapter.mapping(state - epsilon * direction, inputs)
    ) / (2 * epsilon)
    np.testing.assert_allclose(seed @ difference, seed @ product, rtol=2e-3, atol=1e-8)
    previous_replays = adapter.checkpoint_statistics["recomputed_steps"]
    adapter.ram_checkpoints = 4
    low_state, low_inputs = adapter.transpose(state, inputs, seed)
    np.testing.assert_allclose(low_state, high_state, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(low_inputs, high_inputs, rtol=1e-12, atol=1e-12)
    if period_steps > 1:
        assert adapter.checkpoint_statistics["recomputed_steps"] > previous_replays
    else:
        assert adapter.equilibrium_product is not None
        assert adapter.equilibrium_transpose is not None
        assert adapter.equilibrium_transpose_calls == 2
        compiled = adapter.equilibrium_transpose
        adapter.equilibrium_transpose = None
        ordinary_state, ordinary_inputs = adapter.transpose(state, inputs, seed)
        adapter.equilibrium_transpose = compiled
        np.testing.assert_allclose(high_state, ordinary_state, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(high_inputs, ordinary_inputs, rtol=1e-12, atol=1e-12)
        for transpose, vector in ((False, direction), (True, seed)):
            current, total = np.asarray(vector).copy(), np.zeros_like(vector)
            for _ in range(16):
                total += current
                current = (
                    np.asarray(adapter.transpose(state, inputs, current)[0])
                    if transpose
                    else np.asarray(adapter.product(state, inputs, current, np.zeros_like(inputs)))
                )
            power, summation = adapter.polynomial_product(state, inputs, vector, transpose=transpose)
            np.testing.assert_allclose(power, current, rtol=1e-10, atol=1e-8)
            np.testing.assert_allclose(summation, total, rtol=1e-10, atol=1e-8)
        offset = 0
        for name, indices, _ in adapter.state.fields:
            if name == "temp":
                break
            offset += indices.size
        invalid = np.asarray(state).copy()
        invalid[offset] = np.nan
        with pytest.raises(RuntimeError):
            adapter.product(invalid, inputs, direction, np.zeros_like(inputs))
        with pytest.raises(RuntimeError):
            adapter.transpose(invalid, inputs, seed)
        with pytest.raises(RuntimeError):
            adapter.polynomial_product(invalid, inputs, direction)
