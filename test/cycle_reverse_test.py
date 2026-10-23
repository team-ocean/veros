"""Periodic pullbacks preserve controls, discrete leaves, budgets, and failures."""

from collections import defaultdict

import numpy as np
import pytest


class Recurrence:
    def __init__(self, fail_at=-1):
        self.fail_at = fail_at

    def step(self, carry, *, parameters):
        import jax.numpy as jnp

        vs = carry.variables
        iteration = vs["iteration"] + 1
        x = 0.96 * vs["x"] + 0.03 * jnp.sin(vs["y"]) + parameters["forcing"] * jnp.cos(iteration)
        y = 0.85 * vs["y"] + 0.02 * vs["x"] ** 2 + parameters["gain"] * jnp.sin(iteration)
        failed = iteration == self.fail_at
        return (
            carry._replace(
                variables=dict(x=x, y=y, iteration=iteration),
                failure_code=jnp.where(failed, jnp.int32(1), carry.failure_code),
                failed_iteration=jnp.where(failed, iteration, carry.failed_iteration),
            ),
            {},
        )

    @staticmethod
    def validate_parameters(parameters):
        return parameters

    @staticmethod
    def check_configuration():
        pass

    @staticmethod
    def check_status(carry):
        from veros.integration import PreparedIntegrator

        PreparedIntegrator.check_status(carry)


def adapter_and_inputs(*, slots, fail_at=-1):
    import jax
    import jax.numpy as jnp
    from veros.checkpointing import CheckpointedIntegrator
    from veros.integration import IntegrationState
    from veros.ocean_response import OceanMap

    adapter = OceanMap.__new__(OceanMap)
    adapter.prepared = Recurrence(fail_at)
    adapter.ram_checkpoints = slots
    adapter.compiled_cycles = True
    adapter.reverse_block = None
    adapter.checkpoint_statistics = defaultdict(int)
    adapter.driver = CheckpointedIntegrator(adapter.prepared, terminal_objective=lambda carry: jnp.asarray(0.0))
    adapter.cycle_forward = jax.jit(
        lambda carry, parameters, steps: jax.lax.fori_loop(
            0, steps, lambda index, current: adapter.prepared.step(current, parameters=parameters)[0], carry
        )
    )
    initial = IntegrationState(
        dict(x=jnp.array([0.2, 0.3]), y=jnp.array([-0.1, 0.4]), iteration=jnp.int32(0)),
        {},
        jnp.int32(0),
        jnp.int32(-1),
    )
    parameters = dict(forcing=jnp.asarray(0.03), gain=jnp.asarray(0.05))
    return adapter, initial, parameters


@pytest.mark.parametrize(
    "length,slots,blocked",
    [(0, 16, False), (7, 16, False), (8, 10, True), (40, 14, True), (64, 13, False), (64, 14, True)],
)
def test_cycle_pullback_matches_direct_ad_and_preserves_checkpoint_budget(length, slots, blocked):
    import jax
    import jax.numpy as jnp

    adapter, initial, parameters = adapter_and_inputs(slots=slots)
    rng = np.random.default_rng(31)
    seed = jax.tree.map(
        lambda leaf: (
            jnp.asarray(rng.normal(size=leaf.shape), dtype=leaf.dtype)
            if jnp.issubdtype(leaf.dtype, jnp.floating)
            else np.zeros(leaf.shape, dtype=jax.dtypes.float0)
        ),
        initial,
    )

    def advance(carry, parameters):
        for _ in range(length):
            carry = adapter.prepared.step(carry, parameters=parameters)[0]
        return carry

    expected = jax.vjp(advance, initial, parameters)[1](seed)
    actual = adapter.reverse_segment(initial, parameters, length, seed)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        if left.dtype == jax.dtypes.float0:
            assert right.dtype == jax.dtypes.float0
        else:
            np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)
    statistics = adapter.checkpoint_statistics
    assert statistics["hierarchical_evaluations"] == int(blocked)
    assert statistics["peak_ram_checkpoints"] <= slots
    assert statistics["forward_steps"] == statistics["reverse_steps"] == length
    state_bytes = sum(leaf.nbytes for leaf in jax.tree.leaves(initial))
    assert statistics["peak_ram_bytes"] <= slots * state_bytes
    if blocked:
        assert statistics["recomputed_steps"] == 2 * length
        first = adapter.reverse_block
        signatures = first._cache_size()
        adapter.reverse_segment(initial, parameters, length, seed)
        assert adapter.reverse_block is first and first._cache_size() == signatures


@pytest.mark.parametrize("slots", (4, 16))
@pytest.mark.parametrize("fail_at", (5, 35))
def test_cycle_pullback_propagates_trajectory_failures(slots, fail_at):
    import jax
    import jax.numpy as jnp

    adapter, initial, parameters = adapter_and_inputs(slots=slots, fail_at=fail_at)
    seed = jax.tree.map(
        lambda leaf: (
            jnp.zeros_like(leaf)
            if jnp.issubdtype(leaf.dtype, jnp.floating)
            else np.zeros(leaf.shape, dtype=jax.dtypes.float0)
        ),
        initial,
    )
    with pytest.raises(RuntimeError, match=f"{fail_at}"):
        adapter.reverse_segment(initial, parameters, 40, seed)


def test_non_cpu_cycles_keep_host_checkpoint_storage():
    import jax
    import jax.numpy as jnp
    from veros import runtime_settings

    adapter, initial, parameters = adapter_and_inputs(slots=16)
    seed = jax.tree.map(
        lambda leaf: (
            jnp.zeros_like(leaf)
            if jnp.issubdtype(leaf.dtype, jnp.floating)
            else np.zeros(leaf.shape, dtype=jax.dtypes.float0)
        ),
        initial,
    )
    device = runtime_settings.device
    try:
        object.__setattr__(runtime_settings, "device", "gpu")
        adapter.reverse_segment(initial, parameters, 8, seed)
    finally:
        object.__setattr__(runtime_settings, "device", device)
    assert adapter.reverse_block is None
    assert adapter.checkpoint_statistics["hierarchical_evaluations"] == 0


def test_zero_length_pullback_does_not_trace_an_unused_timestep(monkeypatch):
    import jax
    import jax.numpy as jnp

    adapter, initial, parameters = adapter_and_inputs(slots=4)
    seed = jax.tree.map(
        lambda leaf: (
            jnp.ones_like(leaf)
            if jnp.issubdtype(leaf.dtype, jnp.floating)
            else np.zeros(leaf.shape, dtype=jax.dtypes.float0)
        ),
        initial,
    )

    def unused(*args, **kwargs):
        raise AssertionError("A zero-length segment must not evaluate or differentiate a timestep")

    monkeypatch.setattr(adapter.prepared, "step", unused)
    gradient, parameter_gradient = adapter.reverse_segment(initial, parameters, 0, seed)
    for actual, expected in zip(jax.tree.leaves(gradient), jax.tree.leaves(seed)):
        np.testing.assert_array_equal(actual, expected)
    for leaf in jax.tree.leaves(parameter_gradient):
        np.testing.assert_array_equal(leaf, 0)
    assert adapter.checkpoint_statistics["forward_steps"] == adapter.checkpoint_statistics["reverse_steps"] == 0
