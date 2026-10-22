"""Finite response products and inverse positive controls."""

import numpy as np
import pytest

from benchmarks.ad_demos.finite_response import ObservationFunction
from benchmarks.ad_demos.inverse import (
    JaxResidual,
    least_squares,
    multistart_least_squares,
    constrained_minimum,
    observation_covariance,
)
from benchmarks.ad_demos.matrix import CASES
from benchmarks.ad_demos.responses import Response


def test_surface_redistribution_balances_applied_flux_and_has_two_sided_products():
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.research_ocean import balanced_surface_flux

    jax.config.update("jax_enable_x64", True)
    area = 1 + jnp.arange(64.0).reshape(8, 8)
    eligible = jnp.ones((8, 8), bool).at[2:4, 2:4].set(False)
    pattern = jnp.sin(jnp.arange(64.0)).reshape(8, 8)

    def function(z):
        return balanced_surface_flux(z[0] * pattern, eligible, area)

    point, direction = jnp.zeros(1), jnp.ones(1)
    _, tangent = jax.jvp(function, (point,), (direction,))
    for amplitude in (-0.1, 0.1):
        flux = function(jnp.array([amplitude]))
        np.testing.assert_allclose(jnp.sum((flux * area)[2:-2, 2:-2]), 0, atol=1e-12)
        np.testing.assert_array_equal(flux[~eligible], 0)
        np.testing.assert_allclose(flux / amplitude, tangent, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(balanced_surface_flux(pattern, jnp.zeros_like(eligible), area), 0)


class Recurrence:
    dt = 86400.0

    def __init__(self, kind):
        import jax
        import jax.numpy as jnp
        from veros.integration import IntegrationState

        jax.config.update("jax_enable_x64", True)
        self.case = CASES[f"estimation_{kind}"]
        self.initial = IntegrationState(dict(x=jnp.array(0.2), forcing=jnp.array(0.0)), {}, jnp.int32(0), jnp.int32(-1))
        self.prepared = self
        self.traces = 0

    def controls(self, full, *, phase_days=0, **kwargs):
        return self.initial._replace(variables=dict(x=0.2 + full[0], forcing=full[2] + 0.01 * phase_days)), {
            "rate": 0.8 * np.exp(0.0) + full[8] * 0.1
        }

    @staticmethod
    def observe(carry):
        import jax.numpy as jnp

        return jnp.stack((carry.variables["x"], carry.variables["x"] ** 2))

    @staticmethod
    def validate_parameters(parameters):
        return {} if parameters is None else parameters

    def step(self, carry, *, parameters=None):
        import jax

        if isinstance(carry.variables["x"], jax.core.Tracer):
            self.traces += 1
        x = parameters["rate"] * carry.variables["x"] + carry.variables["forcing"]
        return carry._replace(variables=dict(carry.variables, x=x)), {}

    def advance(self, carry, *, n_steps, parameters=None, **kwargs):
        for _ in range(n_steps):
            carry, _ = self.step(carry, parameters=parameters)
        return carry, {}

    @staticmethod
    def check_status(carry):
        pass


def test_realized_local_response_rejects_failed_carry_before_observing():
    import jax.numpy as jnp
    from types import SimpleNamespace
    from benchmarks.ad_demos.research_ocean import ResearchOcean
    from veros.integration import IntegrationState, PreparedIntegrator

    failed = IntegrationState({}, {}, jnp.int32(6), jnp.int32(0))
    ocean = object.__new__(ResearchOcean)
    ocean.case, ocean.dt = CASES["geometry_local"], 86400.0
    ocean.initial = failed
    ocean.observe = lambda carry: jnp.zeros(1)
    ocean.controls = lambda *args, **kwargs: (failed, {})
    ocean.prepared = SimpleNamespace(
        advance=lambda carry, **kwargs: (carry, {}), check_status=PreparedIntegrator.check_status
    )
    response = Response(ocean, (12, 13, 14, 15), dict(mode="study", steps=1))
    with pytest.raises(RuntimeError, match="wet geometry"):
        response.values(jnp.zeros(4), realized=True)


def test_adjusted_validation_audits_stability_once_and_checks_changed_state_products():
    import jax.numpy as jnp
    from benchmarks.ad_demos.common import Checks

    response = Response.__new__(Response)
    response.kind, response.config = "equilibrium", dict(mode="study")
    response.steps, response.windows = 1, 16
    audits, starting_states = [], []

    def adjusted(point, *, initial_shift=0, doubled=False, diagnostics=True):
        starting_states.append(initial_shift)
        result = dict(observations=jnp.array([1.0, 2.0]) + point)
        if diagnostics:
            audits.append(1)
            result.update(
                residual_kind="full_state",
                residual=1e-12,
                spectral_radius=0.8,
                stability_residual=1e-10,
                sensitivity_residual=1e-10,
            )
        return result

    response.adjusted = adjusted
    checks = Checks()
    response.validate(jnp.zeros(2), checks)
    assert all(check["passed"] for check in checks.items)
    assert len(audits) == 1 and 0.3 in starting_states
    assert {"adjusted-state independence", "adjusted-gradient independence"} <= {c["name"] for c in checks.items}


@pytest.mark.parametrize("kind", ["local", "trajectory"])
@pytest.mark.parametrize("heldout", [False, True])
def test_response_products_against_direct_ad(kind, heldout):
    import jax
    import jax.numpy as jnp

    ocean = Recurrence(kind)
    response = Response(
        ocean, (0, 2, 8), dict(mode="smoke", steps=1 if kind == "local" else 9, ram_checkpoints=1, disk_checkpoints=1)
    )
    point, direction = jnp.array([0.1, 0.03, -0.2]), jnp.array([0.2, -0.4, 0.1])

    def direct(z):
        values = []
        for (carry, parameters), boundaries in response.finite_provider().windows(z, heldout):
            previous = 0
            for boundary in boundaries:
                carry, _ = ocean.advance(carry, n_steps=boundary - previous, parameters=parameters)
                values.append(ocean.observe(carry))
                previous = boundary
        return jnp.stack(values)

    expected, tangent = jax.jvp(direct, (point,), (direction,))
    actual, product = response.jvp(point, direction, heldout=heldout)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    np.testing.assert_allclose(product, tangent, rtol=1e-12)
    seed = jnp.sin(jnp.arange(expected.size)).reshape(expected.shape)
    expected_gradient = jax.vjp(direct, point)[1](seed)[0]
    gradient = response.vjp(point, seed, heldout=heldout)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jnp.vdot(seed, tangent), jnp.vdot(direction, gradient), rtol=1e-12)
    function = response.function(lambda values, z: jnp.sum(values**2) + 0.3 * jnp.sum(z**2), heldout=heldout)

    def reference(z):
        return jnp.sum(direct(z) ** 2) + 0.3 * jnp.sum(z**2)

    actual_value, actual_gradient = function.value_and_grad(point)
    from benchmarks.ad_demos.finite_response import forward_value_and_grad

    forward_value, forward_gradient = forward_value_and_grad(function)(point)
    np.testing.assert_allclose(forward_value, actual_value, rtol=1e-12)
    np.testing.assert_allclose(forward_gradient, actual_gradient, rtol=1e-12, atol=1e-12)
    value, gradient = jax.value_and_grad(reference)(point)
    np.testing.assert_allclose(actual_value, value, rtol=1e-12)
    np.testing.assert_allclose(actual_gradient, gradient, rtol=1e-12)
    np.testing.assert_allclose(function.jacobian(point)[0], gradient, rtol=1e-12)
    assert all(s["peak_ram_checkpoints"] <= 1 and s["peak_disk_checkpoints"] <= 1 for s in response.finite.statistics)
    assert response.finite.driver._reverse._cache_size() <= 1
    if kind == "trajectory":
        traces = ocean.traces
        matrix_signatures = response.finite.matrix_step._cache_size()
        response.steps *= 2
        response.vjp(point, jnp.zeros_like(response.values(point)), heldout=heldout)
        response.finite.jacobian(point, heldout=heldout)
        assert response.finite.driver._reverse._cache_size() <= 1
        assert response.finite.matrix_step._cache_size() == matrix_signatures
        assert ocean.traces == traces


@pytest.mark.parametrize("heldout", (False, True))
def test_terminal_feature_matches_direct_ad_without_extra_primal_traversal(monkeypatch, heldout):
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.finite_response import value_and_grad, jvp

    ocean = Recurrence("trajectory")
    response = Response(ocean, (0, 2, 8), dict(mode="smoke", steps=9, ram_checkpoints=1, disk_checkpoints=1))
    point, direction = jnp.array([0.1, 0.03, -0.2]), jnp.array([0.2, -0.4, 0.1])
    function = response.terminal_feature(1, scale=-2.0, offset=0.7, heldout=heldout)

    def direct(z):
        carry, parameters = ocean.controls(response.full(z), phase_days=90 if heldout else 0)
        carry, _ = ocean.advance(carry, n_steps=18 if heldout else 9, parameters=parameters)
        return -2 * ocean.observe(carry)[1] + 0.7

    expected, expected_gradient = jax.value_and_grad(direct)(point)
    np.testing.assert_allclose(function(point), expected, rtol=1e-12)
    np.testing.assert_allclose(jvp(function, point, direction)[1], expected_gradient @ direction, rtol=1e-12)

    def unused(*args, **kwargs):
        raise AssertionError("Terminal reverse products must not request a separate observation traversal")

    monkeypatch.setattr(response, "values", unused)
    monkeypatch.setattr(response.finite_provider(), "values", unused)
    value, gradient = value_and_grad(function)(point)
    np.testing.assert_allclose(value, expected, rtol=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    value, transpose = function.vjp(point, jnp.asarray(0.3))
    np.testing.assert_allclose(value, expected, rtol=1e-12)
    np.testing.assert_allclose(transpose, 0.3 * expected_gradient, rtol=1e-12, atol=1e-12)
    assert len(response.finite.statistics) == 2
    assert all(s["forward_steps"] == (18 if heldout else 9) for s in response.finite.statistics)
    assert all(s["peak_ram_checkpoints"] <= 1 and s["peak_disk_checkpoints"] <= 1 for s in response.finite.statistics)


@pytest.mark.parametrize("kind", ("local", "equilibrium", "seasonal", "statistics"))
def test_terminal_feature_rejects_other_response_semantics(kind):
    ocean = Recurrence(kind)
    response = Response(ocean, (0,), dict(mode="smoke", steps=1))
    with pytest.raises(ValueError, match="trajectory"):
        response.terminal_feature(0)


@pytest.mark.parametrize("slots,direct", ((4, False), (84, True)))
def test_terminal_feature_selects_schedule_from_full_objective_work(slots, direct):
    from benchmarks.ad_demos.finite_response import TerminalFeature

    response = Response(Recurrence("trajectory"), (0, 2, 8), dict(mode="study", steps=3600, ram_checkpoints=slots))
    function = response.terminal_feature(1, scale=-2, offset=0.7)
    assert isinstance(function, TerminalFeature) == direct
    work = function.checkpoint_work
    assert (work["direct"] <= work["segmented"]) == direct
    assert work == (dict(direct=10716, segmented=17865) if direct else dict(direct=46160, segmented=41775))


def test_terminal_feature_segmented_products_match_direct_ad():
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.finite_response import TerminalFeature

    ocean = Recurrence("trajectory")
    response = Response(ocean, (0, 2, 8), dict(mode="smoke", steps=21, ram_checkpoints=1))
    function = response.terminal_feature(1, scale=-2, offset=0.7)
    assert not isinstance(function, TerminalFeature)
    point = jnp.array([0.1, 0.03, -0.2])

    def direct(z):
        carry, parameters = ocean.controls(response.full(z))
        carry, _ = ocean.advance(carry, n_steps=21, parameters=parameters)
        return -2 * ocean.observe(carry)[1] + 0.7

    expected, expected_gradient = jax.value_and_grad(direct)(point)
    value, gradient = function.value_and_grad(point)
    np.testing.assert_allclose(value, expected, rtol=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    assert len(response.finite.statistics) == 3


@pytest.mark.parametrize("kind", ("local", "trajectory"))
def test_global_phase_release_retains_native_product_values(kind):
    import jax
    import jax.numpy as jnp

    ocean = Recurrence(kind)
    response = Response(
        ocean, (0, 2, 8), dict(setup="global_4deg", mode="smoke", steps=1 if kind == "local" else 9, ram_checkpoints=1)
    )
    point = jnp.array([0.1, 0.03, -0.2])

    def direct(z):
        values = []
        for (carry, parameters), boundaries in response.finite_provider().windows(z, False):
            previous = 0
            for boundary in boundaries:
                carry, _ = ocean.advance(carry, n_steps=boundary - previous, parameters=parameters)
                values.append(ocean.observe(carry))
                previous = boundary
        return jnp.stack(values)

    expected = direct(point)
    seed = jnp.cos(jnp.arange(expected.size)).reshape(expected.shape)
    reference = jax.vjp(direct, point)[1](seed)[0]
    np.testing.assert_allclose(response.vjp(point, seed), reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ("local", "trajectory"))
@pytest.mark.parametrize("heldout", (False, True))
def test_mapped_response_products_against_direct_ad(kind, heldout):
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.finite_response import value_and_grad, jvp

    ocean = Recurrence(kind)
    response = Response(ocean, (0, 2, 8), dict(mode="smoke", steps=1 if kind == "local" else 9, ram_checkpoints=1))
    point, direction = jnp.array([0.1, -0.2]), jnp.array([0.2, -0.4])

    def mapping(z):
        return jnp.stack((jnp.sin(z[0]), z[0] * z[1], jnp.asarray(-0.1)))

    def direct(z):
        values = []
        for (carry, parameters), boundaries in response.finite_provider().windows(mapping(z), heldout):
            previous = 0
            for boundary in boundaries:
                carry, _ = ocean.advance(carry, n_steps=boundary - previous, parameters=parameters)
                values.append(ocean.observe(carry))
                previous = boundary
        return jnp.stack(values)

    def scalar(z):
        return jnp.sum(direct(z) ** 2) + 0.3 * jnp.sum(mapping(z) ** 2)

    objective = response.function(lambda values, z: jnp.sum(values**2) + 0.3 * jnp.sum(z**2), heldout=heldout)
    mapped = objective.map_controls(mapping)
    expected_value, expected_gradient = jax.value_and_grad(scalar)(point)
    value, gradient = value_and_grad(mapped)(point)
    np.testing.assert_allclose(value, expected_value, rtol=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jvp(mapped, point, direction)[1], jnp.vdot(expected_gradient, direction), rtol=1e-12)
    vector = response.function(lambda values, z: values, heldout=heldout).map_controls(mapping)
    expected = direct(point)
    seed = jnp.cos(jnp.arange(expected.size)).reshape(expected.shape)
    value, gradient = vector.vjp(point, seed)
    np.testing.assert_allclose(value, expected, rtol=1e-12)
    np.testing.assert_allclose(gradient, jax.vjp(direct, point)[1](seed)[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(vector.jacobian(point), jax.jacfwd(direct)(point).reshape(-1, 2), rtol=1e-12, atol=1e-12)
    assert all(s["peak_ram_checkpoints"] <= 1 for s in response.finite.statistics)


class AnalyticResponse:
    def values(self, z, **kwargs):
        return z

    def jvp(self, z, direction, **kwargs):
        return z, direction

    def vjp(self, z, seed, **kwargs):
        return seed


def test_finite_difference_sweep_resolves_local_branch_without_relaxing_tolerance():
    import jax.numpy as jnp
    from benchmarks.ad_demos.common import Checks, gradient_checks

    def function(z):
        return jnp.where(jnp.abs(z[0]) > 2e-5, 2 * z[0], z[0])

    checks = Checks()
    gradient_checks(function, jnp.zeros(1), checks)
    plateau = next(check for check in checks.items if check["name"] == "finite-difference plateau")
    assert plateau["passed"]
    assert plateau["detail"]["original_scale_matches"] == 0
    assert len(plateau["detail"]["epsilons"]) == 9
    np.testing.assert_allclose(plateau["detail"]["derivatives"][:5], 2)
    np.testing.assert_allclose(plateau["detail"]["derivatives"][-3:], 1)


def test_taylor_sweep_resolves_local_branch_even_when_central_differences_match():
    import jax.numpy as jnp
    from benchmarks.ad_demos.common import Checks, gradient_checks

    def function(z):
        return z[0] + jnp.where(jnp.abs(z[0]) > 2e-5, 1e-3 * jnp.abs(z[0]), z[0] ** 2)

    checks = Checks()
    gradient_checks(function, jnp.zeros(1), checks)
    plateau = next(c for c in checks.items if c["name"] == "finite-difference plateau")
    taylor = next(c for c in checks.items if c["name"] == "Taylor remainder")
    assert plateau["detail"]["original_scale_matches"] == 5
    assert taylor["passed"] and len(taylor["detail"]["epsilons"]) == 9
    np.testing.assert_allclose(taylor["detail"]["orders"][:4], 1)
    np.testing.assert_allclose(taylor["detail"]["orders"][-2:], 2, atol=1e-3)


def test_coarser_taylor_scales_resolve_curvature_above_float32_roundoff():
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.common import Checks, gradient_checks

    jax.config.update("jax_enable_x64", True)

    def objective(z):
        value = (1 + 0.006 * z[0] + 0.003 * z[0] ** 2).astype(jnp.float32)
        return 0.9 + 0.005 * (value.astype(jnp.float64) - 1)

    initial, extended = Checks(), Checks()
    gradient_checks(objective, jnp.zeros(1), initial)
    gradient_checks(objective, jnp.zeros(1), extended, coarser_scales=(0.1, 0.03))
    first = next(c for c in initial.items if c["name"] == "Taylor remainder")
    second = next(c for c in extended.items if c["name"] == "Taylor remainder")
    assert not first["passed"] and second["passed"]
    assert all(c["passed"] for c in extended.items)
    assert second["detail"]["epsilons"][:9] == first["detail"]["epsilons"]
    assert second["detail"]["unresolved_tolerance"] == first["detail"]["unresolved_tolerance"] == 1e-9
    np.testing.assert_allclose(second["detail"]["orders"][-1], 2, atol=0.05)


def test_explicit_finer_scales_resolve_a_narrow_branch_and_retain_failed_samples():
    import jax.numpy as jnp
    from benchmarks.ad_demos.common import Checks, gradient_checks

    def objective(z):
        return jnp.where(jnp.abs(z[0]) > 2e-7, 2 * z[0], z[0]) + 1e6 * z[0] ** 2

    initial, extended = Checks(), Checks()
    gradient_checks(objective, jnp.zeros(1), initial)
    gradient_checks(objective, jnp.zeros(1), extended, finer_scales=(1e-7, 3e-8))
    first = next(c for c in initial.items if c["name"] == "finite-difference plateau")
    second = next(c for c in extended.items if c["name"] == "finite-difference plateau")
    assert not first["passed"] and all(c["passed"] for c in extended.items)
    assert second["detail"]["epsilons"][:9] == first["detail"]["epsilons"]
    assert second["detail"]["derivatives"][:9] == first["detail"]["derivatives"]
    np.testing.assert_allclose(second["detail"]["derivatives"][-2:], 1)


@pytest.mark.parametrize("valid_derivative", (False, True))
@pytest.mark.parametrize("finer_scales", ((), (1e-7, 3e-8)))
def test_response_audit_rejects_unvalidated_derivatives_and_retains_evidence(valid_derivative, finer_scales):
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.common import Checks
    from benchmarks.ad_demos.responses import Response

    @jax.custom_jvp
    def function(z):
        return z[0]

    @function.defjvp
    def product(primals, tangents):
        return function(primals[0]), tangents[0][0] * (1 if valid_derivative else 2)

    response = object.__new__(Response)
    response.kind = "trajectory"
    response.config = dict(gradient_finer_scales=finer_scales)
    checks = Checks()
    checks.require("scientific target", False, "unmet target")
    if valid_derivative:
        response.audit(function, jnp.zeros(1), checks)
        assert all(check["passed"] for check in checks.items[1:])
    else:
        with pytest.raises(RuntimeError, match="Derivative validation failed: finite-difference plateau"):
            response.audit(function, jnp.zeros(1), checks)
        plateau = next(check for check in checks.items if check["name"] == "finite-difference plateau")
        assert not plateau["passed"]
        assert len(plateau["detail"]["epsilons"]) == 9 + len(finer_scales)


@pytest.mark.parametrize("matrix_free", [False, True])
@pytest.mark.parametrize("host_response", [False, True])
def test_scaled_inverse_recovers_identifiable_controls(matrix_free, host_response):
    import jax.numpy as jnp

    truth = jnp.array([0.2, -0.3])
    matrix = jnp.array([[1000.0, 0.0], [0.0, 0.01], [0.1, 0.03]])
    residual = (
        ObservationFunction(AnalyticResponse(), lambda values, z: matrix @ (values - truth))
        if host_response
        else JaxResidual(lambda z: matrix @ (z - truth))
    )
    fit, history, diagnostic = least_squares(residual, jnp.zeros(2), prior_precision=0, matrix_free=matrix_free)
    assert diagnostic["converged"], diagnostic
    np.testing.assert_allclose(fit, truth, atol=1e-7)
    assert history[-1] < 1e-12


def test_inverse_respects_active_domain_and_records_stopping():
    import jax.numpy as jnp

    residual = ObservationFunction(AnalyticResponse(), lambda values, z: values - jnp.array([2.0, 0.0]))
    fit, _, diagnostic = least_squares(residual, jnp.zeros(2), prior_precision=0)
    np.testing.assert_allclose(fit, [1, 0], atol=1e-7)
    assert diagnostic["converged"]
    _, _, limited = least_squares(residual, jnp.zeros(2), max_iterations=0)
    assert not limited["converged"] and limited["stopping"] == "iteration_limit"


def test_dense_inverse_retains_weak_independent_columns():
    import jax.numpy as jnp

    truth = jnp.array([0.2, -0.3])
    matrix = jnp.array([[1.0, 1.0], [0.0, 1e-7]])
    residual = JaxResidual(lambda z: matrix @ (z - truth))
    fit, _, diagnostic = least_squares(residual, jnp.zeros(2), prior_precision=0)
    assert diagnostic["converged"]
    np.testing.assert_allclose(fit, truth, atol=1e-8)


def test_exact_inverse_ends_starts_only_after_converged_zero_loss():
    import jax.numpy as jnp

    residual = JaxResidual(lambda z: z - jnp.array([0.2, -0.3]))
    fit, _, diagnostic = multistart_least_squares(residual, jnp.zeros(2), prior_precision=0, zero_loss_tolerance=1e-14)
    assert diagnostic["converged"] and len(diagnostic["starts"]) == 1
    np.testing.assert_allclose(fit, [0.2, -0.3], atol=1e-10)
    _, _, diagnostic = multistart_least_squares(
        residual, jnp.zeros(2), prior_precision=0, max_iterations=0, zero_loss_tolerance=1e-14
    )
    assert not diagnostic["converged"] and len(diagnostic["starts"]) == 3


@pytest.mark.parametrize("matrix_free", [False, True])
def test_zero_loss_stopping_certifies_exact_fit_without_masking_other_failures(matrix_free):
    import jax.numpy as jnp

    residual = JaxResidual(lambda z: 1000 * z)
    point = jnp.array([1e-10])
    _, history, diagnostic = least_squares(
        residual, point, prior_precision=0, max_iterations=0, matrix_free=matrix_free, zero_loss_tolerance=1e-14
    )
    assert history[-1] <= 1e-14 and diagnostic["projected_gradient_norm"] > diagnostic["gradient_tolerance"]
    assert diagnostic["converged"] and diagnostic["stopping"] == "zero_loss_tolerance"
    for kwargs in ({}, dict(prior_precision=1e-3)):
        options = dict(prior_precision=0, max_iterations=0, matrix_free=matrix_free)
        options.update(kwargs)
        if kwargs:
            options["zero_loss_tolerance"] = 1e-14
        _, _, diagnostic = least_squares(residual, point, **options)
        assert not diagnostic["converged"]
    _, _, diagnostic = least_squares(
        JaxResidual(lambda z: jnp.concatenate((1000 * z, jnp.ones(1)))),
        point,
        prior_precision=0,
        max_iterations=0,
        matrix_free=matrix_free,
        zero_loss_tolerance=1e-14,
    )
    assert not diagnostic["converged"]


@pytest.mark.parametrize("matrix_free", [False, True])
def test_exact_fit_avoids_full_jacobian_but_rejects_nonfinite_gradient(matrix_free):
    import jax.numpy as jnp

    class TransposeOnly(JaxResidual):
        def jacobian(self, point):
            raise AssertionError("An already fitted residual does not need another full Jacobian")

        def jvp(self, point, direction):
            raise AssertionError("An already fitted residual does not need tangent columns")

    residual = TransposeOnly(lambda z: 1000 * z)
    point = jnp.full(6, 1e-11)
    _, _, diagnostic = least_squares(
        residual, point, prior_precision=0, zero_loss_tolerance=1e-14, matrix_free=matrix_free
    )
    assert diagnostic["converged"] and diagnostic["stopping"] == "zero_loss_tolerance"
    assert np.isfinite(diagnostic["gradient_norm"])
    _, _, diagnostic = least_squares(
        TransposeOnly(lambda z: jnp.sqrt(jnp.abs(z))),
        jnp.zeros(6),
        prior_precision=0,
        zero_loss_tolerance=1e-14,
        matrix_free=matrix_free,
    )
    assert not diagnostic["converged"] and diagnostic["stopping"] == "nonfinite_response"


def test_regularized_inverse_matches_map_while_exact_twin_recovers_truth():
    import jax.numpy as jnp

    truth = jnp.array([0.2, 0.3])
    matrix = jnp.diag(jnp.array([1.0, 0.01]))
    residual = ObservationFunction(AnalyticResponse(), lambda values, z: matrix @ (values - truth))
    regularized, _, diagnostic = least_squares(residual, jnp.zeros(2), prior_precision=1e-3)
    expected = np.linalg.solve(np.asarray(matrix.T @ matrix) + 1e-3 * np.eye(2), np.asarray(matrix.T @ matrix @ truth))
    np.testing.assert_allclose(regularized, expected, atol=1e-10)
    assert diagnostic["converged"] and np.linalg.norm(regularized - truth) > 0.15
    exact, _, diagnostic = least_squares(residual, jnp.zeros(2), prior_precision=0)
    np.testing.assert_allclose(exact, truth, atol=1e-10)
    assert diagnostic["converged"]


def test_feasibility_restoration_and_constrained_optimality():
    import jax.numpy as jnp

    constraint = ObservationFunction(AnalyticResponse(), lambda values, z: values[0] + 2 * values[1] - 0.5)
    fit, diagnostic = constrained_minimum(constraint, jnp.zeros(2))
    np.testing.assert_allclose(fit, [0.1, 0.2], atol=1e-7)
    assert diagnostic["feasible"] and diagnostic["stationarity_residual"] < 1e-7
    assert np.max(np.abs(diagnostic["complementarity"])) < 1e-7
    impossible = ObservationFunction(AnalyticResponse(), lambda values, z: values[0] - 2)
    fit, diagnostic = constrained_minimum(impossible, jnp.zeros(2))
    assert not diagnostic["feasible"] and np.linalg.norm(fit) <= 1 + 1e-9
    assert len(diagnostic["feasibility_searches"]) == 3


def test_correlated_observation_covariance_and_cost():
    matrix = np.array([[1, 0], [0, 2], [1, 1]])
    errors = np.array([[1, 0.5, 0], [0.5, 2, 0], [0, 0, 1]])
    actual, cost = observation_covariance(matrix, [0, 1], errors=errors, costs=[2, 3, 1])
    joint = np.eye(2) - matrix[:2].T @ np.linalg.solve(errors[:2, :2] + matrix[:2] @ matrix[:2].T, matrix[:2])
    np.testing.assert_allclose(actual, joint, atol=1e-12)
    assert cost == 5


@pytest.mark.parametrize("scale", (0.047, 1, 1000))
def test_constrained_boundary_search_is_independent_of_observation_scale(scale):
    import jax.numpy as jnp

    normal = jnp.array([1.0, 0.2])
    target = 0.95
    fit, diagnostic = constrained_minimum(lambda z: scale * (normal @ z - target), jnp.zeros(2))
    np.testing.assert_allclose(fit, np.asarray(normal) * target / float(normal @ normal), atol=1e-7)
    assert diagnostic["success"] and diagnostic["feasible"]
    assert diagnostic["stationarity_residual"] < 1e-7
    impossible, diagnostic = constrained_minimum(lambda z: scale * (normal @ z - 2), jnp.zeros(2))
    np.testing.assert_allclose(impossible, np.asarray(normal) / float(jnp.linalg.norm(normal)), atol=1e-5)
    assert not diagnostic["feasible"]
    assert all(search["success"] and search["iterations"] < 20 for search in diagnostic["feasibility_searches"])


@pytest.mark.parametrize("physics", ["default", "fixed"])
def test_scientific_ocean_objective_matches_direct_ad(physics):
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.research_ocean import ResearchOcean

    config = dict(
        setup="acc",
        mode="smoke",
        steps=3,
        spinup_steps=0,
        physics=physics,
        neutral_policy="nonstable",
        ram_checkpoints=1,
        disk_checkpoints=1,
    )
    ocean = ResearchOcean(config, CASES["estimation_trajectory"])
    response = Response(ocean, (0, 1, 2, 4, 8, 9), config)
    point = jnp.array([0.05, -0.02, 0.1, 0.03, 0.02, -0.01])

    def observation_loss(values, z):
        return jnp.sum(values[:, ocean.heat_index]) + jnp.sum(values[:, ocean.acc_index]) + 0.1 * jnp.sum(z**2)

    def direct(z):
        carry, parameters = ocean.controls(response.full(z))
        values = []
        for _ in range(3):
            carry, _ = ocean.prepared.advance(carry, n_steps=1, parameters=parameters, final_only=True)
            values.append(ocean.observe(carry))
        return observation_loss(jnp.stack(values), z)

    function = response.function(observation_loss)
    reference, reference_gradient = jax.value_and_grad(direct)(point)
    value, gradient = function.value_and_grad(point)
    np.testing.assert_allclose(value, reference, rtol=1e-11)
    np.testing.assert_allclose(gradient, reference_gradient, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(function.jacobian(point)[0], reference_gradient, rtol=1e-8, atol=1e-10)
    direction = jnp.arange(1, point.size + 1, dtype=point.dtype)
    np.testing.assert_allclose(function.jvp(point, direction)[1], jnp.vdot(gradient, direction), rtol=1e-8)
    jax.clear_caches()


@pytest.mark.parametrize(
    "physics,tke_policy", (("default", "default"), ("default", "positive_diffusion"), ("fixed", "default"))
)
def test_scientific_ocean_terminal_feature_matches_direct_ad(physics, tke_policy):
    import jax
    import jax.numpy as jnp
    from benchmarks.ad_demos.research_ocean import ResearchOcean

    config = dict(
        setup="acc",
        mode="smoke",
        steps=3,
        spinup_steps=0,
        physics=physics,
        neutral_policy="nonstable",
        tke_policy=tke_policy,
        ram_checkpoints=1,
        disk_checkpoints=1,
    )
    ocean = ResearchOcean(config, CASES["estimation_trajectory"])
    response = Response(ocean, (0, 1, 2, 4, 8, 9), config)
    point = jnp.array([0.05, -0.02, 0.1, 0.03, 0.02, -0.01])

    def direct(z):
        carry, parameters = ocean.controls(response.full(z))
        carry, _ = ocean.prepared.advance(carry, n_steps=3, parameters=parameters, final_only=True)
        return -(ocean.observe(carry)[ocean.heat_index] - 0.4)

    expected, expected_gradient = jax.value_and_grad(direct)(point)
    function = response.terminal_feature(ocean.heat_index, scale=-1, offset=0.4)
    value, gradient = function.value_and_grad(point)
    np.testing.assert_allclose(value, expected, rtol=1e-11)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-8, atol=1e-10)
    direction = jnp.arange(1, point.size + 1, dtype=point.dtype)
    np.testing.assert_allclose(function.jvp(point, direction)[1], jnp.vdot(gradient, direction), rtol=1e-8, atol=1e-10)
    (statistics,) = response.finite.statistics
    assert statistics["forward_steps"] == 3 and statistics["terminal_feature"]
    jax.clear_caches()


@pytest.mark.parametrize(
    "task,days", (("sensitivity", 30), ("geometry", 30), ("estimation", 360), ("perturbation", 3600))
)
def test_default_trajectory_windows_match_declared_scientific_program(task, days):
    from types import SimpleNamespace
    import jax.numpy as jnp

    ocean = SimpleNamespace(dt=86400.0, case=CASES[task + "_trajectory"], initial=None, observe=lambda _: jnp.zeros(1))
    response = Response(ocean, (), dict(mode="study", steps=None))
    assert response.steps == days
