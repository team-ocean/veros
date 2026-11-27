import numpy as np
import pytest

from veros import runtime_settings


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("wet_index", (None, (1, 1, 0), (1, 0, 4)))
def test_numpy_tdma_empty_and_single_wet_cell(dtype, wet_index):
    from veros.core.operators import solve_tridiagonal_numpy

    shape = (2, 3, 5)
    a, b, c, d = (np.full(shape, value, dtype=dtype) for value in (-0.2, 4.0, -0.3, 2.0))
    water = np.zeros(shape, dtype=bool)
    expected = np.zeros(shape, dtype=dtype)
    if wet_index is not None:
        water[wet_index] = True
        expected[wet_index] = 0.5
    before = [value.copy() for value in (a, b, c, d)]
    result = solve_tridiagonal_numpy(a, b, c, d, water, water)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == dtype
    for value, original in zip((a, b, c, d), before):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("depth", (1, 6))
def test_tdma_implicit_gradients(dtype, depth):
    if runtime_settings.backend != "jax":
        pytest.skip("TDMA derivatives require JAX")
    import jax
    import jax.numpy as jnp
    from veros.core.special.tdma_ import tdma, HAS_CPU_EXT, HAS_GPU_EXT
    from veros.core.operators import solve_tridiagonal_jax

    if not (HAS_CPU_EXT if runtime_settings.device == "cpu" else HAS_GPU_EXT):
        pytest.skip("accelerated TDMA extension is unavailable")
    rng = np.random.default_rng(17)
    shape = (2, 3, depth)
    coefficients = tuple(
        jnp.asarray(value, dtype=dtype)
        for value in (
            rng.uniform(-0.4, -0.1, shape),
            rng.uniform(2, 3, shape),
            rng.uniform(-0.3, -0.1, shape),
            rng.normal(size=shape),
        )
    )
    water = np.ones(shape, dtype=bool)
    water[0, 0] = False
    water[0, 1, : depth // 2] = False
    edge = water & ~np.concatenate((np.zeros_like(water[..., :1]), water[..., :-1]), axis=-1)
    raw = getattr(solve_tridiagonal_jax.function, "__wrapped__", solve_tridiagonal_jax.function)
    previous = runtime_settings.use_special_tdma
    object.__setattr__(runtime_settings, "use_special_tdma", False)
    try:
        reference = jax.jit(lambda *values: raw(*values, jnp.asarray(water), jnp.asarray(edge)))
        expected = reference(*coefficients)
    finally:
        object.__setattr__(runtime_settings, "use_special_tdma", previous)
    accelerated = jax.jit(lambda *values: tdma(*values, jnp.asarray(water), jnp.asarray(edge)))
    actual = accelerated(*coefficients)
    tolerance = 2e-5 if dtype == "float32" else 1e-10
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    weights = jnp.asarray(rng.normal(size=shape), dtype=dtype)
    directions = tuple(jnp.asarray(rng.normal(size=shape), dtype=dtype) for _ in coefficients)

    def objective(*values):
        return jnp.sum(weights * accelerated(*values))

    gradients = jax.jit(jax.grad(objective, argnums=(0, 1, 2, 3)))(*coefficients)
    _, tangent = jax.jit(lambda *values: jax.jvp(accelerated, values, directions))(*coefficients)
    contraction = sum(jnp.vdot(gradient, direction) for gradient, direction in zip(gradients, directions))
    np.testing.assert_allclose(contraction, jnp.vdot(weights, tangent), rtol=tolerance, atol=tolerance)
    step = 1e-2 if dtype == "float32" else 1e-4
    finite_difference = (
        objective(*(value + step * direction for value, direction in zip(coefficients, directions)))
        - objective(*(value - step * direction for value, direction in zip(coefficients, directions)))
    ) / (2 * step)
    np.testing.assert_allclose(contraction, finite_difference, rtol=2e-3, atol=tolerance)
    for gradient in gradients:
        assert np.isfinite(gradient).all()
        np.testing.assert_array_equal(np.asarray(gradient)[~water], 0)
