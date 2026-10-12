import numpy as np
import pytest

from veros import runtime_settings


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_zero_square_root_derivative(dtype):
    if runtime_settings.backend != "jax":
        pytest.skip("operator derivatives require JAX")
    from veros.core.operators import safe_sqrt
    import jax
    import jax.numpy as jnp

    values = jnp.array([0, 1, 9], dtype=dtype)
    result, tangent = jax.jit(lambda x: jax.jvp(safe_sqrt, (x,), (jnp.ones_like(x),)))(values)
    gradient = jax.jit(jax.grad(lambda x: safe_sqrt(x).sum()))(values)
    np.testing.assert_array_equal(result, [0, 1, 3])
    np.testing.assert_allclose(gradient, [0, 0.5, 1 / 6], rtol=1e-7)
    np.testing.assert_allclose(gradient, tangent, rtol=1e-7)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_small_denominator_derivatives(dtype):
    if runtime_settings.backend != "jax":
        pytest.skip("operator derivatives require JAX")
    from veros.core.operators import stable_divide
    from veros.core.advection import _calc_cr, limiter
    import jax
    import jax.numpy as jnp

    numerator = jnp.array([0, 1e-20, 1], dtype=dtype)
    denominator = jnp.array([1e-20, 1e-20, 2], dtype=dtype)
    gradients = jax.jit(jax.grad(lambda a, b: stable_divide(a, b).sum(), argnums=(0, 1)))(numerator, denominator)
    np.testing.assert_allclose(gradients[0], [1e20, 1e20, 0.5], rtol=1e-7)
    np.testing.assert_allclose(gradients[1], [0, -1e20, -0.25], rtol=1e-7)
    assert all(np.isfinite(value).all() for value in gradients)

    def limited(r):
        return limiter(_calc_cr(r[0], r[1], r[2], jnp.array(1, dtype=dtype)))

    flat = jnp.zeros(3, dtype=dtype)
    gradient = jax.jit(jax.grad(limited))(flat)
    _, tangent = jax.jit(lambda x: jax.jvp(limited, (x,), (jnp.zeros_like(x),)))(flat)
    assert np.isfinite(gradient).all()
    assert float(tangent) == 0


@pytest.mark.parametrize("distributed", (False, True))
def test_variable_pytree_transformations(distributed):
    if runtime_settings.backend != "jax":
        pytest.skip("PyTree transformations require JAX")
    import jax
    from veros.state import VerosVariables, DistSafeVariableWrapper
    from veros.variables import Variable

    variables = VerosVariables(
        {
            "value": Variable("value", None, initial=2.0),
            "counter": Variable("counter", None, dtype="int32", initial=0),
        },
        {},
    )
    if distributed:
        wrapper = DistSafeVariableWrapper(variables, ("value", "counter"))
        with wrapper.unlock():
            wrapper.update(variables)
        variables = wrapper

    shape = jax.eval_shape(lambda values: values, variables)
    assert isinstance(shape.value, jax.ShapeDtypeStruct)
    assert shape.value.shape == variables.value.shape
    assert shape.value.dtype == variables.value.dtype

    gradient = jax.grad(lambda values: values.value**2, allow_int=True)(variables)
    np.testing.assert_allclose(gradient.value, 4)
    assert gradient.counter.dtype == jax.dtypes.float0
