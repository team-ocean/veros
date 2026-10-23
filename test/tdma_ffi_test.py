"""Check native FFI solves against independent dense systems."""

import numpy as np
import pytest

from veros import runtime_settings

pytestmark = pytest.mark.skipif(runtime_settings.backend != "jax", reason="requires JAX")


@pytest.fixture(autouse=True)
def native_tdma():
    import jax
    from veros.core.special import tdma_

    available = tdma_.HAS_CPU_EXT if runtime_settings.device == "cpu" else tdma_.HAS_GPU_EXT
    if not available:
        pytest.skip("native TDMA extension is not installed")
    yield tdma_.tdma
    jax.clear_caches()


def make_systems(shape, dtype):
    rng = np.random.default_rng(199 + shape[-1])
    values = tuple(
        np.asarray(value, dtype=dtype)
        for value in (
            rng.uniform(-0.4, -0.1, shape),
            rng.uniform(2, 3, shape),
            rng.uniform(-0.3, -0.1, shape),
            rng.normal(size=shape),
        )
    )
    depths = rng.integers(0, shape[-1] + 1, shape[:-1])
    depths.flat[:3] = (0, 1, shape[-1])
    water = np.arange(shape[-1]) >= shape[-1] - depths[..., None]
    edge = water & ~np.concatenate((np.zeros_like(water[..., :1]), water[..., :-1]), axis=-1)
    return values, water, edge


def dense_solution(values, water, edge):
    expected = np.zeros(water.shape)
    for index in np.ndindex(water.shape[:-1]):
        a, b, c, rhs = [np.asarray(value[index], dtype="float64") for value in values]
        a = np.where(water[index] & ~edge[index], a, 0)
        b = np.where(water[index], b, 1)
        c = np.where(water[index], c, 0)
        rhs = np.where(water[index], rhs, 0)
        matrix = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
        expected[index] = np.linalg.solve(matrix, rhs)
    return expected


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("depth", (1, 2, 3, 15, 60))
def test_dense_oracle(native_tdma, dtype, depth):
    import jax
    import jax.numpy as jnp

    values, water, edge = make_systems((2, 3, depth), dtype)
    inputs = tuple(jnp.asarray(value) for value in values)
    expected = dense_solution(values, water, edge)
    tolerance = 5e-6 if dtype == "float32" else 2e-12
    for solve in (native_tdma, jax.jit(native_tdma)):
        actual = solve(*inputs, jnp.asarray(water), jnp.asarray(edge))
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("shape", ((15,), (6, 15), (2, 3, 15), (2, 2, 3, 15)))
def test_column_layout(native_tdma, shape):
    import jax
    import jax.numpy as jnp

    if len(shape) == 1:
        values, water, edge = make_systems((3, shape[-1]), "float64")
        values, water, edge = tuple(v[1] for v in values), water[1], edge[1]
    else:
        values, water, edge = make_systems(shape, "float64")
    result = jax.jit(native_tdma)(*(jnp.asarray(v) for v in values), jnp.asarray(water), jnp.asarray(edge))
    np.testing.assert_allclose(result, dense_solution(values, water, edge), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
@pytest.mark.parametrize("batch_coefficients", (False, True))
@pytest.mark.parametrize("batch_masks", (False, True))
def test_batching(native_tdma, batch_coefficients, batch_masks, dtype):
    import jax
    import jax.numpy as jnp

    examples = [make_systems((2, 3, 15), dtype) for _ in range(3)]
    coefficients = tuple(jnp.asarray(value) for value in examples[0][0])
    coefficients = tuple(
        jnp.stack(tuple(value + i * 0.1 for i in range(3))) if batch_coefficients or k == 3 else value
        for k, value in enumerate(coefficients)
    )
    mask_pairs = [(example[1], example[2]) for example in examples]
    mask_pairs[1] = tuple(np.flip(mask, axis=0).copy() for mask in mask_pairs[1])
    masks = tuple(
        jnp.stack(tuple(pair[k] for pair in mask_pairs)) if batch_masks else jnp.asarray(mask_pairs[0][k])
        for k in range(2)
    )
    axes = (0 if batch_coefficients else None,) * 3 + (0,) + (0 if batch_masks else None,) * 2
    inputs = (*coefficients, *masks)
    expected = np.stack(
        tuple(
            dense_solution(
                tuple(np.asarray(value if axis is None else value[i]) for value, axis in zip(coefficients, axes)),
                *(np.asarray(value if axis is None else value[i]) for value, axis in zip(masks, axes[-2:]))
            )
            for i in range(3)
        )
    )
    result = jax.jit(jax.vmap(native_tdma, in_axes=axes))(*inputs)
    tolerance = 5e-6 if dtype == "float32" else 2e-12
    np.testing.assert_allclose(result, expected, rtol=tolerance, atol=tolerance)


def test_empty_horizontal_domain(native_tdma):
    import jax
    import jax.numpy as jnp

    values = (jnp.empty((0, 3, 15)),) * 4
    masks = (jnp.empty((0, 3, 15), dtype=bool),) * 2
    assert jax.jit(native_tdma)(*values, *masks).shape == (0, 3, 15)


@pytest.mark.parametrize("invalid", ("dtype", "mixed_dtype", "shape", "mask", "depth", "device"))
def test_invalid_arguments(native_tdma, invalid):
    import jax.numpy as jnp

    values = [jnp.ones((2, 3, 4)) for _ in range(4)]
    masks = [jnp.ones((2, 3, 4), dtype=bool) for _ in range(2)]
    kwargs = {}
    if invalid == "dtype":
        values = [value.astype("int32") for value in values]
    elif invalid == "mixed_dtype":
        values[0] = values[0].astype("float32")
    elif invalid == "shape":
        values[0] = values[0][..., :-1]
    elif invalid == "mask":
        masks[0] = masks[0][..., :-1]
    elif invalid == "depth":
        values = [value[..., :0] for value in values]
        masks = [mask[..., :0] for mask in masks]
    else:
        kwargs["device"] = "tpu"
    with pytest.raises((ValueError, TypeError)):
        native_tdma(*values, *masks, **kwargs)


def test_handler_rejects_mismatched_buffers(native_tdma):
    import jax
    import jax.numpy as jnp

    gpu = runtime_settings.device == "gpu"
    values = [jnp.ones((2, 3, 4)) for _ in range(4)]
    values[1] = jnp.ones((2, 3, 3))
    result = jax.ShapeDtypeStruct((2, 3, 4), jnp.float64)
    if gpu:
        name, outputs = "tdma_cuda_double", (result, result)
    else:
        name = "tdma_cython_double"
        outputs = (result, jax.ShapeDtypeStruct((4,), jnp.float64))
        values.append(jnp.full((2, 3), 4, dtype="int32"))
    with pytest.raises(jax.errors.JaxRuntimeError, match="TDMA buffer shapes do not match"):
        jax.block_until_ready(jax.ffi.ffi_call(name, outputs)(*values))
