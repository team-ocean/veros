"""Independent recurrence and full-state checks for discrete shadowing pilots."""

import numpy as np
import pytest

from benchmarks.ad_demos.nonintrusive_shadowing import nilss, nilsas


@pytest.mark.parametrize("segment_length", [2, 4, 8])
def test_full_subspace_matches_dense_minimum_norm(segment_length):
    length, dimension = 32, 2
    rng = np.random.default_rng(12)
    matrices = rng.normal(scale=0.1, size=(length, dimension, dimension)) + 1.3 * np.eye(dimension)
    forcing = rng.normal(size=(length, dimension, 3))
    result = nilss(lambda step, vectors: matrices[step] @ vectors, forcing, np.eye(dimension), segment_length)
    operator = np.zeros((length * dimension, (length + 1) * dimension))
    for step, matrix in enumerate(matrices):
        operator[step * dimension : (step + 1) * dimension, step * dimension : (step + 1) * dimension] = -matrix
        operator[step * dimension : (step + 1) * dimension, (step + 1) * dimension : (step + 2) * dimension] = np.eye(
            dimension
        )
    expected = np.linalg.lstsq(operator, forcing.reshape(length * dimension, 3), rcond=None)[0]
    np.testing.assert_allclose(result["values"].reshape(-1, 3), expected, atol=1e-11)
    assert result["recurrence_residual"] < 1e-11


@pytest.mark.parametrize("segment_length", [4, 8])
def test_one_unstable_mode_with_rotating_stable_direction_and_backward_adjoint(segment_length):
    length = 256
    angle = 0.2 * np.sin(np.arange(length + 1))
    bases = np.array([[[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]] for a in angle])
    matrices = bases[1:] @ np.diag([1.7, 0.4]) @ bases[:-1].transpose(0, 2, 1)
    target = bases @ np.array([[0.2, -0.3], [0.7, 0.4]])
    forcing = target[1:] - matrices @ target[:-1]
    tangent = nilss(lambda step, vectors: matrices[step] @ vectors, forcing, bases[0, :, :1], segment_length)
    # Compare with the known interior bounded solution, allowing boundary transients.
    np.testing.assert_allclose(tangent["values"][64:-64], target[64:-64], atol=1e-10)
    desired_adjoint = bases @ np.array([[0.6], [-0.2]])
    gradient = desired_adjoint[:-1] - matrices.transpose(0, 2, 1) @ desired_adjoint[1:]
    adjoint = nilsas(lambda step, vectors: matrices[step].T @ vectors, gradient, bases[-1, :, :1], segment_length)
    np.testing.assert_allclose(adjoint["values"][64:-64], desired_adjoint[64:-64], atol=1e-10)
    for result in (tangent, adjoint):
        assert result["homogeneous_modes"] == 1 < result["state_dimension"]
        assert result["recurrence_residual"] < 1e-10
        np.testing.assert_allclose(result["qr_log_growth"], np.log(1.7), atol=1e-13)


def test_adjoint_uses_vjp_without_tangent_or_parameter_directions():
    calls = []

    def vjp(step, vectors):
        calls.append(step)
        return 2 * vectors

    gradient = np.ones((32, 1, 1))
    result = nilsas(vjp, gradient, np.ones((1, 1)), 4)
    np.testing.assert_allclose(result["values"][8:-8], -1, atol=0.01)
    assert set(calls) == set(range(32))


def test_rejects_incomplete_segments():
    with pytest.raises(ValueError, match="complete"):
        nilss(lambda _, vectors: 2 * vectors, np.ones((7, 1, 1)), np.ones((1, 1)), 4)


def test_native_jvp_and_vjp_callbacks_match_independent_dense_products():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    matrix = jnp.array([[2.0, 1.0], [1.0, 1.0]])
    point = jnp.array([0.13, 0.21])

    def mapping(x):
        return matrix @ x

    @jax.jit
    def jvp(vectors):
        return jax.vmap(lambda vector: jax.jvp(mapping, (point,), (vector,))[1], in_axes=1, out_axes=1)(vectors)

    @jax.jit
    def vjp(vectors):
        pullback = jax.vjp(mapping, point)[1]
        return jax.vmap(lambda vector: pullback(vector)[0], in_axes=1, out_axes=1)(vectors)

    _, basis = np.linalg.eigh(np.asarray(matrix))
    forcing = np.random.default_rng(9).normal(size=(32, 2, 2))
    for solve, native, dense in (
        (nilss, jvp, np.asarray(matrix)),
        (nilsas, vjp, np.asarray(matrix).T),
    ):
        expected = solve(lambda _, vectors: dense @ vectors, forcing, basis[:, 1:], 4)
        actual = solve(lambda _, vectors: np.asarray(native(vectors)), forcing, basis[:, 1:], 4)
        np.testing.assert_allclose(actual["values"], expected["values"], atol=1e-11)
