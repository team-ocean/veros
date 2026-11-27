"""Full invariant responses must include terms absent from bounded shadowing."""

import numpy as np
import pytest

from benchmarks.ad_demos.chaotic_response import (
    doubling_transfer,
    expanding_density_response,
    expanding_shadowing_response,
    expanding_stationary_observations,
)


def test_transfer_preserves_mass_and_handles_nyquist_without_doubling_amplitude():
    size = 64
    x = np.arange(size) / size
    for frequency in (2, 8, size // 2):
        np.testing.assert_allclose(
            doubling_transfer(np.cos(2 * np.pi * frequency * x)), np.cos(np.pi * frequency * x), atol=1e-13
        )
    np.testing.assert_allclose(doubling_transfer(np.ones(size)), 1, atol=1e-14)
    np.testing.assert_allclose(doubling_transfer(np.sin(2 * np.pi * 3 * x)), 0, atol=1e-13)


@pytest.mark.parametrize("size", [64, 128, 256])
def test_full_density_response_resolves_nonzero_analytic_cases_and_adjoint(size):
    result = expanding_density_response(size)
    # A forcing of frequency 2 creates density -cos(2*pi*x). Frequency 4
    # creates -2*cos(2*pi*x)-2*cos(4*pi*x). Frequency 1 has no density response.
    expected = [[0, -0.5, -1], [0, 0, -1], [0, 0, 0]]
    np.testing.assert_allclose(result["gradient"], expected, atol=2e-13)
    np.testing.assert_allclose(result["adjoint_gradient"], expected, atol=2e-13)
    assert max(result[name] for name in ("residual", "adjoint_residual", "mass_error")) < 1e-12


def test_nonzero_full_response_and_zero_shadowing_can_disagree_despite_transpose_agreement():
    shadowing = expanding_shadowing_response(32768)
    np.testing.assert_allclose(shadowing["gradient"], shadowing["adjoint_gradient"], atol=1e-12)
    assert shadowing["continuity_residual"] < 1e-12
    assert max(item["maximum_difference"] for item in shadowing["segmented"]) < 1e-9
    assert abs(shadowing["gradient"][0, 1]) < 0.03
    assert abs(expanding_density_response()["gradient"][0, 1] + 0.5) < 1e-12


@pytest.mark.parametrize("harmonic,expected", [(2, -0.5), (4, -1)])
def test_nonlinear_inverse_branch_density_resolves_the_nonzero_response(harmonic, expected):
    epsilon = 1e-3
    plus, minus = [expanding_stationary_observations(sign * epsilon, harmonic) for sign in (1, -1)]
    for result in (plus, minus):
        assert max(result[key] for key in ("residual", "mass_error", "inverse_residual")) < 1e-12
        assert result["minimum_density"] > 0
    difference = (plus["observations"][0] - minus["observations"][0]) / (2 * epsilon)
    np.testing.assert_allclose(difference, expected, atol=1e-5)
