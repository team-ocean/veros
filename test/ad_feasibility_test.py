"""Conservative reference projection and statistical feasibility controls."""

import numpy as np
import pytest

from benchmarks.ad_demos.feasibility import conservative_project, expanding_shadowing, sampling_summary


def test_nested_projection_preserves_inventory_and_dry_cells():
    rng = np.random.default_rng(8)
    volume = rng.uniform(1, 3, (4, 6, 8))
    volume[:2, :2, :2] = 0
    values = rng.normal(size=volume.shape)
    projected, coarse_volume = conservative_project(values, volume, (2, 3, 4))
    assert projected[0, 0, 0] == coarse_volume[0, 0, 0] == 0
    np.testing.assert_allclose(np.sum(projected * coarse_volume), np.sum(values * volume), rtol=1e-12)
    np.testing.assert_allclose(np.sum(coarse_volume), np.sum(volume), rtol=1e-12)
    with pytest.raises(ValueError, match="aligned"):
        conservative_project(values, volume, (3, 3, 4))


def test_shadowing_transpose_can_agree_while_full_response_is_biased():
    result = expanding_shadowing(8192)
    np.testing.assert_allclose(result["tangent_estimate"], result["adjoint_estimate"], rtol=1e-12)
    assert result["continuity_residual"] < 1e-10
    assert abs(result["tangent_estimate"] - 0.25) < 0.02
    assert abs(result["tangent_estimate"] - result["exact_full_response"]) > 0.2


def test_sampling_accounts_for_autocorrelation():
    rng = np.random.default_rng(29)
    x, samples = 0, []
    for _ in range(16384):
        x = 0.8 * x + rng.normal()
        samples.append(x)
    result = sampling_summary(samples)
    assert 6 < result["correlation_time_steps"] < 12
    assert result["effective_sample_size"] < len(samples) / 6
    assert result["standard_error"] > 0
