"""Geometry provider scope and matrix acceptance remain separate."""

import pytest


def test_undeclared_statistical_geometry_method_stops_before_ocean_setup(monkeypatch):
    from benchmarks.ad_demos import research
    from benchmarks.ad_demos.common import Checks

    def expensive(*args, **kwargs):
        raise AssertionError("Unimplemented reference allocated an ocean experiment")

    monkeypatch.setattr(research, "ResearchOcean", expensive)
    with pytest.raises(NotImplementedError, match="validated invariant_periodic"):
        research.run("geometry_statistics", dict(mode="study", setup="global_4deg"), Checks())


@pytest.mark.parametrize("response", ("equilibrium", "seasonal", "statistics"))
@pytest.mark.usefixtures("jax_backend")
def test_adjusted_reference_dispatch_retains_forward_climate_scope(monkeypatch, response):
    from types import SimpleNamespace
    import numpy as np
    from benchmarks.ad_demos import adjusted_geometry_reference, geometry_reference

    calls = []

    class Engine:
        def __init__(self, reference, carry, parameters, kind, steps):
            calls.append(("create", reference, carry, parameters, kind, steps))
            self.codec = SimpleNamespace(pack=lambda value: value)

        def transformed(self, example):
            return lambda point: np.arange(9.0)

        def forward(self, point, seed):
            calls.append(("forward", point, seed))
            return 10 + np.arange(9.0)

    ocean = SimpleNamespace(config=dict(setup="adjusted_basin"))
    reference = object()
    monkeypatch.setattr(geometry_reference, "RefinedGeometry", lambda owner, refinement: reference)
    monkeypatch.setattr(adjusted_geometry_reference, "AdjustedGeometryError", Engine)
    carry = np.asarray([0.3])
    result = geometry_reference.evaluate(ocean, carry, parameters={}, refinement=2, response=response, steps=4)
    forward = geometry_reference.evaluate(
        ocean, carry, parameters={}, refinement=2, response=response, steps=4, forward=True, seed=71
    )
    assert calls[0] == ("create", reference, carry, {}, response, 4)
    assert result["error"] == 0 and result["operator_change_norm"] == 8
    assert forward["error"] == (10 if response == "statistics" else 0)
    assert len(calls) == (2 if response == "statistics" else 1)
    if response == "statistics":
        assert calls[1][0] == "forward" and calls[1][2] == 71
