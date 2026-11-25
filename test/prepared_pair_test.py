"""Check a paired advance from independently audited initial wet columns."""

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def prepared_pair(pytestconfig):
    if pytestconfig.getoption("--backend") != "jax":
        pytest.skip("requires the JAX backend")
    jax = pytest.importorskip("jax")
    from benchmarks.ad_demos.prepared_pair import Column, Side, WetFace

    jax.config.update("jax_enable_x64", True)
    data = json.loads((Path(__file__).parent / "reference_data/prepared_pair.json").read_text())
    columns = [
        Column(**{key: np.array(value) if isinstance(value, list) else value for key, value in fields.items()})
        for fields in data["columns"]
    ]
    faces = []
    for fields in data["faces"]:
        sides = tuple(
            Side(
                side["column"],
                side["offset"],
                None if side["column"] is not None else np.array(side["exterior_viscosity"]),
            )
            for side in fields["sides"]
        )
        faces.append(
            WetFace(
                np.array(fields["velocity"]),
                np.array(fields["thickness"]),
                np.array(fields["spacing"]),
                fields["area"],
                sides,
            )
        )
    return columns, faces, data


def test_paired_advance_from_initial_state(prepared_pair):
    from benchmarks.ad_demos.prepared_pair import solve

    columns, faces, data = prepared_pair
    result = solve(columns, faces, data["settings"], *data["clocks"])
    for name, stage in (("predictor", "early_work"), ("corrected", "full_work")):
        work = result[stage]
        assert abs(np.sum(work["assigned"]) + work["exterior"] - work["union"]) <= np.sum(work["bounds"])
        for actual, expected in zip(result[name], data["expected"][name]):
            assert len(actual["trials"]) <= 16
            assert np.all(actual["inputs"]["source_schedule"] >= actual["inputs"]["eke_rate"])
            assert abs(actual["output"]["work"] - expected["work"]) <= 0.01
            for field in ("cn_delta", "end", "Nnew"):
                np.testing.assert_allclose(actual["output"][field], expected[field], rtol=1e-11, atol=1e-16)
            np.testing.assert_allclose(actual["eddy_energy"], expected["eddy_energy"], rtol=1e-11, atol=1e-16)


def test_prospective_advance_preserves_independent_work(prepared_pair):
    from benchmarks.ad_demos.prepared_pair import solve_prospective

    columns, faces, data = prepared_pair
    result = solve_prospective(columns, faces, data["settings"], *data["clocks"])
    # Independent pressure-dependent density work and staggered kinetic loss
    # for the archived ocean state in the reference fixture, in joules.
    mixing = [181031097663.6204, 144897840940.85666]
    friction = [42632796326.84349, 1465235743.0070345]
    actual = [step["output"]["work"] for step in result["corrected"]]
    np.testing.assert_allclose(actual, mixing, rtol=0, atol=0.01)
    np.testing.assert_allclose(result["full_work"]["assigned"], friction, rtol=0, atol=0.01)
    for column, step, schedule in zip(result["columns"], result["corrected"], result["schedule"], strict=True):
        assert schedule.shape == (8, len(column.energy))
        assert np.all(schedule >= 0)
        np.testing.assert_array_equal(column.energy, step["output"]["end"])
        np.testing.assert_array_equal(column.preceding_shear, step["inputs"]["outgoing_shear"])
