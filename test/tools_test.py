import numpy as np
import pytest

from veros import tools, veros_kernel
from veros.core.operators import numpy as npx


@pytest.mark.parametrize("month", range(12))
def test_periodic_interval_monthly_midpoints(month):
    month_length = 30 * 86400.0
    current_time = npx.array((month + 0.5) * month_length)
    records = npx.arange(12, dtype="float64")

    (n1, f1), (n2, f2) = tools.get_periodic_interval(current_time, 12 * month_length, month_length, 12)

    assert int(n1) == month
    assert float(f1) == 1.0
    assert float(f2) == 0.0
    assert float(f1 * records[n1] + f2 * records[n2]) == month


@pytest.mark.parametrize(
    "current_day, expected",
    [
        (-15.0, ((11, 1.0), (0, 0.0))),
        (0.0, ((11, 0.5), (0, 0.5))),
        (7.5, ((11, 0.25), (0, 0.75))),
        (22.5, ((0, 0.75), (1, 0.25))),
        (30.0, ((0, 0.5), (1, 0.5))),
        (352.5, ((11, 0.75), (0, 0.25))),
        (360.0, ((11, 0.5), (0, 0.5))),
        (375.0, ((0, 1.0), (1, 0.0))),
    ],
)
def test_periodic_interval_monthly_weights(current_day, expected):
    day_length = 86400.0
    actual = tools.get_periodic_interval(current_day * day_length, 360 * day_length, 30 * day_length, 12)

    for (index, weight), (expected_index, expected_weight) in zip(actual, expected):
        assert int(index) == expected_index
        assert float(weight) == pytest.approx(expected_weight)


@pytest.mark.parametrize("cycle", (-2, -1, 0, 1, 2))
@pytest.mark.parametrize("rec_spacing", (1.0, 6.0, 30 * 86400.0))
def test_periodic_interval_arbitrary_record_spacing(cycle, rec_spacing):
    # Four records centred at 0.5, 1.5, 2.5, and 3.5 record spacings.
    # At 1.25 spacings, interpolation is 25% record 0 and 75% record 1.
    current_time = (4 * cycle + 1.25) * rec_spacing
    (n1, f1), (n2, f2) = tools.get_periodic_interval(current_time, 4 * rec_spacing, rec_spacing, 4)

    np.testing.assert_allclose([n1, n2], [0, 1])
    np.testing.assert_allclose([f1, f2], [0.25, 0.75])


def test_periodic_interval_single_record():
    (n1, f1), (n2, f2) = tools.get_periodic_interval(0.0, 10.0, 10.0, 1)

    assert int(n1) == int(n2) == 0
    assert float(f1 + f2) == 1.0


@pytest.mark.parametrize("current_time, expected", ((0.0, 23.0), (0.5, 2.0), (1.0, 5.0), (4.0, 23.0), (-0.5, 44.0)))
def test_periodic_interval_in_kernel(current_time, expected):
    @veros_kernel
    def interpolate_records(current_time, records):
        (n1, f1), (n2, f2) = tools.get_periodic_interval(current_time, 4.0, 1.0, 4)
        return f1 * records[n1] + f2 * records[n2]

    records = npx.array([2.0, 8.0, 20.0, 44.0])
    assert float(interpolate_records(npx.array(current_time), records)) == pytest.approx(expected)
