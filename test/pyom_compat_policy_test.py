"""State transfers must reject enabled physics absent from PyOM."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "option", ("enable_partial_cells", "enable_neutral_nonstable_zero_slope", "enable_tke_positive_diffusion")
)
def test_pyom_rejects_unsupported_physics_before_touching_library(option):
    from veros.pyom_compat import pyom_from_state

    state = SimpleNamespace(settings={option: True})
    with pytest.raises(ValueError, match=option):
        pyom_from_state(state, None)
