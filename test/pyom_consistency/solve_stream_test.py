from veros.core import external
from veros.pyom_compat import get_random_state

from test_base import compare_state

TEST_SETTINGS = dict(
    nx=70,
    ny=60,
    nz=50,
    dt_tracer=3600,
    dt_mom=3600,
    enable_cyclic_x=True,
    enable_streamfunction=True,
)


def test_solve_streamfunction(pyom2_lib):
    vs_state, pyom_obj = get_random_state(pyom2_lib, extra_settings=TEST_SETTINGS)
    vs_state.variables.update(external.solve_streamfunction(vs_state))
    pyom_obj.solve_streamfunction()
    compare_state(vs_state, pyom_obj)
