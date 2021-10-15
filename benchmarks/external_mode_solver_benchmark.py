from time import time

from veros.core.streamfunction import solve_streamfunction
from veros.core.streamfunction.solve_pressure import solve_pressure
from veros.setups.global_1deg.global_1deg import GlobalOneDegreeSetup
from veros.setups.global_4deg.global_4deg import GlobalFourDegreeSetup
from veros.setups.global_flexible import GlobalFlexibleResolutionSetup
import pytest
from veros.pyom_compat import load_pyom, pyom_from_state, run_pyom

from veros import logger
from veros import runtime_settings
object.__setattr__(runtime_settings, "diskless_mode", True)


@pytest.fixture(autouse=True)
def ensure_diskless():
    from veros import runtime_settings

    object.__setattr__(runtime_settings, "diskless_mode", True)

def test_external_benchmark_4deg():
    

    sim = GlobalFourDegreeSetup()
    settings = sim.state.settings
    sim.setup()
    state = sim.state
    vs = state.variables

    with settings.unlock():
        settings.enable_streamfunction = False
        settings.enable_free_surface = True
        settings.runlen = 86400 * 20
        settings.dt_mom = 600
        settings.dt_tracer = 600

    sim.run()

test_external_benchmark_4deg()
    
