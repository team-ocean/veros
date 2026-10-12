import numpy as np

from veros.core.operators import numpy as npx, update, at
from veros.diagnostics.energy import diagnose_kernel
from veros.state import get_default_state


def test_internal_wave_energy_tendency():
    state = get_default_state()
    with state.settings.unlock():
        state.settings.update(nx=2, ny=3, nz=2, enable_idemix=True, dt_tracer=2)
    state.initialize_variables()

    with state.variables.unlock():
        vs = state.variables
        vs.area_t = npx.full_like(vs.area_t, 3.0)
        vs.dzw = npx.full_like(vs.dzw, 2.0)
        vs.maskW = npx.ones_like(vs.maskW)
        vs.E_iw = update(vs.E_iw, at[..., vs.tau], 1.0)
        vs.E_iw = update(vs.E_iw, at[..., vs.taup1], 5.0)

    energies = diagnose_kernel(state)

    np.testing.assert_allclose(energies.iw_m, 18 * (2 + 1))
    np.testing.assert_allclose(energies.diw_m, 18 * (2 + 1) * (5 - 1) / 2)
