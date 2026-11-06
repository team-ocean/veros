import numpy as np

from veros import veros_routine
from veros.core.operators import numpy as npx, update, at
from veros.diagnostics.energy import Energy, diagnose_kernel
from veros.setups.acc import ACCSetup
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


class EnergySamplingSetup(ACCSetup):
    @veros_routine
    def set_diagnostics(self, state):
        state.diagnostics.clear()
        energy = state.diagnostics["energy"] = Energy(state)
        energy.sampling_frequency = state.settings.dt_tracer

    @veros_routine
    def after_timestep(self, state):
        vs = state.variables
        # Independently integrate dE/dt over wet W cells before indices rotate.
        volume = (
            np.asarray(vs.area_t[2:-2, 2:-2, None])
            * np.asarray(vs.dzw[None, None, :])
            * np.asarray(vs.maskW[2:-2, 2:-2, :])
        )
        volume[:, :, -1] *= 0.5
        for name, dt in (
            ("eke", state.settings.dt_tracer),
            ("tke", state.settings.dt_mom),
            ("E_iw", state.settings.dt_tracer),
        ):
            field = np.asarray(vs.get(name))[2:-2, 2:-2, :]
            tendency = np.sum(volume * (field[..., vs.taup1] - field[..., vs.tau])) / dt
            self.expected_tendencies[name] += tendency


def test_energy_samples_before_rotation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    simulation = EnergySamplingSetup(override=dict(nx=12, ny=12, enable_idemix=True, restart_output_filename=None))
    simulation.expected_tendencies = dict(eke=0.0, tke=0.0, E_iw=0.0)
    simulation.setup()
    assert simulation.state.settings.dt_mom != simulation.state.settings.dt_tracer

    for step in range(6):
        simulation.step(simulation.state)
        energy_vs = simulation.state.diagnostics["energy"].variables
        assert energy_vs.nitts == step + 1
        for field, diagnostic in (("eke", "deke_m"), ("tke", "dtke_m"), ("E_iw", "diw_m")):
            np.testing.assert_allclose(energy_vs.get(diagnostic), simulation.expected_tendencies[field], rtol=1e-12)

    assert all(value != 0.0 for value in simulation.expected_tendencies.values())
