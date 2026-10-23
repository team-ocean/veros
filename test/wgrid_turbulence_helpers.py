"""Small turbulence configurations for coupled partial-cell validation."""

from copy import copy

import numpy as np


def configure_turbulence(settings):
    settings.enable_tke = settings.enable_eke = settings.enable_conserve_energy = True
    settings.eq_of_state_type = 5
    settings.enable_neutral_diffusion = settings.enable_skew_diffusion = True
    settings.K_iso_0 = settings.K_iso_steep = 1000.0
    settings.iso_dslope, settings.iso_slopec = 4e-3, 1e-3
    settings.enable_hor_friction = settings.enable_hor_friction_cos_scaling = True
    settings.A_h, settings.hor_friction_cosPower = 100.0, 1
    settings.c_k, settings.c_eps, settings.alpha_tke = 0.1, 0.7, 30.0
    settings.mxl_min, settings.tke_mxl_choice = 1e-8, 2
    settings.kappaM_min, settings.kappaH_min = 2e-4, 2e-5
    settings.enable_kappaH_profile = True
    settings.enable_tke_superbee_advection = settings.enable_eke_superbee_advection = True
    settings.eke_k_max, settings.eke_c_k, settings.eke_c_eps = 1e4, 0.4, 0.5
    settings.eke_cross, settings.eke_crhin, settings.eke_lmin = 2.0, 1.0, 100.0
    settings.enable_eke_diss_surfbot, settings.eke_diss_surfbot_frac = True, 0.2
    settings.dt_mom = 150.0


def channel_simulation(*, cyclic, opened=False, public=False):
    import jax.numpy as jnp
    from bathymetry_helpers import make_simulation
    from veros import veros_routine
    from veros.state import get_default_state
    from veros.diagnostics.energy import Energy

    bottom = np.ones((8, 10), dtype=int)
    bottom[4, :] = 0
    if opened:
        bottom[4, 5] = 4
    original = make_simulation(bottom_indices=bottom, cyclic_x=cyclic, wind=1e-4)

    class TurbulentChannel(type(original)):
        @veros_routine
        def set_parameter(self, state):
            super().set_parameter(state)
            if public:
                configure_turbulence(state.settings)

        @veros_routine
        def set_grid(self, state):
            super().set_grid(state)
            state.variables.dzt = jnp.asarray([60.0, 90.0, 120.0, 150.0])

        @veros_routine(supports_compilation=True)
        def rebuild_wet_forcing(self, state):
            state.variables.forc_tke_surface = 1e-5 * state.variables.maskW[..., -1]

    simulation = TurbulentChannel()
    simulation.setup()
    if not public:
        base = simulation.state
        state = get_default_state()
        state._settings = copy(base.settings)
        with state.settings.unlock():
            configure_turbulence(state.settings)
        state.initialize_variables()
        with state.variables.unlock():
            for name, value in base.variables.items():
                if name in state.variables:
                    setattr(state.variables, name, value)
        simulation.state = state
    state = simulation.state
    diagnostic = Energy(state)
    diagnostic.initialize_variables(state)
    diagnostic.output_variables = ("tke_m", "eke_m", "dtke_m", "deke_m")
    diagnostic.sampling_frequency = diagnostic.output_frequency = state.settings.dt_tracer
    state._diagnostics = {"energy": diagnostic}
    with state.variables.unlock():
        vs = state.variables
        profile = 1 + 0.1 * jnp.arange(state.settings.nz)
        for name, scale in (("tke", 1e-3), ("eke", 0.02)):
            values = scale * profile * vs.maskW
            setattr(vs, name, jnp.broadcast_to(values[..., None], getattr(vs, name).shape))
        for name, scale in (("dtke", 1e-9), ("deke", 2e-9)):
            values = scale * jnp.sin(jnp.arange(state.settings.nz)) * vs.maskW
            setattr(vs, name, values[..., None] * jnp.asarray([1.0, -0.5, 0.25]))
        vs.K_diss_v = 2e-7 * vs.maskW
        vs.coriolis_t = jnp.full_like(vs.coriolis_t, 1e-4)
        vs.beta = jnp.full_like(vs.beta, 1e-11)
        simulation.rebuild_wet_forcing(state)
    prepared = simulation.prepare_integrator(record_diagnostics=("energy",))
    initial = prepared.pack(state)
    depth = initial.variables.ht[2:-2, 2:-2]
    reduction = jnp.where(initial.variables.kbot[2:-2, 2:-2] == 1, 42.0, 0.0)
    initial = prepared.with_bathymetry(initial, depth - reduction)
    prepared.check_status(initial)
    return simulation, prepared, initial
