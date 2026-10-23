"""Energy exchanges for a closed, locally divergence-free stepped basin."""

from copy import copy

import numpy as np
import pytest


@pytest.mark.parametrize(
    "option",
    (
        "enable_streamfunction",
        "enable_idemix",
        "enable_TEM_friction",
        "enable_biharmonic_friction",
        "enable_biharmonic_mixing",
    ),
)
def test_partial_energy_keeps_unimplemented_physics_guarded(option):
    from veros.settings import check_setting_conflicts
    from veros.state import get_default_state

    state = get_default_state()
    with state.settings.unlock():
        state.settings.enable_partial_cells = True
        state.settings.enable_streamfunction = False
        state.settings.enable_conserve_energy = True
        state.settings.enable_implicit_vert_friction = True
        setattr(state.settings, option, True)
    with pytest.raises(ValueError, match=option):
        check_setting_conflicts(state.settings)


@pytest.mark.parametrize("tke,eke", ((True, False), (False, True), (True, True)))
def test_partial_energy_supports_turbulence_settings(tke, eke):
    from veros import runtime_settings
    from veros.settings import check_setting_conflicts
    from veros.state import get_default_state

    state = get_default_state()
    with state.settings.unlock():
        state.settings.enable_partial_cells = True
        state.settings.enable_streamfunction = False
        state.settings.enable_conserve_energy = True
        state.settings.enable_implicit_vert_friction = True
        state.settings.enable_tke, state.settings.enable_eke = tke, eke
    previous = runtime_settings.backend
    object.__setattr__(runtime_settings, "backend", "jax")
    try:
        check_setting_conflicts(state.settings)
    finally:
        object.__setattr__(runtime_settings, "backend", previous)


def flow_state(partial):
    import jax.numpy as jnp
    from bathymetry_helpers import make_simulation
    from veros import runtime_settings
    from veros.core import momentum, numerics
    from veros.routines import tracing_context

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    bottom = 1 + (np.arange(8)[:, None] + np.arange(10)[None, :]) % 4
    simulation = make_simulation(partial_cells=partial, bottom_indices=bottom, wind=0)
    prepared = simulation.prepare_integrator(record_diagnostics=())
    initial = prepared.pack(simulation.state)
    if partial:
        depths = initial.variables.ht[2:-2, 2:-2]
        pattern = jnp.sin(jnp.arange(depths.size).reshape(depths.shape))
        initial = prepared.with_bathymetry(initial, depths - 25 - 5 * pattern)
        prepared.check_status(initial)
    state = prepared._make_state(initial.variables)
    state._settings = copy(state.settings)
    with state.settings.unlock():
        state.settings.enable_superbee_advection = False
    vs = state.variables
    fraction_u, fraction_v = (vs.hfacU, vs.hfacV) if partial else (vs.maskU, vs.maskV)
    hu, hv = jnp.sum(vs.dzt * fraction_u, axis=2), jnp.sum(vs.dzt * fraction_v, axis=2)
    psi = 1e5 * jnp.sin(jnp.pi * jnp.arange(9)[:, None] / 8) * jnp.sin(jnp.pi * jnp.arange(11)[None, :] / 10)
    psi = psi.at[0].set(0).at[-1].set(0).at[:, 0].set(0).at[:, -1].set(0)
    transport_u = psi[1:, 1:] - psi[1:, :-1]
    transport_v = -psi[1:, 1:] + psi[:-1, 1:]
    u = transport_u / (vs.dyt[None, 2:-2] * jnp.where(hu[2:-2, 2:-2] > 0, hu[2:-2, 2:-2], 1))
    v = transport_v / (vs.dxt[2:-2, None] * jnp.where(hv[2:-2, 2:-2] > 0, hv[2:-2, 2:-2], 1))
    rng = np.random.default_rng(52)
    with vs.unlock():
        vs.u = vs.u.at[2:-2, 2:-2].set(u[..., None, None] * vs.maskU[2:-2, 2:-2, :, None])
        vs.v = vs.v.at[2:-2, 2:-2].set(v[..., None, None] * vs.maskV[2:-2, 2:-2, :, None])
        vs.temp += 0.2 * jnp.asarray(rng.normal(size=vs.maskT.shape))[..., None] * vs.maskT[..., None]
        vs.salt += 0.02 * jnp.asarray(rng.normal(size=vs.maskT.shape))[..., None] * vs.maskT[..., None]
        with tracing_context():
            vs.update(numerics.calc_initial_conditions_kernel(state))
            vs.update(momentum.vertical_velocity_kernel(state))
        vs.w = jnp.broadcast_to(vs.w[..., vs.taup1, None], vs.w.shape)
    fe = vs.u[..., vs.tau] * vs.dyt[None, :, None] * vs.dzt * fraction_u
    fn = vs.v[..., vs.tau] * vs.dxt[:, None, None] * vs.dzt * fraction_v
    ft = vs.w[..., vs.tau] * vs.area_t[..., None]
    residual = fe[2:-2, 2:-2] - fe[1:-3, 2:-2] + fn[2:-2, 2:-2] - fn[2:-2, 1:-3]
    residual += ft[2:-2, 2:-2] - jnp.pad(ft[2:-2, 2:-2, :-1], ((0, 0), (0, 0), (1, 0)))
    scale = max(float(jnp.max(jnp.abs(fe))), float(jnp.max(jnp.abs(fn))), 1)
    np.testing.assert_allclose(residual, 0, atol=1e-13 * scale)
    np.testing.assert_allclose(ft[:, :, -1], 0, atol=1e-13 * scale)
    return state


@pytest.mark.parametrize("partial", (False, True))
@pytest.mark.usefixtures("jax_backend")
def test_momentum_advection_preserves_closed_basin_kinetic_energy(partial):
    import jax.numpy as jnp
    from veros.core.momentum import momentum_advection
    from veros.routines import tracing_context

    state = flow_state(partial)
    vs = state.variables
    with tracing_context():
        result = momentum_advection(state)
    work, activity = jnp.asarray(0.0), jnp.asarray(0.0)
    for name, grid, tendency in (("u", "U", result.du_adv), ("v", "V", result.dv_adv)):
        fraction = getattr(vs, "hfac" + grid) if partial else getattr(vs, "mask" + grid)
        volume = getattr(vs, "area_" + grid.lower())[..., None] * vs.dzt * fraction
        local = (getattr(vs, name)[..., vs.tau] * tendency * volume)[2:-2, 2:-2]
        work += jnp.sum(local)
        activity += jnp.sum(jnp.abs(local))
    assert float(activity) > 1e-6
    np.testing.assert_allclose(work, 0, atol=1e-12 * float(activity))


@pytest.mark.parametrize("partial", (False, True))
@pytest.mark.usefixtures("jax_backend")
def test_hydrostatic_work_balances_centered_enthalpy_advection(partial):
    import jax.numpy as jnp
    from veros.core.external.solve_pressure import prepare_forcing
    from veros.core.thermodynamics import advect_tracer
    from veros.routines import tracing_context

    state = flow_state(partial)
    vs = state.variables
    with tracing_context():
        dtemp = advect_tracer(state, vs.temp[..., vs.tau])
        dsalt = advect_tracer(state, vs.salt[..., vs.tau])
        result, _ = prepare_forcing(state)
    occupied = vs.hfacT if partial else vs.maskT
    volume = vs.area_t[..., None] * vs.dzt * occupied
    enthalpy = (
        -state.settings.grav
        / state.settings.rho_0
        * (vs.int_drhodT[..., vs.tau] * dtemp + vs.int_drhodS[..., vs.tau] * dsalt)
    )
    potential = jnp.sum((enthalpy * volume)[2:-2, 2:-2])
    kinetic = jnp.asarray(0.0)
    for name, grid, tendency in (("u", "U", result.du[..., vs.tau]), ("v", "V", result.dv[..., vs.tau])):
        fraction = getattr(vs, "hfac" + grid) if partial else getattr(vs, "mask" + grid)
        volume = getattr(vs, "area_" + grid.lower())[..., None] * vs.dzt * fraction
        kinetic += jnp.sum((getattr(vs, name)[..., vs.tau] * tendency * volume)[2:-2, 2:-2])
    assert abs(float(kinetic)) > 1e-6
    np.testing.assert_allclose(kinetic + potential, 0, atol=1e-10 * abs(float(kinetic)))


@pytest.mark.parametrize("eos", (1, 3))
@pytest.mark.usefixtures("jax_backend")
def test_native_partial_energy_records_and_mixing_work(eos):
    import jax.numpy as jnp
    from bathymetry_helpers import make_simulation
    from veros import runtime_settings
    from veros.core.numerics import calc_initial_conditions_kernel
    from veros.routines import tracing_context

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    bottom = 1 + (np.arange(8)[:, None] + np.arange(10)[None, :]) % 4
    simulation = make_simulation(
        partial_cells=True,
        bottom_indices=bottom,
        wind=0.02,
        enable_conserve_energy=True,
        eq_of_state_type=eos,
        enable_hor_friction=True,
        A_h=100.0,
        enable_ray_friction=True,
        r_ray=1e-5,
        enable_bottom_friction=True,
        r_bot=2e-5,
        enable_neutral_diffusion=True,
        enable_skew_diffusion=True,
        K_iso_0=100.0,
        K_gm_0=50.0,
    )
    energy = simulation.state.diagnostics["energy"]
    energy.sampling_frequency = energy.output_frequency = simulation.state.settings.dt_tracer
    prepared = simulation.prepare_integrator(record_diagnostics=("energy",))
    carry = prepared.pack(simulation.state)
    depths = carry.variables.ht[2:-2, 2:-2]
    pattern = jnp.sin(jnp.arange(depths.size).reshape(depths.shape))
    carry = prepared.with_bathymetry(carry, depths - 25 - 5 * pattern)
    rng = np.random.default_rng(53)
    carry = prepared.replace_variables(
        carry,
        temp=carry.variables.temp
        + 0.05
        * jnp.asarray(rng.normal(size=carry.variables.maskT.shape))[..., None]
        * carry.variables.maskT[..., None],
    )
    state = prepared._make_state(carry.variables)
    with state.variables.unlock(), tracing_context():
        state.variables.update(calc_initial_conditions_kernel(state))
    carry = carry._replace(variables=state.variables)
    for _ in range(4):
        carry, records = prepared.step(carry)
        prepared.check_status(carry)
        assert bool(records["energy"].valid)
        assert all(np.isfinite(value).all() for value in records["energy"].values.values())
    vs = carry.variables
    volume = vs.area_t[..., None] * vs.dzt * vs.hfacT
    weights = (vs.area_t[..., None] * vs.dzw * vs.maskW).at[:, :, -1].multiply(0.5)
    old, new = vs.taum1, vs.tau
    horizontal = (
        -simulation.state.settings.grav
        / simulation.state.settings.rho_0
        * (
            vs.int_drhodT[..., old] * (vs.dtemp_hmix + vs.dtemp_iso)
            + vs.int_drhodS[..., old] * (vs.dsalt_hmix + vs.dsalt_iso)
        )
    )
    vertical = (
        -simulation.state.settings.grav
        / simulation.state.settings.rho_0
        * (vs.int_drhodT[..., new] * vs.dtemp_vmix + vs.int_drhodS[..., new] * vs.dsalt_vmix)
    )
    # Mixing tendencies recover small increments from O(10) temperature and O(35)
    # salinity values. Account for the weighted resolution of those subtractions.
    input_scale = (
        volume
        * simulation.state.settings.grav
        / simulation.state.settings.rho_0
        * (
            jnp.abs(vs.int_drhodT[..., old] * vs.temp[..., old])
            + jnp.abs(vs.int_drhodS[..., old] * vs.salt[..., old])
            + jnp.abs(vs.int_drhodT[..., new] * vs.temp[..., new])
            + jnp.abs(vs.int_drhodS[..., new] * vs.salt[..., new])
        )
    )
    recovery_scale = (
        0.5 * jnp.finfo(vs.temp.dtype).eps / simulation.state.settings.dt_tracer * jnp.sum(input_scale[2:-2, 2:-2])
    )
    for work, diagnosed in (
        (horizontal, vs.P_diss_hmix + vs.P_diss_iso + vs.P_diss_skew),
        (vertical, vs.P_diss_v + vs.P_diss_nonlin),
    ):
        expected = jnp.sum((work * volume)[2:-2, 2:-2])
        actual = jnp.sum((diagnosed * weights)[2:-2, 2:-2])
        assert abs(float(expected)) > 1e-6
        assert float(recovery_scale) < 1e-5 * abs(float(expected))
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8 + float(recovery_scale))
    momentum_work = jnp.asarray(0.0)
    for name, grid, tendency in (("u", "U", vs.du_mix), ("v", "V", vs.dv_mix)):
        velocity_volume = getattr(vs, "area_" + grid.lower())[..., None] * vs.dzt * getattr(vs, "hfac" + grid)
        momentum_work -= jnp.sum((getattr(vs, name)[..., old] * tendency * velocity_volume)[2:-2, 2:-2])
    diagnosed = jnp.sum(((vs.K_diss_h + vs.K_diss_v + vs.K_diss_bot) * weights)[2:-2, 2:-2])
    assert float(momentum_work) > 0
    np.testing.assert_allclose(diagnosed, momentum_work, rtol=1e-9, atol=1e-8)


@pytest.mark.usefixtures("jax_backend")
def test_partial_energy_source_records_and_geometry_checkpoint_pullback():
    import jax
    import jax.numpy as jnp
    from bathymetry_helpers import make_simulation
    from veros import runtime_settings
    from veros.checkpointing import CheckpointedIntegrator

    object.__setattr__(runtime_settings, "diskless_mode", True)
    object.__setattr__(runtime_settings, "linear_solver", "scipy_jax")
    bottom = 1 + (np.arange(8)[:, None] + np.arange(10)[None, :]) % 4
    simulation = make_simulation(
        partial_cells=True,
        bottom_indices=bottom,
        wind=0,
        enable_conserve_energy=True,
        enable_tempsalt_sources=True,
        kappaH_0=0,
        kappaM_0=0,
        K_h=0,
    )
    energy = simulation.state.diagnostics["energy"]
    energy.sampling_frequency = energy.output_frequency = simulation.state.settings.dt_tracer
    prepared = simulation.prepare_integrator(record_diagnostics=("energy",))
    initial = prepared.pack(simulation.state)
    depths = initial.variables.ht[2:-2, 2:-2] - 30
    pattern = jnp.sin(jnp.arange(depths.size).reshape(depths.shape))
    source = (
        1 + 0.2 * jnp.sin(jnp.arange(initial.variables.maskT.size).reshape(initial.variables.maskT.shape))
    ) * initial.variables.maskT

    def controls(z):
        carry = prepared.with_bathymetry(initial, depths + z[0] * pattern)
        return prepared.replace_variables(carry, temp_source=1e-4 * z[1] * source, salt_source=-2e-5 * z[1] * source)

    def objective(records):
        return jnp.sum(records["energy"].values["dHd_sources_m"]) / 1e10

    def function(z):
        start = controls(z)
        final, records = prepared.step(start)
        volume = final.variables.area_t[..., None] * final.variables.dzt * final.variables.hfacT
        change = final.variables.Hd[..., final.variables.tau] - start.variables.Hd[..., start.variables.tau]
        work = (
            jnp.sum((change * volume)[2:-2, 2:-2])
            * simulation.state.settings.rho_0
            / simulation.state.settings.dt_tracer
            / 1e10
        )
        return jnp.stack((objective(records), work))

    point, direction = jnp.array([0.2, 0.2]), jnp.array([0.3, 0.1])
    final, records = prepared.step(controls(point))
    prepared.check_status(final)
    assert bool(records["energy"].valid)
    value, tangent = jax.jvp(function, (point,), (direction,))
    np.testing.assert_allclose(value[0], value[1], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(tangent[0], tangent[1], rtol=1e-9, atol=1e-11)
    for epsilon in (1e-2, 1e-3):
        difference = (function(point + epsilon * direction) - function(point - epsilon * direction)) / (2 * epsilon)
        np.testing.assert_allclose(difference, tangent, rtol=1e-8, atol=1e-10)
    gradient = jax.grad(lambda z: function(z)[0])(point)
    replay = CheckpointedIntegrator(prepared, record_objective=objective).value_and_grad(controls(point), n_steps=1)
    pulled = jax.vjp(controls, point)[1](replay.gradient)[0]
    np.testing.assert_allclose(pulled, gradient, rtol=1e-10, atol=1e-10)
