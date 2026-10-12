"""Ocean controls and physical observation operators for the capability matrix."""

import numpy as np

from .common import CP

YEAR = 360 * 86400.0
CONTROL_NAMES = (
    "initial_temperature",
    "initial_salinity",
    "heat_mean",
    "heat_seasonal",
    "wind_mean",
    "wind_seasonal",
    "salt_mean",
    "salt_seasonal",
    "isopycnal_mixing",
    "vertical_mixing",
    "closure_scale",
    "closure_state",
    "sill_depth",
    "basin_depth",
    "vertical_grid",
    "channel_opening",
)


def closure_law(variables, weights):
    """Return positive diffusivity in m²/s."""
    import jax.numpy as jnp

    return 1e-5 * jnp.exp(weights[0] + weights[1] * jnp.tanh(variables.Nsqr[..., variables.tau] / 1e-5))


class ResearchOcean:
    def __init__(self, config, case):
        import jax.numpy as jnp
        from veros import veros_routine
        from veros.variables import Variable
        from veros.core.operators import safe_sqrt

        self.config, self.case = config, case
        if config["setup"] == "global_4deg":
            from veros.setups.global_4deg import GlobalFourDegreeSetup as Setup

            overrides = {}
        else:
            from veros.setups.acc_basic import ACCBasicSetup as Setup

            overrides = dict(nx=8, ny=8, dt_tracer=600.0, dt_mom=600.0)
        if config["physics"] == "fixed":
            overrides.update(enable_tke=False, enable_eke=False, enable_idemix=False, kappaH_0=1e-5)
        if case.task == "geometry":
            overrides.update(enable_partial_cells=True, enable_streamfunction=False)
        fields = {
            "research_heat_mean": ("W/m^2", ("xt", "yt")),
            "research_heat_seasonal": ("W/m^2", ("xt", "yt")),
            "research_wind_mean": ("N/m^2", ("xu", "yt")),
            "research_wind_seasonal": ("N/m^2", ("xu", "yt")),
            "research_salt_mean": ("g/kg m/s", ("xt", "yt")),
            "research_salt_seasonal": ("g/kg m/s", ("xt", "yt")),
            "research_heat_input": ("J", None),
            "research_salt_input": ("g/kg m^3", None),
            "research_closure_scale": ("1", None),
            "research_closure_state": ("1", None),
        }

        class Simulation(Setup):
            @veros_routine
            def set_parameter(self, state):
                super().set_parameter(state)
                for name, (units, dims) in fields.items():
                    state.var_meta[name] = Variable(name, dims, units)

            @veros_routine(supports_compilation=True)
            def set_forcing(self, state):
                vs = state.variables
                time = vs.time
                if case.response == "equilibrium":
                    vs.time = jnp.zeros_like(time)
                super().set_forcing(state)
                phase = 2 * jnp.pi * vs.time / YEAR
                # cos is nonzero at phase zero, including in local seasonal-control tests.
                seasonal = jnp.cos(phase)
                vs.time = time
                heat = (vs.research_heat_mean + seasonal * vs.research_heat_seasonal) * vs.maskT[..., -1]
                salt = (vs.research_salt_mean + seasonal * vs.research_salt_seasonal) * vs.maskT[..., -1]
                if config["setup"] == "global_4deg":
                    blocked = (vs.temp[..., -1, vs.tau] < -1.8) & (
                        vs.forc_temp_surface + heat / (CP * state.settings.rho_0) < 0
                    )
                    heat, salt = jnp.where(blocked, 0, heat), jnp.where(blocked, 0, salt)
                vs.forc_temp_surface = vs.forc_temp_surface + heat / (CP * state.settings.rho_0)
                vs.forc_salt_surface = vs.forc_salt_surface + salt
                vs.surface_taux = (
                    vs.surface_taux + (vs.research_wind_mean + seasonal * vs.research_wind_seasonal) * vs.maskU[..., -1]
                )
                if state.settings.enable_tke:
                    taux = 0.5 * (vs.surface_taux[1:-1, 1:-1] + vs.surface_taux[:-2, 1:-1])
                    tauy = 0.5 * (vs.surface_tauy[1:-1, 1:-1] + vs.surface_tauy[1:-1, :-2])
                    vs.forc_tke_surface = vs.forc_tke_surface.at[1:-1, 1:-1].set(
                        safe_sqrt((taux / state.settings.rho_0) ** 2 + (tauy / state.settings.rho_0) ** 2) ** 1.5
                    )
                vs.research_heat_input += state.settings.dt_tracer * jnp.sum(heat[2:-2, 2:-2] * vs.area_t[2:-2, 2:-2])
                vs.research_salt_input += state.settings.dt_tracer * jnp.sum(salt[2:-2, 2:-2] * vs.area_t[2:-2, 2:-2])

            @veros_routine
            def set_diagnostics(self, state):
                for diagnostic in state.diagnostics.values():
                    diagnostic.output_frequency = diagnostic.sampling_frequency = None

        self.sim = Simulation(override=overrides)
        self.sim.setup()
        self.prepared = self.sim.prepare_integrator(record_diagnostics=(), checkpoint=True)
        self.initial = self.prepared.pack(self.sim.state)
        if case.task == "geometry":
            lower, upper = self.prepared.bathymetry_bounds(self.initial)
            self.initial = self.prepared.with_bathymetry(self.initial, (lower + upper) / 2)
        self.dt = self.sim.state.settings.dt_tracer
        preparation_steps = config["spinup_steps"] or (12 if case.response == "local" else 0)
        if preparation_steps:
            self.initial, _ = self.prepared.advance(self.initial, n_steps=preparation_steps, final_only=True)
            self.prepared.check_status(self.initial)
        vs = self.initial.variables
        longitude = (vs.xt[2:-2] + 180) % 360 - 180
        latitude = vs.yt[2:-2]
        area = vs.area_t[2:-2, 2:-2] * vs.maskT[2:-2, 2:-2, -1]
        self.area = area
        if config["setup"] == "global_4deg":
            self.north = (
                (longitude[:, None] > -80)
                & (longitude[:, None] < 20)
                & ((latitude[None, :] > 15) & (latitude[None, :] < 65))
            )
            self.south = jnp.broadcast_to((latitude[None, :] > -65) & (latitude[None, :] < -30), area.shape)
            self.section = int(np.argmin(np.abs(np.asarray(vs.yu) - 26)))
            longitude_u = (np.asarray(vs.xu) + 180) % 360 - 180
            self.drake = int(np.argmin(np.abs(longitude_u + 68)))
            self.atlantic = (longitude > -80) & (longitude < 20)
            self.drake_latitudes = (latitude > -63) & (latitude < -53)
            sill = (jnp.abs(longitude[:, None] + 68) < 6) & self.drake_latitudes[None, :]
        else:
            self.north = jnp.broadcast_to(latitude[None, :] > jnp.mean(latitude), area.shape)
            self.south = ~self.north
            self.section, self.drake = vs.v.shape[1] // 2, vs.u.shape[0] // 2
            self.atlantic, self.drake_latitudes = jnp.ones_like(longitude, bool), jnp.ones_like(latitude, bool)
            sill = jnp.broadcast_to((jnp.abs(jnp.arange(area.shape[0]) + 2 - self.drake) <= 1)[:, None], area.shape)
        self.depth_patterns = jnp.stack((sill.astype(vs.temp.dtype), self.north.astype(vs.temp.dtype)))
        phase = (latitude - jnp.min(latitude)) / (jnp.max(latitude) - jnp.min(latitude))
        heat = self.north.astype(vs.temp.dtype)
        salt = jnp.broadcast_to(jnp.cos(jnp.pi * phase)[None, :], area.shape)
        seasonal = jnp.broadcast_to(jnp.cos(2 * jnp.pi * phase)[None, :], area.shape)
        self.patterns = jnp.stack([self.balance(pattern) for pattern in (heat, seasonal, salt)])
        wind = self.south * vs.maskU[2:-2, 2:-2, -1]
        self.wind = wind / jnp.maximum(jnp.max(wind), 1)
        self.nz = self.sim.state.settings.nz
        # Profiles use observation uncertainties. Scalar entries have fixed physical scales.
        self.heat_index, self.heat_transport_index, self.acc_index, self.amoc_index = range(
            4 * self.nz, 4 * self.nz + 4
        )
        self.observation_scales = jnp.concatenate(
            (
                jnp.tile(jnp.concatenate((jnp.full(self.nz, 0.05), jnp.full(self.nz, 0.02))), 2),
                jnp.array([1e21, 1e14, 1e6, 1e6]),
            )
        )

    def balance(self, pattern):
        import jax.numpy as jnp

        pattern = pattern - jnp.sum(pattern * self.area) / jnp.sum(self.area)
        pattern *= self.initial.variables.maskT[2:-2, 2:-2, -1]
        return pattern / jnp.maximum(jnp.sqrt(jnp.sum(pattern**2 * self.area) / jnp.sum(self.area)), 1e-12)

    @staticmethod
    def fraction(variables, grid):
        name = "hfac" + grid
        return getattr(variables, name) if name in variables else getattr(variables, "mask" + grid)

    @staticmethod
    def volume(carry):
        vs = carry.variables
        return vs.area_t[2:-2, 2:-2, None] * vs.dzt[None, None, :] * ResearchOcean.fraction(vs, "T")[2:-2, 2:-2]

    @staticmethod
    def upper_fraction(variables, fraction):
        import jax.numpy as jnp

        thickness = variables.dzt * fraction
        top = variables.zt + variables.dzt / 2
        return jnp.clip(top + 1000, 0, thickness) / jnp.maximum(thickness, 1e-20)

    def controls(self, full, *, phase_days=0, realized=False):
        import jax.numpy as jnp
        from veros.core.numerics import calc_initial_conditions_kernel
        from veros.core.utilities import enforce_boundaries
        from veros.routines import tracing_context

        vs = self.initial.variables
        carry = self.initial
        if self.case.task == "geometry":
            widths = vs.dzt * jnp.exp(0.15 * full[14] * jnp.linspace(-1, 1, self.nz))
            widths *= jnp.sum(vs.dzt) / jnp.sum(widths)
            carry = self.prepared.with_vertical_grid(carry, widths=widths, conserve=("temp", "salt", "momentum"))
            lower, upper = self.prepared.bathymetry_bounds(carry)
            room = upper - lower
            mode = self.depth_patterns
            # Compensate redistribution with the actual depth-change volume weights.
            weight = self.area * room
            mode = mode - jnp.sum(mode * weight, axis=(1, 2))[:, None, None] / jnp.maximum(jnp.sum(weight), 1)
            mode /= jnp.maximum(jnp.max(jnp.abs(mode), axis=(1, 2))[:, None, None], 1)
            depth = (upper + lower) / 2 + 0.1 * room * jnp.einsum("k,kij->ij", full[12:14], mode)
            carry = self.prepared.with_bathymetry(carry, depth)
            carry = self.prepared.with_wet_geometry(
                carry,
                opening_m=100 * jnp.exp(0.5 * full[15]),
                channel="Panama" if self.config["setup"] == "global_4deg" else "sill",
                realized=realized,
                conserve=("temp", "salt", "momentum"),
            )
        vs = carry.variables
        interior_shift = self.patterns[0, ..., None] * jnp.exp(vs.zt / 300) * vs.maskT[2:-2, 2:-2]
        volume = self.volume(carry)
        level_mean = jnp.sum(interior_shift * volume, axis=(0, 1)) / jnp.maximum(jnp.sum(volume, axis=(0, 1)), 1)
        interior_shift = (interior_shift - level_mean) * vs.maskT[2:-2, 2:-2]
        shift = jnp.zeros_like(vs.maskT, dtype=vs.temp.dtype).at[2:-2, 2:-2].set(interior_shift)
        changes = dict(
            temp=vs.temp + full[0] * shift[..., None],
            salt=vs.salt + 0.1 * full[1] * shift[..., None],
            time=vs.time + phase_days * 86400,
            research_heat_input=jnp.zeros_like(vs.research_heat_input),
            research_salt_input=jnp.zeros_like(vs.research_salt_input),
            research_closure_scale=full[10],
            research_closure_state=full[11],
        )
        for name, scale, index, pattern in (
            ("heat_mean", 20, 2, self.patterns[0]),
            ("heat_seasonal", 20, 3, self.patterns[1]),
            ("wind_mean", 0.02, 4, self.wind),
            ("wind_seasonal", 0.02, 5, self.wind),
            ("salt_mean", 3.5e-6, 6, self.patterns[2]),
            ("salt_seasonal", 3.5e-6, 7, self.patterns[1]),
        ):
            field = jnp.zeros_like(getattr(vs, "research_" + name)).at[2:-2, 2:-2].set(scale * full[index] * pattern)
            changes["research_" + name] = enforce_boundaries(field, self.sim.state.settings.enable_cyclic_x)
        carry = self.prepared.replace_variables(carry, **changes)
        state = self.prepared._make_state(carry.variables)
        with tracing_context(), state.variables.unlock():
            state.variables.update(calc_initial_conditions_kernel(state))
        carry = carry._replace(variables=state.variables)
        if self.case.task == "closure":
            carry = self.prepared.with_tracer_closure(carry, law=closure_law, weights=full[10:12])
        vertical_name = "c_k" if self.sim.state.settings.enable_tke else "kappaH_0"
        parameters = {
            "K_iso_0": jnp.asarray(self.sim.state.settings.K_iso_0) * jnp.exp(full[8]),
            vertical_name: jnp.asarray(getattr(self.sim.state.settings, vertical_name)) * jnp.exp(full[9]),
        }
        return carry, parameters

    def observe(self, carry):
        import jax.numpy as jnp

        vs = carry.variables
        volume = self.volume(carry)
        temperature, salinity = vs.temp[..., vs.tau], vs.salt[..., vs.tau]
        features = []
        for region in (self.north, self.south):
            weights = volume * region[..., None]
            denominator = jnp.maximum(jnp.sum(weights, axis=(0, 1)), 1)
            for tracer in (temperature, salinity):
                features.append(jnp.sum(tracer[2:-2, 2:-2] * weights, axis=(0, 1)) / denominator)
        upper = self.upper_fraction(vs, self.fraction(vs, "T")[2:-2, 2:-2])
        heat = (
            CP
            * self.sim.state.settings.rho_0
            * jnp.sum(temperature[2:-2, 2:-2] * volume * self.north[..., None] * upper)
        )
        section = self.section
        face_area = vs.dxt[2:-2, None] * vs.cosu[section] * vs.dzt * self.fraction(vs, "V")[2:-2, section]
        volume_flux = vs.v[2:-2, section, :, vs.tau] * face_area
        # Add the vertically differentiated meridional GM streamfunction.
        if self.sim.state.settings.enable_skew_diffusion:
            b = vs.B1_gm[2:-2, section]
            volume_flux += jnp.concatenate((b[:, :1], jnp.diff(b, axis=-1)), axis=-1) * (
                vs.dxt[2:-2, None] * vs.cosu[section] * vs.maskV[2:-2, section]
            )
        face_temp = 0.5 * (temperature[2:-2, section] + temperature[2:-2, section + 1])
        heat_transport = CP * self.sim.state.settings.rho_0 * jnp.sum(volume_flux * face_temp * self.atlantic[:, None])
        upper_v = self.upper_fraction(vs, self.fraction(vs, "V")[2:-2, section])
        amoc = -jnp.sum(volume_flux * (1 - upper_v) * self.atlantic[:, None])
        drake_area = vs.dyt[2:-2, None] * vs.dzt * self.fraction(vs, "U")[self.drake, 2:-2]
        acc = jnp.sum(vs.u[self.drake, 2:-2, :, vs.tau] * drake_area * self.drake_latitudes[:, None])
        return jnp.concatenate((*features, jnp.stack((heat, heat_transport, acc, amoc)))) / self.observation_scales

    def advance(self, full, steps, *, phase_days=0, realized=False):
        carry, parameters = self.controls(full, phase_days=phase_days, realized=realized)
        carry, _ = self.prepared.advance(carry, n_steps=steps, parameters=parameters, final_only=True)
        return carry

    def check_controls(self, full, checks):
        import jax.numpy as jnp

        carry, _ = self.controls(full)
        self.prepared.check_status(carry)
        vs = carry.variables
        area = vs.area_t[2:-2, 2:-2] * vs.maskT[2:-2, 2:-2, -1]
        for name, scale in (("heat", 20), ("salt", 3.5e-6)):
            for harmonic in ("mean", "seasonal"):
                field = getattr(vs, f"research_{name}_{harmonic}")[2:-2, 2:-2]
                checks.close(f"prescribed {name} {harmonic} budget", jnp.sum(field * area) / jnp.sum(area) / scale, 0)
        checks.require("positive salinity", bool(jnp.all(vs.salt >= 0)), float(jnp.min(vs.salt)))
        if self.case.task != "geometry":
            volume = self.volume(carry)
            for name in ("temp", "salt"):
                before = getattr(self.initial.variables, name)[2:-2, 2:-2, :, self.initial.variables.tau]
                after = getattr(vs, name)[2:-2, 2:-2, :, vs.tau]
                checks.close("initial " + name + " inventory", jnp.sum((after - before) * volume) / jnp.sum(volume), 0)
        if self.case.task == "closure":
            diffusivity = closure_law(vs, full[10:12])
            checks.require(
                "positive finite closure",
                bool(jnp.all(jnp.isfinite(diffusivity) & (diffusivity > 0))),
                dict(minimum=float(jnp.min(diffusivity)), maximum=float(jnp.max(diffusivity))),
            )
