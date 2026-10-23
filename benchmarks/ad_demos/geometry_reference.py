"""Independent native refinements of a supplied physical geometry and carry."""


class RefinedGeometry:
    """Prepare a fixed reference grid and conservatively reconstruct candidate data.

    The fine vertical levels come from the original reference grid. Candidate
    wet depths, initial profiles, histories and boundary data are reconstructed
    onto that grid. Numerical timesteps retain their native metric budgets.
    Exact physical measures define comparisons between resolutions.
    """

    def __init__(self, ocean, refinement):
        import jax.numpy as jnp
        from veros import veros_routine
        from veros.variables import Variable
        from .grid_reference import refined_horizontal_origins, refined_vertical_widths

        if refinement < 2 or int(refinement) != refinement:
            raise ValueError("Geometry references require an integer refinement >= 2")
        if not ocean.sim.state.settings.enable_partial_cells or ocean.sim.state.settings.enable_streamfunction:
            raise ValueError("Geometry references require partial cells and surface pressure")
        self.ocean, self.refinement = ocean, int(refinement)
        base, settings = ocean.initial.variables, ocean.sim.state.settings
        self.global_setup = ocean.config["setup"] == "global_4deg"
        fixed = dict(settings.items())
        fixed.update(
            nx=settings.nx * refinement,
            ny=settings.ny * refinement,
            nz=settings.nz * refinement,
            dt_tracer=settings.dt_tracer / refinement,
            dt_mom=settings.dt_mom / refinement,
            **refined_horizontal_origins(base, settings, refinement),
        )
        owner = self

        class Refined(type(ocean.sim)):
            @veros_routine
            def set_parameter(self, state):
                super().set_parameter(state)
                if owner.global_setup:
                    state.var_meta["geometry_surface_ratio"] = Variable(
                        "salinity restoring conductance ratio", None, time_dependent=False
                    )

            @veros_routine
            def set_grid(self, state):
                vs = state.variables
                units = settings.degtom if settings.coord_degree else 1
                for axis in ("x", "y"):
                    data = getattr(base, "d" + axis + "t")[2:-2] / (refinement * units)
                    setattr(vs, "d" + axis + "t", jnp.pad(jnp.repeat(data, refinement), (2, 2), mode="edge"))
                vs.dzt = refined_vertical_widths(base, refinement)

            @veros_routine
            def set_topography(self, state):
                parent = owner.horizontal(base.kbot)
                state.variables.kbot = jnp.where(parent > 0, (parent - 1) * refinement + 1, 0)

            @veros_routine
            def set_initial_conditions(self, state):
                if not owner.global_setup:
                    super().set_initial_conditions(state)
                vs = state.variables
                for name in ("temp", "salt"):
                    data = owner.horizontal(getattr(base, name))
                    setattr(vs, name, jnp.repeat(data, refinement, axis=2) * vs.maskT[..., None])
                for name in owner.prescribed(base):
                    if name in vs:
                        setattr(vs, name, owner.horizontal(getattr(base, name)))
                if owner.global_setup:
                    vs.geometry_surface_ratio = base.dzt[-1] / vs.dzt[-1]

            @veros_routine(supports_compilation=True)
            def set_forcing(self, state):
                super().set_forcing(state)
                if owner.global_setup:
                    from .research_ocean import balanced_surface_flux, YEAR

                    vs = state.variables
                    seasonal = jnp.cos(2 * jnp.pi * vs.time / YEAR)
                    research = vs.research_salt_mean + seasonal * vs.research_salt_seasonal
                    eligible = vs.maskT[..., -1] & (vs.temp[..., -1, vs.tau] >= -1.8)
                    research = balanced_surface_flux(research, eligible, vs.area_t)
                    vs.forc_salt_surface = research + vs.geometry_surface_ratio * (vs.forc_salt_surface - research)

        self.simulation = Refined(override=fixed)
        self.simulation.setup()
        import numpy as np

        for name in ("dzt", "dzw"):
            values = np.asarray(getattr(self.simulation.state.variables, name))
            if not np.all(np.isfinite(values) & (values > 0)):
                raise ValueError("Refined geometry requires positive native vertical and staggered spacings")
        self.prepared = self.simulation.prepare_integrator(record_diagnostics=(), checkpoint=True)
        self.template = self.prepared.pack(self.simulation.state)

    @staticmethod
    def prescribed(variables):
        monthly = {"sss_clim", "sst_clim", "qnec", "qnet", "taux", "tauy", "t_star", "t_rest"}
        return sorted(name for name in variables.fields() if name in monthly or name.startswith("research_"))

    def horizontal(self, values):
        """Reconstruct horizontal cell averages while retaining scalars."""
        import jax.numpy as jnp

        if values.ndim < 2:
            return values
        values = jnp.repeat(jnp.repeat(values[2:-2, 2:-2], self.refinement, axis=0), self.refinement, axis=1)
        return jnp.pad(values, ((2, 2), (2, 2)) + ((0, 0),) * (values.ndim - 2), mode="edge")

    def initial(self, carry):
        import jax.numpy as jnp
        from veros.core import eke, tke
        from veros.core.geometry import bathymetry_geometry, wgrid_thickness
        from veros.core.momentum import vertical_velocity_kernel
        from veros.core.numerics import calc_initial_conditions_kernel, calc_topo_kernel
        from veros.core.utilities import enforce_boundaries
        from veros.remapping import remap_cell_averages, remap_column_inventories
        from veros.routines import tracing_context

        source, target = carry.variables, self.template.variables
        depth = self.horizontal(source.ht)
        count = jnp.sum(depth[..., None] > -target.zw, axis=-1)
        bottom = jnp.where(depth > 0, target.maskT.shape[2] - count + 1, 0).astype(target.kbot.dtype)
        current = self.prepared.replace_variables(self.template, kbot=bottom)
        with tracing_context():
            fields = calc_topo_kernel(self.prepared._make_state(current.variables))
            current = self.prepared.replace_variables(current, **fields._asdict())
            fields = bathymetry_geometry(self.prepared._make_state(current.variables), depth)
        current = self.prepared.replace_variables(current, **fields._asdict())
        target = current.variables
        changes = {}

        def bounded(values):
            with tracing_context():
                return enforce_boundaries(values, self.simulation.state.settings.enable_cyclic_x)

        for name in self.prescribed(source):
            if name in target:
                changes[name] = (
                    bounded(self.horizontal(getattr(source, name)))
                    if getattr(source, name).ndim >= 2
                    else getattr(source, name)
                )
        old_t = self.horizontal(source.dzt * source.hfacT)
        new_t = target.dzt * target.hfacT
        for name in ("temp", "salt", "dtemp", "dsalt", "dHd"):
            if name in target and name in source:
                changes[name] = bounded(
                    remap_cell_averages(self.horizontal(getattr(source, name)), old_t, new_t, axis=2)
                )
        for name, grid, normal in (("u", "U", 0), ("v", "V", 1), ("du", "U", 0), ("dv", "V", 1)):
            old_width = source.dzt * getattr(source, "hfac" + grid)
            new_width = target.dzt * getattr(target, "hfac" + grid)
            values = getattr(source, name)
            if normal == 1:
                values = values * source.cosu[None, :, None, None]
            previous_values = values.at[2:-2, 2:-2].set(values[1:-3, 2:-2] if normal == 0 else values[2:-2, 1:-3])
            previous_width = old_width.at[2:-2, 2:-2].set(
                old_width[1:-3, 2:-2] if normal == 0 else old_width[2:-2, 1:-3]
            )
            left = remap_column_inventories(
                self.horizontal(previous_values), self.horizontal(previous_width), new_width, axis=2
            )
            right = remap_column_inventories(self.horizontal(values), self.horizontal(old_width), new_width, axis=2)
            size = target.maskT.shape[normal] - 4
            coordinate = ((jnp.arange(size) % self.refinement) + 1) / self.refinement
            coordinate = jnp.pad(coordinate, (2, 2), mode="edge")
            coordinate = coordinate.reshape((-1, 1, 1, 1) if normal == 0 else (1, -1, 1, 1))
            mapped = (1 - coordinate) * left + coordinate * right
            if normal == 1:
                mapped /= jnp.where(target.cosu != 0, target.cosu, 1)[None, :, None, None]
            changes[name] = bounded(mapped * getattr(target, "mask" + grid)[..., None])
        if "psi" in source:
            changes["psi"] = bounded(self.horizontal(source.psi))
        with tracing_context():
            old_w = wgrid_thickness(self.ocean.prepared._make_state(source))
            new_w = wgrid_thickness(self.prepared._make_state(target))
        for name in ("tke", "dtke", "eke", "deke", "K_diss_v"):
            if name not in source or name not in target:
                continue
            old, new = old_w, new_w
            if name == "K_diss_v":
                old = jnp.broadcast_to(source.dzw, source.maskW.shape).at[..., -1].multiply(0.5) * source.maskW
                new = jnp.broadcast_to(target.dzw, target.maskW.shape).at[..., -1].multiply(0.5) * target.maskW
            field = getattr(source, name)
            changes[name] = bounded(remap_column_inventories(self.horizontal(field), self.horizontal(old), new, axis=2))
        changes.update(
            time=source.time, itt=source.itt * self.refinement, tau=source.tau, taum1=source.taum1, taup1=source.taup1
        )
        if self.global_setup:
            changes["geometry_surface_ratio"] = source.dzt[-1] / target.dzt[-1]
        current = self.prepared.replace_variables(current, **changes)
        shell = self.prepared._make_state(current.variables)
        with tracing_context(), shell.variables.unlock():
            rebuild = getattr(self.simulation, "rebuild_vertical_forcing", None)
            if rebuild is not None:
                rebuild(shell)
            shell.variables.update(calc_initial_conditions_kernel(shell))
            for index in range(3):
                shell.variables.taup1 = jnp.asarray(index, source.taup1.dtype)
                shell.variables.update(vertical_velocity_kernel(shell))
            shell.variables.taup1 = source.taup1
            if shell.settings.enable_eke:
                eke.set_eke_diffusivities(shell)
            if shell.settings.enable_tke:
                tke.set_tke_diffusivities(shell)
        valid = jnp.all(jnp.isfinite(depth) & (depth >= 0) & (depth <= jnp.sum(target.dzt) + 1e-10))
        return current._replace(
            variables=shell.variables,
            failure_code=jnp.where(valid, carry.failure_code, 5),
            failed_iteration=jnp.where(valid, carry.failed_iteration, source.itt),
        )


def release(ocean):
    """Release completed fine-grid states before starting another design stage."""
    import gc
    import ctypes
    import sys
    import jax

    for name in ("_geometry_errors", "_geometry_references"):
        getattr(ocean, name, {}).clear()
    if ocean.config.get("setup") == "global_4deg":
        jax.clear_caches()
        gc.collect()
        if sys.platform == "linux":
            allocator = ctypes.CDLL(None)
            if hasattr(allocator, "malloc_trim"):
                allocator.malloc_trim(0)


def require_response(response, config=None):
    """Check provider scope before allocating a refined ocean experiment."""
    if response not in ("local", "trajectory", "equilibrium", "seasonal", "statistics"):
        raise NotImplementedError("No validated geometry reference provider for this response")
    if (
        response == "statistics"
        and config is not None
        and config.get("statistical_method", "invariant_periodic" if config["setup"] == "adjusted_basin" else None)
        != "invariant_periodic"
    ):
        raise NotImplementedError("Statistical geometry requires a declared, validated invariant_periodic method")


def evaluate(ocean, carry, *, parameters, refinement, response, steps, forward=False, seed=0):
    """Compare native finite endpoints or validated adjusted field responses."""
    import jax
    from .geometry_response import FiniteGeometryError

    require_response(response, ocean.config)
    if response == "local" and steps != 1:
        raise ValueError("Local geometry references require one coarse step")
    key = (refinement, response, steps, tuple(parameters))
    if not hasattr(ocean, "_geometry_references"):
        ocean._geometry_references = {}
        ocean._geometry_errors = {}
    if key not in ocean._geometry_errors:
        if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(carry)):
            raise ValueError("Prepare the geometry reference with a concrete carry before differentiating")
        if refinement not in ocean._geometry_references:
            ocean._geometry_references[refinement] = RefinedGeometry(ocean, refinement)
        reference = ocean._geometry_references[refinement]
        if response in ("local", "trajectory"):
            engine = FiniteGeometryError(reference, carry, parameters, steps)
        else:
            from .adjusted_geometry_reference import AdjustedGeometryError

            engine = AdjustedGeometryError(reference, carry, parameters, response, steps)
        ocean._geometry_errors[key] = engine
    engine = ocean._geometry_errors[key]
    point = engine.codec.pack((carry, parameters))
    values = engine.forward(point, seed) if response == "statistics" and forward else engine.transformed(point)(point)
    return dict(
        error=values[0],
        remap_budget_residual=values[1],
        projection_geometry_residual=values[2],
        projection_horizontal_residual=values[3],
        field_errors=values[4:8],
        operator_change_norm=values[8],
    )
