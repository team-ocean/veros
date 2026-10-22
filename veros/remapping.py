"""Conservative vertical remapping of cell averages and timestep histories."""


def remap_cell_averages(values, old_thickness, new_thickness, *, axis=-1):
    """Integrate a continuous quadratic reconstruction over the new cells.

    Columns have the same total wet depth, ordered from bottom to surface.
    Zero thickness denotes a dry cell. Shared face values make the cumulative
    inventory continuously differentiable when a new face crosses an old face.
    The Bernstein coefficients retain the sign of sign-definite cell averages.
    Extra trailing dimensions, such as timestep histories, remap independently.
    """
    import jax.numpy as jnp

    values = jnp.moveaxis(values, axis, -1)
    if old_thickness.shape[:-1] != new_thickness.shape[:-1] or old_thickness.shape[-1] != values.shape[-1]:
        raise ValueError("Thickness columns and the old cell-average level count must match")
    extra = values.ndim - old_thickness.ndim
    if extra < 0 or values.shape[: old_thickness.ndim - 1] != old_thickness.shape[:-1]:
        raise ValueError("Thickness must match the cell-average columns")
    columns = old_thickness.shape[:-1] + (1,) * extra
    old = old_thickness.reshape(columns + (old_thickness.shape[-1],))
    new = new_thickness.reshape(columns + (new_thickness.shape[-1],))
    wet = old > 0
    left, right = values[..., :-1], values[..., 1:]
    face = jnp.where(
        left * right > 0,
        jnp.sign(left)
        * jnp.minimum(0.5 * (jnp.abs(left) + jnp.abs(right)), 1.5 * jnp.minimum(jnp.abs(left), jnp.abs(right))),
        0,
    )
    face = jnp.where(wet[..., :-1], jnp.where(wet[..., 1:], face, left), right)
    lower = jnp.concatenate((values[..., :1], face), axis=-1)
    upper = jnp.concatenate((face, values[..., -1:]), axis=-1)
    middle = 3 * values - lower - upper
    old_edges = jnp.concatenate((jnp.zeros_like(old[..., :1]), jnp.cumsum(old, axis=-1)), axis=-1)
    new_edges = jnp.concatenate((jnp.zeros_like(new[..., :1]), jnp.cumsum(new, axis=-1)), axis=-1)
    position = jnp.clip(new_edges, 0, old_edges[..., -1:])
    containing = wet[..., None, :] & (position[..., :, None] >= old_edges[..., None, :-1])
    index = jnp.maximum(jnp.max(jnp.where(containing, jnp.arange(old.shape[-1]), -1), axis=-1), 0)

    def select(array):
        return jnp.take_along_axis(array, index, axis=-1)

    width = select(old)
    coordinate = (position - select(old_edges[..., :-1])) / jnp.where(width > 0, width, 1)
    prefix = jnp.concatenate((jnp.zeros_like(values[..., :1]), jnp.cumsum(values * old, axis=-1)), axis=-1)
    lo, mid, hi = select(lower), select(middle), select(upper)
    # Evaluate one continuous primitive. Clipping each cell separately can
    # split coincident-face tangents inconsistently after floating-point division.
    integral = select(prefix[..., :-1]) + width * (
        lo * coordinate + (mid - lo) * coordinate**2 + (lo - 2 * mid + hi) * coordinate**3 / 3
    )
    inventory = jnp.diff(integral, axis=-1)
    result = jnp.where(new > 0, inventory / jnp.where(new > 0, new, 1), 0)
    return jnp.moveaxis(result, -1, axis)


def remap_column_inventories(values, old_thickness, new_thickness, *, axis=-1):
    """Stretch columns in relative depth and retain their original inventories.

    Volume increases dilute concentrations. Decreases concentrate them.
    The policy does not supply tracers to newly wetted columns. Geometry with
    changed horizontal connectivity requires a separate redistribution policy.
    """
    import jax.numpy as jnp

    old_depth, new_depth = jnp.sum(old_thickness, axis=-1), jnp.sum(new_thickness, axis=-1)
    old_unit = old_thickness / jnp.where(old_depth > 0, old_depth, 1)[..., None]
    new_unit = new_thickness / jnp.where(new_depth > 0, new_depth, 1)[..., None]
    mapped = remap_cell_averages(values, old_unit, new_unit, axis=axis)
    mapped = jnp.moveaxis(mapped, axis, -1)
    ratio = old_depth / jnp.where(new_depth > 0, new_depth, 1)
    mapped *= ratio.reshape(ratio.shape + (1,) * (mapped.ndim - ratio.ndim))
    return jnp.moveaxis(mapped, -1, axis)


def remap_connected_inventories(values, old_thickness, new_thickness, area):
    """Conserve domain inventories when wet columns appear or disappear.

    Existing columns stretch in relative depth. New columns receive the
    retained domain's volume-weighted mean, with proportional dilution of the
    retained fields. Inventories of removed columns spread over the target
    volume. This linear pooling policy preserves nonnegative fields. Arrays
    include two halo cells. Trailing history axes are independent.
    """
    import jax.numpy as jnp

    mapped = remap_column_inventories(values, old_thickness, new_thickness, axis=2)
    extra = (1,) * (values.ndim - 3)
    old_volume = area[..., None] * old_thickness
    new_volume = area[..., None] * new_thickness
    interior = jnp.zeros_like(area, dtype=bool).at[2:-2, 2:-2].set(True)
    old_volume *= interior[..., None]
    new_volume *= interior[..., None]
    old_wet = jnp.sum(old_thickness, axis=2) > 0
    new_wet = jnp.sum(new_thickness, axis=2) > 0
    retained_volume = jnp.sum(jnp.where(old_wet[..., None], new_volume, 0))
    total_volume = jnp.sum(new_volume)
    retained = jnp.sum(values * (old_volume * new_wet[..., None]).reshape(old_volume.shape + extra), axis=(0, 1, 2))
    lost = jnp.sum(values * (old_volume * (~new_wet)[..., None]).reshape(old_volume.shape + extra), axis=(0, 1, 2))
    mean = retained / jnp.where(retained_volume > 0, retained_volume, 1)
    result = jnp.where(old_wet.reshape(old_wet.shape + (1,) * (values.ndim - 2)), mapped, mean)
    result = (result * retained_volume + lost) / jnp.where(total_volume > 0, total_volume, 1)
    return result * (new_thickness > 0).reshape(new_thickness.shape + extra)


def remap_carry(prepared, original, updated, *, stretch=False, redistribute=False, forcing=None):
    """Remap prognostic cell averages and rebuild dependent diagnostic fields."""
    import jax.numpy as jnp
    from .core.numerics import calc_initial_conditions_kernel
    from .core.momentum import vertical_velocity_kernel
    from .core.utilities import enforce_boundaries
    from .core.geometry import wgrid_thickness
    from .routines import tracing_context

    settings, vs = prepared._template.settings, original.variables
    if settings.enable_idemix:
        raise ValueError("Vertical-grid remapping of IDEMIX energy is not implemented")
    remap = remap_column_inventories if stretch else remap_cell_averages
    fields = {}
    for name, grid in (
        ("temp", "T"),
        ("salt", "T"),
        ("dtemp", "T"),
        ("dsalt", "T"),
        ("dHd", "T"),
        ("u", "U"),
        ("v", "V"),
        ("du", "U"),
        ("dv", "V"),
    ):
        if name not in vs:
            continue
        old_fraction = getattr(vs, "hfac" + grid) if settings.enable_partial_cells else getattr(vs, "mask" + grid)
        new_fraction = getattr(updated.variables, "hfac" + grid) if settings.enable_partial_cells else old_fraction
        if redistribute:
            field = remap_connected_inventories(
                getattr(vs, name),
                vs.dzt * old_fraction,
                updated.variables.dzt * new_fraction,
                getattr(vs, "area_" + grid.lower()),
            )
        else:
            field = remap(getattr(vs, name), vs.dzt * old_fraction, updated.variables.dzt * new_fraction, axis=2)
        with tracing_context():
            fields[name] = enforce_boundaries(field, settings.enable_cyclic_x)
    if settings.enable_tke or settings.enable_eke:
        old_width = wgrid_thickness(prepared._make_state(vs))
        new_width = wgrid_thickness(prepared._make_state(updated.variables))
        energy_fields = [name for name in ("tke", "dtke", "eke", "deke") if name in vs]
        if settings.enable_tke:
            energy_fields.append("K_diss_v")
        for name in energy_fields:
            old, new = old_width, new_width
            if name == "K_diss_v":
                old = jnp.broadcast_to(vs.dzw, vs.maskW.shape).at[..., -1].multiply(0.5) * vs.maskW
                new_vs = updated.variables
                new = jnp.broadcast_to(new_vs.dzw, new_vs.maskW.shape).at[..., -1].multiply(0.5) * new_vs.maskW
            if redistribute:
                field = remap_connected_inventories(getattr(vs, name), old, new, vs.area_t)
            else:
                # Moving levels can change the covered W depth at a fixed bed.
                field = remap_column_inventories(getattr(vs, name), old, new, axis=2)
            with tracing_context():
                fields[name] = enforce_boundaries(field, settings.enable_cyclic_x)
    updated = prepared.replace_variables(updated, **fields)
    with tracing_context():
        state = prepared._make_state(updated.variables)
        with state.variables.unlock():
            if forcing is not None:
                forcing(state)
            state.variables.update(calc_initial_conditions_kernel(state))
            original_index = state.variables.taup1
            for index in range(vs.u.shape[-1]):
                state.variables.taup1 = jnp.asarray(index, original_index.dtype)
                state.variables.update(vertical_velocity_kernel(state))
            state.variables.taup1 = original_index
            if settings.enable_eke:
                from .core.eke import set_eke_diffusivities

                set_eke_diffusivities(state)
            if settings.enable_tke:
                from .core.tke import set_tke_diffusivities

                set_tke_diffusivities(state)
        return updated._replace(variables=state.variables)


def vertical_grid(prepared, carry, widths, *, conserve):
    """Rebuild fixed-topology vertical metrics and conservatively remap a carry."""
    import jax.numpy as jnp
    from .core.geometry import bathymetry_geometry
    from .core.numerics import u_centered_grid
    from .routines import tracing_context

    if set(conserve) != {"temp", "salt", "momentum"}:
        raise ValueError("Vertical-grid changes require temp, salt, and momentum conservation")
    settings, vs = prepared._template.settings, carry.variables
    forcing = getattr(prepared.simulation, "rebuild_vertical_forcing", None)
    if (settings.enable_tempsalt_sources or settings.enable_momentum_sources) and not callable(forcing):
        raise ValueError("Volume sources require a rebuild_vertical_forcing hook for a moving vertical grid")
    declaration = prepared.simulation.__dict__.get(
        "rebuild_vertical_forcing", getattr(type(prepared.simulation), "rebuild_vertical_forcing", None)
    )
    if forcing is not None and not getattr(declaration, "supports_compilation", False):
        raise ValueError("rebuild_vertical_forcing must declare supports_compilation=True")
    widths = jnp.asarray(widths, dtype=vs.dzt.dtype)
    if widths.shape != vs.dzt.shape:
        raise ValueError(f"Vertical widths must have shape {vs.dzt.shape}")
    valid = jnp.all(jnp.isfinite(widths) & (widths > 0)) & jnp.isclose(
        jnp.sum(widths), jnp.sum(vs.dzt), rtol=0, atol=1e-10
    )
    full_depth = jnp.sum(vs.maskT * widths, axis=-1)
    if settings.enable_partial_cells:
        bottom_width = widths[jnp.maximum(vs.kbot - 1, 0)]
        valid &= jnp.all((vs.ht >= full_depth - 0.9 * bottom_width) & (vs.ht <= full_depth + 1e-10))
    else:
        valid &= jnp.all(jnp.abs(full_depth - vs.ht) < 1e-10)
    safe = jnp.where(valid, widths, vs.dzt)
    with tracing_context():
        dzw, zt, zw = u_centered_grid(safe, jnp.zeros_like(safe), jnp.zeros_like(safe), jnp.zeros_like(safe))
    valid &= jnp.all(jnp.isfinite(dzw) & (dzw > 0))
    updated = prepared.replace_variables(
        carry,
        dzt=jnp.where(valid, safe, vs.dzt),
        dzw=jnp.where(valid, dzw, vs.dzw),
        zt=jnp.where(valid, zt - zw[-1], vs.zt),
        zw=jnp.where(valid, zw - zw[-1], vs.zw),
    )
    if settings.enable_partial_cells:
        with tracing_context():
            geometry = bathymetry_geometry(prepared._make_state(updated.variables), vs.ht)
        updated = prepared.replace_variables(updated, **geometry._asdict())
    updated = remap_carry(prepared, carry, updated, forcing=forcing)
    return updated._replace(
        failure_code=jnp.where((carry.failure_code == 0) & ~valid, 5, carry.failure_code),
        failed_iteration=jnp.where((carry.failure_code == 0) & ~valid, vs.itt, carry.failed_iteration),
    )
