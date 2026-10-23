"""Conservative projections for independently prepared nested geometry references."""


def refined_horizontal_origins(variables, settings, refinement):
    """Retain the parent's western/southern faces on the native staggered grid.

    The configured origin is the first interior positive-side face. Halving
    widths with the same origin translates the domain by half a parent cell.
    """
    if refinement < 2 or int(refinement) != refinement:
        raise ValueError("Use an integer horizontal refinement >= 2")
    units = settings.degtom if settings.coord_degree else 1
    return {
        axis
        + "_origin": float(
            getattr(variables, axis + "u")[2] - getattr(variables, "d" + axis + "t")[2] / units * (1 - 1 / refinement)
        )
        for axis in ("x", "y")
    }


def refined_vertical_widths(variables, refinement):
    """Nest native staggered faces while keeping both spacing sets positive.

    Within each parent cell the fine tracer centres have spacing dzt/r.
    Across a parent face they have spacing dzw/r. Averaging adjacent centre
    spacings gives the fine cell widths, whose parent sums retain every face.
    """
    import jax.numpy as jnp

    if refinement < 2 or int(refinement) != refinement:
        raise ValueError("Use an integer vertical refinement >= 2")
    spacing = jnp.repeat(variables.dzt / refinement, refinement)
    spacing = spacing.at[refinement - 1 :: refinement].set(variables.dzw / refinement)
    return jnp.concatenate((spacing[:1], (spacing[:-1] + spacing[1:]) / 2))


def horizontal_alignment_residual(coarse, fine, *, refinement, coord_degree, metres_per_degree):
    """Check every parent face and the western/southern domain boundaries."""
    import jax.numpy as jnp

    units = metres_per_degree if coord_degree else 1
    errors = []
    for axis in ("x", "y"):
        parent_face = getattr(coarse, axis + "u")[2:-2]
        child_face = getattr(fine, axis + "u")[2:-2]
        parent_width = getattr(coarse, "d" + axis + "t")[2:-2] / units
        child_width = getattr(fine, "d" + axis + "t")[2:-2] / units
        if child_face.size != parent_face.size * refinement:
            raise ValueError("Reference horizontal face counts do not match refinement")
        delta = child_face[refinement - 1 :: refinement] - parent_face
        west = child_face[0] - child_width[0] - (parent_face[0] - parent_width[0])
        if axis == "x" and coord_degree:
            delta = (delta + 180) % 360 - 180
            west = (west + 180) % 360 - 180
        scale = jnp.maximum(jnp.max(abs(parent_width)), 1)
        errors.append(jnp.maximum(jnp.max(abs(delta)), abs(west)) / scale)
    return jnp.max(jnp.stack(errors))


def physical_horizontal_measures(variables, settings):
    """Integrate spherical areas between physical faces for reference comparison.

    Native midpoint metric areas still define the model's own budgets. These
    comparison areas add under nested refinement and preserve constant tracers.
    Face transports retain the native physical meridional and zonal lengths.
    """
    import jax.numpy as jnp

    vs = variables
    width_u = jnp.broadcast_to(vs.dyt[None, 2:-2], (settings.nx, settings.ny))
    width_v = vs.dxt[2:-2, None] * vs.cosu[None, 2:-2]
    if settings.coord_degree:
        north = vs.yu[2:-2] * settings.pi / 180
        south = north - vs.dyt[2:-2] / settings.degtom * settings.pi / 180
        meridional_area = settings.radius * (jnp.sin(north) - jnp.sin(south))
        dual_meridional_area = settings.radius * (
            jnp.sin(vs.yt[3:-1] * settings.pi / 180) - jnp.sin(vs.yt[2:-2] * settings.pi / 180)
        )
        area_t = vs.dxt[2:-2, None] * meridional_area
        area_u = vs.dxu[2:-2, None] * meridional_area
        area_v = vs.dxt[2:-2, None] * dual_meridional_area
    else:
        area_t, area_u, area_v = (getattr(vs, "area_" + grid)[2:-2, 2:-2] for grid in ("t", "u", "v"))
    return dict(area_t=area_t, area_u=area_u, area_v=area_v, width_u=width_u, width_v=width_v)


def _ratios(fine_shape, coarse_shape):
    ratios = tuple(fine // coarse for fine, coarse in zip(fine_shape, coarse_shape))
    if len(fine_shape) != 2 or any(r < 1 or f != c * r for f, c, r in zip(fine_shape, coarse_shape, ratios)):
        raise ValueError("Reference horizontal cells must be aligned integer refinements")
    return ratios


def _relative(error, scale):
    import jax.numpy as jnp

    return jnp.max(jnp.abs(error)) / jnp.maximum(jnp.max(jnp.abs(scale)), 1)


def project_tracer(values, fine_thickness, fine_area, coarse_thickness, coarse_area):
    """Project cell inventories after vertical integration onto candidate levels.

    Arrays contain interior cells. Fine columns must have the same physical wet
    depths as their coarse parents. A nonzero geometry residual identifies a
    mismatch that conservation alone cannot make admissible.
    Trailing history axes are projected independently.
    """
    import jax.numpy as jnp
    from veros.remapping import remap_cell_averages

    nx, ny, nz = coarse_thickness.shape
    rx, ry = _ratios(fine_thickness.shape[:2], (nx, ny))
    target = jnp.repeat(jnp.repeat(coarse_thickness, rx, axis=0), ry, axis=1)
    mapped = remap_cell_averages(values, fine_thickness, target, axis=2)
    extra = values.shape[3:]
    expanded = (1,) * len(extra)
    volume = fine_area[..., None] * target
    inventory = (mapped * volume.reshape(volume.shape + expanded)).reshape((nx, rx, ny, ry, nz) + extra)
    inventory = jnp.sum(inventory, axis=(1, 3))
    coarse_volume = coarse_area[..., None] * coarse_thickness
    result = inventory / jnp.where(coarse_volume > 0, coarse_volume, 1).reshape(coarse_volume.shape + expanded)
    fine_volume = fine_area[..., None] * fine_thickness
    before = jnp.sum(values * fine_volume.reshape(fine_volume.shape + expanded), axis=(0, 1, 2))
    after = jnp.sum(result * coarse_volume.reshape(coarse_volume.shape + expanded), axis=(0, 1, 2))
    area_sum = jnp.sum(fine_area.reshape(nx, rx, ny, ry), axis=(1, 3))
    return dict(
        values=result,
        inventory_residual=_relative(after - before, before),
        geometry_residual=jnp.maximum(
            _relative(jnp.sum(target, axis=2) - jnp.sum(fine_thickness, axis=2), jnp.sum(target, axis=2)),
            _relative(area_sum - coarse_area, coarse_area),
        ),
    )


def project_velocity(values, fine_thickness, fine_width, coarse_thickness, coarse_width, *, normal_axis):
    """Preserve transport on aligned U or V faces, including histories.

    Widths include the tangential metric (and cosine on V faces). Face cells
    have their native positive-side indexing, so a coarse face selects the last
    fine face in its parent block. Fine and coarse open depths must agree.
    """
    import jax.numpy as jnp
    from veros.remapping import remap_cell_averages

    if normal_axis not in (0, 1):
        raise ValueError("normal_axis must select U (0) or V (1) faces")
    nx, ny, nz = coarse_thickness.shape
    rx, ry = _ratios(fine_thickness.shape[:2], (nx, ny))
    target = jnp.repeat(jnp.repeat(coarse_thickness, rx, axis=0), ry, axis=1)
    select = (slice(rx - 1, None, rx), slice(None)) if normal_axis == 0 else (slice(None), slice(ry - 1, None, ry))
    fine, target = fine_thickness[select], target[select]
    width, values = fine_width[select], values[select]
    mapped = remap_cell_averages(values, fine, target, axis=2)
    extra = values.shape[3:]
    expanded = (1,) * len(extra)
    area = width[..., None] * target
    flux = mapped * area.reshape(area.shape + expanded)
    if normal_axis == 0:
        flux = jnp.sum(flux.reshape((nx, ny, ry, nz) + extra), axis=2)
        width_sum = jnp.sum(width.reshape(nx, ny, ry), axis=2)
    else:
        flux = jnp.sum(flux.reshape((nx, rx, ny, nz) + extra), axis=1)
        width_sum = jnp.sum(width.reshape(nx, rx, ny), axis=1)
    coarse_area = coarse_width[..., None] * coarse_thickness
    result = flux / jnp.where(coarse_area > 0, coarse_area, 1).reshape(coarse_area.shape + expanded)
    before = values * (width[..., None] * fine).reshape(fine.shape + expanded)
    before = jnp.sum(before, axis=2)
    if normal_axis == 0:
        before = jnp.sum(before.reshape((nx, ny, ry) + extra), axis=2)
    else:
        before = jnp.sum(before.reshape((nx, rx, ny) + extra), axis=1)
    after = jnp.sum(result * coarse_area.reshape(coarse_area.shape + expanded), axis=2)
    return dict(
        values=result,
        inventory_residual=_relative(after - before, before),
        geometry_residual=jnp.maximum(
            _relative(jnp.sum(target, axis=2) - jnp.sum(fine, axis=2), jnp.sum(target, axis=2)),
            _relative(width_sum - coarse_width, coarse_width),
        ),
    )


def trajectory_error(
    coarse,
    initial,
    fine,
    fine_initial,
    *,
    steps,
    refinement,
    parameters=None,
    fine_parameters=None,
    measure="native",
):
    """Compare native endpoint fields with an independent refined integration.

    Preparation and physical initial conditions belong to the caller. Both
    carries must describe the same domain, wet depths, and physical time. This
    finite-window component does not supply adjusted or statistical references.
    ``measure="physical"`` uses exact spherical comparison areas while leaving
    both numerical timesteps and their native budget measures unchanged.
    """
    import jax
    import jax.numpy as jnp

    if coarse is fine or refinement < 2 or int(refinement) != refinement or steps < 1 or int(steps) != steps:
        raise ValueError("Use distinct prepared models, integer refinement >= 2, and a positive integer horizon")
    if measure not in ("native", "physical"):
        raise ValueError("Reference measure must be native or physical")
    cs, fs = coarse._template.settings, fine._template.settings
    if (fs.nx, fs.ny, fs.nz) != (cs.nx * refinement, cs.ny * refinement, cs.nz * refinement):
        raise ValueError("Independent model dimensions must match the declared refinement")
    if abs(fs.dt_tracer * refinement - cs.dt_tracer) > 1e-10:
        raise ValueError("Reference timesteps must cover the same physical horizon")
    alignment = horizontal_alignment_residual(
        initial.variables,
        fine_initial.variables,
        refinement=refinement,
        coord_degree=cs.coord_degree,
        metres_per_degree=cs.degtom,
    )
    if not isinstance(alignment, jax.core.Tracer) and float(alignment) > 1e-10:
        raise ValueError("Reference horizontal faces describe a different physical domain")
    final = coarse.advance(initial, n_steps=steps, parameters=parameters, final_only=True)[0]
    refined = fine.advance(fine_initial, n_steps=steps * refinement, parameters=fine_parameters, final_only=True)[0]
    result = endpoint_error(final.variables, refined.variables, cs, fs, measure=measure)
    valid = (final.failure_code == 0) & (refined.failure_code == 0)
    valid &= jnp.abs(initial.variables.time - fine_initial.variables.time) < 1e-10
    return dict(
        result,
        error=jnp.where(valid, result["error"], jnp.nan),
        coarse_failure_code=final.failure_code,
        fine_failure_code=refined.failure_code,
    )


def endpoint_error(cv, fv, cs, fs, *, measure="native"):
    """Compare completed native fields on a declared common physical measure."""
    import jax.numpy as jnp

    if measure not in ("native", "physical"):
        raise ValueError("Reference measure must be native or physical")
    alignment = horizontal_alignment_residual(
        cv, fv, refinement=fs.nx // cs.nx, coord_degree=cs.coord_degree, metres_per_degree=cs.degtom
    )
    measures = (
        (physical_horizontal_measures(cv, cs), physical_horizontal_measures(fv, fs))
        if measure == "physical"
        else tuple(
            {"area_" + grid: getattr(vs, "area_" + grid)[2:-2, 2:-2] for grid in ("t", "u", "v")} for vs in (cv, fv)
        )
    )
    coarse_measure, fine_measure = measures

    def thickness(vs, grid):
        return vs.dzt * (getattr(vs, "hfac" + grid) if "hfac" + grid in vs else getattr(vs, "mask" + grid))[2:-2, 2:-2]

    errors, budgets, geometry = [], [], []
    for name, grid, scale in (("temp", "T", 0.05), ("salt", "T", 0.02), ("u", "U", 0.01), ("v", "V", 0.01)):
        old, target = thickness(fv, grid), thickness(cv, grid)
        field = getattr(fv, name)[2:-2, 2:-2, :, fv.tau]
        if grid == "T":
            projected = project_tracer(field, old, fine_measure["area_t"], target, coarse_measure["area_t"])
        else:
            fine_width = fv.dyt[None, 2:-2] if grid == "U" else fv.dxt[2:-2, None] * fv.cosu[None, 2:-2]
            width = cv.dyt[None, 2:-2] if grid == "U" else cv.dxt[2:-2, None] * cv.cosu[None, 2:-2]
            projected = project_velocity(
                field,
                old,
                jnp.broadcast_to(fine_width, old.shape[:2]),
                target,
                jnp.broadcast_to(width, target.shape[:2]),
                normal_axis=0 if grid == "U" else 1,
            )
        volume = coarse_measure["area_" + grid.lower()][..., None] * target
        difference = (getattr(cv, name)[2:-2, 2:-2, :, cv.tau] - projected["values"]) / scale
        errors.append(jnp.sum(volume * difference**2) / jnp.maximum(jnp.sum(volume), 1))
        budgets.append(projected["inventory_residual"])
        geometry.append(projected["geometry_residual"])
    geometry_residual = jnp.maximum(jnp.max(jnp.stack(geometry)), alignment)
    return dict(
        error=jnp.where(geometry_residual < 1e-10, jnp.mean(jnp.stack(errors)), jnp.nan),
        remap_budget_residual=jnp.max(jnp.stack(budgets)),
        projection_geometry_residual=geometry_residual,
        projection_horizontal_residual=alignment,
        field_errors=jnp.stack(errors),
    )
