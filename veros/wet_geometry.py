"""Channel-depth controls and discrete realizations on fixed-size arrays."""


def opening(prepared, carry, opening_m, channel, *, realized, conserve, minimum_fraction):
    import jax.numpy as jnp
    from .core.geometry import bathymetry_geometry
    from .core.numerics import calc_topo_kernel
    from .remapping import remap_carry
    from .routines import tracing_context

    settings, vs = prepared._template.settings, carry.variables
    if not settings.enable_partial_cells:
        raise ValueError("Channel controls require enable_partial_cells=True before setup")
    if set(conserve) != {"temp", "salt", "momentum"}:
        raise ValueError("Channel opening requires temp, salt, and momentum conservation")
    if not 0 < minimum_fraction <= 1:
        raise ValueError("minimum_fraction must lie in (0, 1]")

    def compiled_hook(name):
        hook = getattr(prepared.simulation, name, None)
        declaration = prepared.simulation.__dict__.get(name, getattr(type(prepared.simulation), name, None))
        if not callable(hook) or not getattr(declaration, "supports_compilation", False):
            raise ValueError(f"Channel opening requires {name} with supports_compilation=True")
        return hook

    forcing = compiled_hook("rebuild_wet_forcing")
    if isinstance(channel, str):
        with tracing_context():
            channel = compiled_hook("wet_geometry_channel")(prepared._make_state(vs), channel)
    channel = jnp.asarray(channel)
    if channel.shape != (settings.nx, settings.ny) or channel.dtype != jnp.bool_:
        raise ValueError("channel must be a boolean interior mask or a name declared by the setup")
    channel = jnp.pad(channel, ((2, 2), (2, 2))) & (vs.kbot == 0)
    depth = jnp.asarray(opening_m, dtype=vs.dzt.dtype)
    if depth.ndim:
        raise ValueError("opening_m must be a scalar depth in meters")
    valid = jnp.isfinite(depth) & (depth >= 0) & (depth <= jnp.sum(vs.dzt)) & jnp.any(channel)
    depth = jnp.where(valid, depth, 0)
    if realized:
        interfaces = jnp.concatenate((jnp.zeros(1, dtype=depth.dtype), jnp.cumsum(vs.dzt[::-1])))
        # Keep the selected topology: zw and reverse cumulative widths can round differently.
        count = jnp.argmin(abs(interfaces - depth))
    else:
        count = jnp.sum(depth > -vs.zw)
        bottom = jnp.clip(settings.nz - count, 0, settings.nz - 1)
        fraction = (depth + vs.zw[bottom]) / vs.dzt[bottom]
        valid &= (depth == 0) | (fraction >= minimum_fraction)
        depth = jnp.where(valid, depth, 0)
        count = jnp.sum(depth > -vs.zw)
    kbot = jnp.where(channel, jnp.where(count > 0, settings.nz - count + 1, 0), vs.kbot).astype(vs.kbot.dtype)
    target = prepared.replace_variables(carry, kbot=kbot)
    with tracing_context():
        fields = calc_topo_kernel(prepared._make_state(target.variables))
        target = prepared.replace_variables(target, **fields._asdict())
        if realized:
            depth = target.variables.ht
        fields = bathymetry_geometry(prepared._make_state(target.variables), jnp.where(channel, depth, vs.ht))
    target = prepared.replace_variables(target, **fields._asdict())
    target = remap_carry(prepared, carry, target, redistribute=True, forcing=forcing)
    return target._replace(
        failure_code=jnp.where((carry.failure_code == 0) & ~valid, 6, carry.failure_code),
        failed_iteration=jnp.where((carry.failure_code == 0) & ~valid, vs.itt, carry.failed_iteration),
    )
