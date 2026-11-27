"""Fixed-topology partial bottom cells on the reference z grid."""

from veros import veros_kernel, KernelOutput
from veros.core.operators import numpy as npx, update, at
from veros.core import utilities


def validate_diagnostics(state):
    """Reject diagnostics whose partial-cell budgets have not been implemented."""
    if not state.settings.enable_partial_cells:
        return
    for name in ("overturning",):
        diagnostic = state.diagnostics.get(name)
        if diagnostic is not None and (diagnostic.sampling_frequency or diagnostic.output_frequency):
            raise ValueError(f"Partial cells do not yet support the {name} diagnostic")


def cell_thickness(state, grid="T"):
    """Cell thickness, with nominal thickness in dry cells for safe division."""
    vs = state.variables
    nominal = npx.broadcast_to(vs.dzt, vs.maskT.shape)
    if not state.settings.enable_partial_cells:
        return nominal
    fraction = getattr(vs, "hfac" + grid)
    return nominal * npx.where(getattr(vs, "mask" + grid), fraction, 1.0)


def wgrid_thickness(state):
    """Occupied native W control volumes, bounded by tracer centres and the surface."""
    vs = state.variables
    nominal = npx.broadcast_to(vs.dzw, vs.maskW.shape)
    nominal = update(nominal, at[..., -1], 0.5 * nominal[..., -1])
    if not state.settings.enable_partial_cells:
        return nominal * vs.maskW
    submerged = nominal - npx.maximum(-vs.ht[..., None] - vs.zt, 0)
    return npx.maximum(submerged, 0) * vs.maskW


def wgrid_horizontal_divergence(state, flux_east, flux_north):
    """Divergence of W-grid flux densities through occupied shared faces."""
    vs = state.variables
    thickness = wgrid_thickness(state)
    east = update(flux_east, at[:-1], flux_east[:-1] * npx.minimum(thickness[:-1], thickness[1:]))
    north = update(flux_north, at[:, :-1], flux_north[:, :-1] * npx.minimum(thickness[:, :-1], thickness[:, 1:]))
    divergence = (east[2:-2, 2:-2] - east[1:-3, 2:-2]) / (
        vs.cost[npx.newaxis, 2:-2, npx.newaxis] * vs.dxt[2:-2, npx.newaxis, npx.newaxis]
    ) + (north[2:-2, 2:-2] - north[2:-2, 1:-3]) / (
        vs.cost[npx.newaxis, 2:-2, npx.newaxis] * vs.dyt[npx.newaxis, 2:-2, npx.newaxis]
    )
    return divergence * vs.maskW[2:-2, 2:-2] / npx.where(vs.maskW[2:-2, 2:-2], thickness[2:-2, 2:-2], 1)


@veros_kernel
def bathymetry_bounds(state, minimum_fraction):
    """Admissible positive depths for the prepared bottom-cell indices."""
    vs = state.variables
    wet = vs.kbot > 0
    bottom_width = vs.dzt[npx.maximum(vs.kbot - 1, 0)]
    full_depth = npx.sum(vs.maskT * vs.dzt, axis=2)
    lower = npx.where(wet, full_depth - (1 - minimum_fraction) * bottom_width, 0)
    return lower, full_depth


@veros_kernel
def bathymetry_geometry(state, depth):
    """Construct volume and face fractions without changing wet-cell indices."""
    vs = state.variables
    settings = state.settings
    depth = utilities.enforce_boundaries(depth, settings.enable_cyclic_x)
    full_depth = npx.sum(vs.maskT * vs.dzt, axis=2)
    bottom = npx.arange(settings.nz) == vs.kbot[..., None] - 1
    hfacT = npx.where(bottom, 1 + (depth - full_depth)[..., None] / vs.dzt, 1) * vs.maskT
    hfacT = utilities.enforce_boundaries(hfacT, settings.enable_cyclic_x)
    hfacU = update(hfacT, at[:-1], npx.minimum(hfacT[:-1], hfacT[1:])) * vs.maskU
    hfacV = update(hfacT, at[:, :-1], npx.minimum(hfacT[:, :-1], hfacT[:, 1:])) * vs.maskV
    hfacU = utilities.enforce_boundaries(hfacU, settings.enable_cyclic_x)
    hfacV = utilities.enforce_boundaries(hfacV, settings.enable_cyclic_x)
    ht, hu, hv = (npx.sum(fraction * vs.dzt, axis=2) for fraction in (hfacT, hfacU, hfacV))
    hur = npx.where(hu > 0, 1 / npx.where(hu > 0, hu, 1), 0)
    hvr = npx.where(hv > 0, 1 / npx.where(hv > 0, hv, 1), 0)
    return KernelOutput(hfacT=hfacT, hfacU=hfacU, hfacV=hfacV, ht=ht, hu=hu, hv=hv, hur=hur, hvr=hvr)
