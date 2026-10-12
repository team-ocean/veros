"""Fixed-topology partial bottom cells on the reference z grid."""

from veros import veros_kernel, KernelOutput
from veros.core.operators import numpy as npx, update, at
from veros.core import utilities


def validate_diagnostics(state):
    """Reject diagnostics whose partial-cell budgets have not been implemented."""
    if not state.settings.enable_partial_cells:
        return
    for name in ("energy", "overturning"):
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
