"""Declared research channels and forcing extension from the reference ocean."""

MONTHLY_FIELDS = ("sss_clim", "sst_clim", "qnec", "qnet", "taux", "tauy")


def add_reference_variables(state):
    from veros.variables import Variable

    for name, dtype in (
        ("geometry_reference_wet", "bool"),
        ("geometry_forcing_x", "int32"),
        ("geometry_forcing_y", "int32"),
    ):
        state.var_meta[name] = Variable(name, ("xt", "yt"), dtype=dtype, time_dependent=False, write_to_restart=False)


def initialize_reference(state):
    """Freeze reference wet cells and nearest-ocean forcing donors during setup.

    Spherical grids use great-circle distance, and Cartesian grids use
    Euclidean distance. Donors are always original interior ocean cells. Chunking bounds
    setup memory independently of the number of dry cells.
    """
    import numpy as np
    from veros.core.operators import numpy as npx

    vs = state.variables
    wet = np.asarray(vs.maskT[..., -1]).astype(bool)
    interior = wet.copy()
    interior[:2] = interior[-2:] = False
    interior[:, :2] = interior[:, -2:] = False
    ocean = np.argwhere(interior)
    if not ocean.size:
        raise ValueError("Wet forcing extension requires reference interior ocean cells")
    x, y = np.broadcast_arrays(np.asarray(vs.xt)[:, None], np.asarray(vs.yt)[None, :])
    if state.settings.coord_degree:
        lon, lat = np.deg2rad(x), np.deg2rad(y)
        points = np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)
    else:
        points = np.stack((x, y), axis=-1)
    dry = np.argwhere(~wet)
    donors = np.stack(np.indices(wet.shape), axis=-1)
    targets = points[tuple(ocean.T)]
    for start in range(0, len(dry), 256):
        cells = dry[start : start + 256]
        locations = points[tuple(cells.T)]
        if state.settings.coord_degree:
            nearest = np.argmax(locations @ targets.T, axis=1)
        else:
            distances = np.sum((locations[:, None, :] - targets[None, :, :]) ** 2, axis=-1)
            nearest = np.argmin(distances, axis=1)
        donors[tuple(cells.T)] = ocean[nearest]
    vs.geometry_reference_wet = npx.asarray(wet)
    vs.geometry_forcing_x = npx.asarray(donors[..., 0], dtype="int32")
    vs.geometry_forcing_y = npx.asarray(donors[..., 1], dtype="int32")


def channel_mask(state, setup, name):
    """Return the dry part of the declared passage in the reference geometry."""
    vs = state.variables
    if setup == "global_4deg" and name == "Panama":
        longitude = (vs.xt[2:-2] + 180) % 360 - 180
        latitude = vs.yt[2:-2]
        # The four-degree reference passage is the dry cell at 82W, 10N.
        region = (
            (longitude[:, None] >= -84)
            & (longitude[:, None] < -80)
            & (latitude[None, :] >= 8)
            & (latitude[None, :] < 12)
        )
        return region & ~vs.geometry_reference_wet[2:-2, 2:-2]
    if setup == "adjusted_basin" and name == "sill":
        raise ValueError("The declared adjusted basin has no dry interior columns for a sill opening")
    raise ValueError(f"No research channel {name!r} is declared for setup {setup!r}")


def extend_forcing(state):
    """Extend monthly boundary data onto new wet cells without a timestep.

    Each new column uses its nearest reference ocean column's twelve monthly
    values. Original wet-column values and accumulated forcing budgets stay
    unchanged. This is an explicit coastal forcing extension policy.
    """
    from veros.core.operators import numpy as npx

    vs = state.variables
    newly_wet = vs.maskT[..., -1] & ~vs.geometry_reference_wet
    for name in MONTHLY_FIELDS:
        if name not in vs:
            continue
        values = getattr(vs, name)
        donor = values[vs.geometry_forcing_x, vs.geometry_forcing_y]
        setattr(vs, name, npx.where(newly_wet[..., None], donor, values))
