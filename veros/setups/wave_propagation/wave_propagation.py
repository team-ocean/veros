#!/usr/bin/env python

import os

import h5netcdf
from PIL import Image
import scipy.ndimage

from veros import logger, veros_routine, veros_kernel, VerosSetup, KernelOutput
from veros.variables import Variable
import veros.tools
from veros.core.operators import numpy as npx, update, at
from veros.core.utilities import enforce_boundaries

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILES = veros.tools.get_assets("wave_propagation", os.path.join(BASE_PATH, "assets.json"))
TOPO_MASK_FILE = os.path.join(BASE_PATH, "topography_idealized.png")
NA_MASK_FILE = os.path.join(BASE_PATH, "na_mask.png")


class WavePropagationSetup(VerosSetup):
    """
    Global model with flexible resolution and idealized geometry in the
    Atlantic to examine coastal wave propagation.

    Reference:
        Hafner, D., Jacobsen, R. L., Eden, C., Kristensen, M. R. B., Jochum, M., Nuterman, R., & Vinter, B. (2018).
        Veros v0.1-a fast and versatile ocean simulator in pure Python. Geoscientific Model Development, 11(8), 3299-3312.
        `<https://doi.org/10.5194/gmd-11-3299-2018>`_.

    """

    # settings for north atlantic
    na_shelf_depth = 250
    na_shelf_distance = 0
    na_slope_length = 600e3
    na_max_depth = 4000

    # global settings
    max_depth = 5600.0
    equatorial_grid_spacing = 0.5
    polar_grid_spacing = None

    # southern ocean wind modifier
    so_wind_interval = (-69.0, -27.0)
    so_wind_factor = 1.0

    @veros_routine
    def set_parameter(self, state):
        settings = state.settings

        settings.identifier = "wave_propagation"

        settings.nx = 180
        settings.ny = 180
        settings.nz = 60
        settings.dt_mom = settings.dt_tracer = 0
        settings.runlen = 86400 * 10

        settings.x_origin = 90.0
        settings.y_origin = -80.0

        settings.coord_degree = True
        settings.enable_cyclic_x = True

        # friction
        settings.enable_hor_friction = True
        settings.A_h = 5e4
        settings.enable_hor_friction_cos_scaling = True
        settings.hor_friction_cosPower = 1
        settings.enable_tempsalt_sources = True
        settings.enable_implicit_vert_friction = True

        settings.eq_of_state_type = 5

        # isoneutral
        settings.enable_neutral_diffusion = True
        settings.K_iso_0 = 1000.0
        settings.K_iso_steep = 50.0
        settings.iso_dslope = 0.005
        settings.iso_slopec = 0.005
        settings.enable_skew_diffusion = True

        # tke
        settings.enable_tke = True
        settings.c_k = 0.1
        settings.c_eps = 0.7
        settings.alpha_tke = 30.0
        settings.mxl_min = 1e-8
        settings.tke_mxl_choice = 2
        settings.kappaM_min = 2e-4
        settings.kappaH_min = 2e-5
        settings.enable_kappaH_profile = True
        settings.enable_tke_superbee_advection = True

        # eke
        settings.enable_eke = True
        settings.eke_k_max = 1e4
        settings.eke_c_k = 0.4
        settings.eke_c_eps = 0.5
        settings.eke_cross = 2.0
        settings.eke_crhin = 1.0
        settings.eke_lmin = 100.0
        settings.enable_eke_superbee_advection = True
        settings.enable_eke_isopycnal_diffusion = True

        # idemix
        settings.enable_idemix = False
        settings.enable_eke_diss_surfbot = True
        settings.eke_diss_surfbot_frac = 0.2
        settings.enable_idemix_superbee_advection = True
        settings.enable_idemix_hor_diffusion = True

        # custom variables
        state.dimensions["nmonths"] = 12
        state.var_meta.update(
            t_star=Variable("t_star", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
            s_star=Variable("s_star", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
            qnec=Variable("qnec", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
            qnet=Variable("qnet", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
            qsol=Variable("qsol", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
            divpen_shortwave=Variable("divpen_shortwave", ("zt",), "", "", time_dependent=False),
            taux=Variable("taux", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
            tauy=Variable("tauy", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
            na_mask=Variable(
                "Mask for North Atlantic",
                ("xt", "yt"),
                "",
                "Mask for North Atlantic",
                dtype="bool",
                time_dependent=False,
            ),
        )

    def _get_data(self, var):
        with h5netcdf.File(DATA_FILES["forcing"], "r") as forcing_file:
            var_obj = forcing_file.variables[var]
            return npx.array(var_obj, dtype=str(var_obj.dtype)).T

    @veros_routine(dist_safe=False, local_variables=["dxt", "dyt", "dzt"])
    def set_grid(self, state):
        vs = state.variables
        settings = state.settings

        if settings.ny % 2:
            raise ValueError("ny has to be an even number of grid cells")

        vs.dxt = update(vs.dxt, at[...], 360.0 / settings.nx)
        vs.dyt = update(
            vs.dyt,
            at[2:-2],
            veros.tools.get_vinokur_grid_steps(
                settings.ny,
                160.0,
                self.equatorial_grid_spacing,
                upper_stepsize=self.polar_grid_spacing,
                two_sided_grid=True,
            ),
        )
        vs.dzt = veros.tools.get_vinokur_grid_steps(settings.nz, self.max_depth, 10.0, refine_towards="lower")

    @veros_routine
    def set_coriolis(self, state):
        vs = state.variables
        settings = state.settings
        vs.coriolis_t = update(
            vs.coriolis_t, at[...], 2 * settings.omega * npx.sin(vs.yt[npx.newaxis, :] / 180.0 * settings.pi)
        )

    def _shift_longitude_array(self, vs, lon, arr):
        wrap_i = npx.where((lon[:-1] < vs.xt.min()) & (lon[1:] >= vs.xt.min()))[0][0]
        new_lon = npx.concatenate((lon[wrap_i:-1], lon[:wrap_i] + 360.0))
        new_arr = npx.concatenate((arr[wrap_i:-1, ...], arr[:wrap_i, ...]))
        return new_lon, new_arr

    @veros_routine(dist_safe=False, local_variables=["kbot", "xt", "yt", "zt", "na_mask"])
    def set_topography(self, state):
        vs = state.variables
        settings = state.settings

        with h5netcdf.File(DATA_FILES["topography"], "r") as topography_file:
            topo_x, topo_y, topo_z = (npx.array(topography_file.variables[k], dtype="float").T for k in ("x", "y", "z"))
        topo_z = npx.where(topo_z > 0, 0.0, topo_z)
        topo_mask = (npx.flipud(npx.array(Image.open(TOPO_MASK_FILE))).T / 255).astype("bool")
        gaussian_sigma = (0.5 * len(topo_x) / settings.nx, 0.5 * len(topo_y) / settings.ny)
        topo_z_smoothed = scipy.ndimage.gaussian_filter(topo_z, sigma=gaussian_sigma)
        topo_z_smoothed = npx.where(~topo_mask & (topo_z_smoothed >= 0.0), -100.0, topo_z_smoothed)
        topo_masked = npx.where(topo_mask, 0.0, topo_z_smoothed)

        na_mask_image = npx.flipud(npx.array(Image.open(NA_MASK_FILE))).T / 255.0
        topo_x_shifted, na_mask_shifted = self._shift_longitude_array(vs, topo_x, na_mask_image)
        coords = (vs.xt[2:-2], vs.yt[2:-2])
        vs.na_mask = update(
            vs.na_mask,
            at[2:-2, 2:-2],
            npx.logical_not(
                veros.tools.interpolate(
                    (topo_x_shifted, topo_y), na_mask_shifted, coords, kind="nearest", fill=False
                ).astype("bool")
            ),
        )

        topo_x_shifted, topo_masked_shifted = self._shift_longitude_array(vs, topo_x, topo_masked)
        z_interp = veros.tools.interpolate(
            (topo_x_shifted, topo_y), topo_masked_shifted, coords, kind="nearest", fill=False
        )
        z_interp = npx.where(vs.na_mask[2:-2, 2:-2], -self.na_max_depth, z_interp)

        grid_coords = npx.meshgrid(*coords, indexing="ij")
        coastline_distance = veros.tools.get_coastline_distance(
            grid_coords, z_interp >= 0, spherical=True, radius=settings.radius
        )
        if self.na_slope_length:
            slope_distance = coastline_distance - self.na_shelf_distance
            slope_mask = npx.logical_and(vs.na_mask[2:-2, 2:-2], slope_distance < self.na_slope_length)
            z_interp = npx.where(
                slope_mask,
                -(
                    self.na_shelf_depth
                    + slope_distance / self.na_slope_length * (self.na_max_depth - self.na_shelf_depth)
                ),
                z_interp,
            )
        if self.na_shelf_distance:
            shelf_mask = npx.logical_and(vs.na_mask[2:-2, 2:-2], coastline_distance < self.na_shelf_distance)
            z_interp = npx.where(shelf_mask, -self.na_shelf_depth, z_interp)

        depth_levels = 1 + npx.argmin(npx.abs(z_interp[:, :, npx.newaxis] - vs.zt[npx.newaxis, npx.newaxis, :]), axis=2)
        vs.kbot = update(vs.kbot, at[2:-2, 2:-2], npx.where(z_interp < 0.0, depth_levels, 0))
        vs.kbot = npx.where(vs.kbot < settings.nz, vs.kbot, 0)

    @staticmethod
    def _north_atlantic_zonal_mean(vs, arr):
        """Calculate zonal mean forcing over masked area (na_mask)."""
        newaxes = (slice(2, -2), slice(2, -2)) + (npx.newaxis,) * (arr.ndim - 2)
        invalid_mask = npx.logical_or(~vs.na_mask[newaxes], arr == 0.0)
        arr_masked = npx.where(invalid_mask, npx.nan, arr)
        zonal_mean_na = npx.nanmean(arr_masked, axis=0)[npx.newaxis, ...]
        return npx.where(invalid_mask, arr, zonal_mean_na)

    @veros_routine(
        dist_safe=False,
        local_variables=[
            "qnet",
            "temp",
            "salt",
            "maskT",
            "taux",
            "tauy",
            "xt",
            "yt",
            "zt",
            "qnec",
            "qsol",
            "t_star",
            "s_star",
            "na_mask",
            "maskW",
            "divpen_shortwave",
            "dzt",
            "zw",
        ],
    )
    def set_initial_conditions(self, state):
        vs = state.variables
        settings = state.settings

        rpart_shortwave = 0.58
        efold1_shortwave = 0.35
        efold2_shortwave = 23.0

        t_grid = (vs.xt[2:-2], vs.yt[2:-2], vs.zt)
        xt_forc, yt_forc, zt_forc = (self._get_data(k) for k in ("xt", "yt", "zt"))
        zt_forc = zt_forc[::-1]

        # initial conditions
        temp_data = veros.tools.interpolate(
            (xt_forc, yt_forc, zt_forc), self._get_data("temperature")[:, :, ::-1], t_grid, missing_value=0.0
        )
        vs.temp = update(vs.temp, at[2:-2, 2:-2, ...], (temp_data * vs.maskT[2:-2, 2:-2, :])[..., npx.newaxis])

        salt_data = veros.tools.interpolate(
            (xt_forc, yt_forc, zt_forc), self._get_data("salinity")[:, :, ::-1], t_grid, missing_value=0.0
        )
        vs.salt = update(vs.salt, at[2:-2, 2:-2, ...], (salt_data * vs.maskT[2:-2, 2:-2, :])[..., npx.newaxis])

        # wind stress on MIT grid
        time_grid = (vs.xt[2:-2], vs.yt[2:-2], npx.arange(12))
        taux_data = veros.tools.interpolate(
            (xt_forc, yt_forc, npx.arange(12)), self._get_data("tau_x"), time_grid, missing_value=0.0
        )
        vs.taux = update(vs.taux, at[2:-2, 2:-2, :], taux_data)
        mask = npx.logical_and(vs.yt > self.so_wind_interval[0], vs.yt < self.so_wind_interval[1])[..., npx.newaxis]
        vs.taux = npx.where(
            mask,
            vs.taux
            + vs.taux
            * (self.so_wind_factor - 1.0)
            * npx.sin(
                npx.pi
                * (vs.yt[npx.newaxis, :, npx.newaxis] - self.so_wind_interval[0])
                / (self.so_wind_interval[1] - self.so_wind_interval[0])
            ),
            vs.taux,
        )

        tauy_data = veros.tools.interpolate(
            (xt_forc, yt_forc, npx.arange(12)), self._get_data("tau_y"), time_grid, missing_value=0.0
        )
        vs.tauy = update(vs.tauy, at[2:-2, 2:-2, :], tauy_data)

        vs.taux = enforce_boundaries(vs.taux, settings.enable_cyclic_x)
        vs.tauy = enforce_boundaries(vs.tauy, settings.enable_cyclic_x)

        # Qnet and dQ/dT and Qsol
        qnet_data = veros.tools.interpolate(
            (xt_forc, yt_forc, npx.arange(12)), self._get_data("q_net"), time_grid, missing_value=0.0
        )
        vs.qnet = update(vs.qnet, at[2:-2, 2:-2, :], -qnet_data * vs.maskT[2:-2, 2:-2, -1, npx.newaxis])

        qnec_data = veros.tools.interpolate(
            (xt_forc, yt_forc, npx.arange(12)), self._get_data("dqdt"), time_grid, missing_value=0.0
        )
        vs.qnec = update(vs.qnec, at[2:-2, 2:-2, :], qnec_data * vs.maskT[2:-2, 2:-2, -1, npx.newaxis])

        qsol_data = veros.tools.interpolate(
            (xt_forc, yt_forc, npx.arange(12)), self._get_data("swf"), time_grid, missing_value=0.0
        )
        vs.qsol = update(vs.qsol, at[2:-2, 2:-2, :], -qsol_data * vs.maskT[2:-2, 2:-2, -1, npx.newaxis])

        # SST and SSS
        sst_data = veros.tools.interpolate(
            (xt_forc, yt_forc, npx.arange(12)), self._get_data("sst"), time_grid, missing_value=0.0
        )
        vs.t_star = update(vs.t_star, at[2:-2, 2:-2, :], sst_data * vs.maskT[2:-2, 2:-2, -1, npx.newaxis])

        sss_data = veros.tools.interpolate(
            (xt_forc, yt_forc, npx.arange(12)), self._get_data("sss"), time_grid, missing_value=0.0
        )
        vs.s_star = update(vs.s_star, at[2:-2, 2:-2, :], sss_data * vs.maskT[2:-2, 2:-2, -1, npx.newaxis])

        if settings.enable_idemix:
            tidal_energy_data = veros.tools.interpolate(
                (xt_forc, yt_forc), self._get_data("tidal_energy"), t_grid[:-1], missing_value=0.0
            )
            mask_x, mask_y = (i + 2 for i in npx.indices((settings.nx, settings.ny)))
            mask_z = npx.maximum(0, vs.kbot[2:-2, 2:-2] - 1)
            tidal_energy_data *= vs.maskW[mask_x, mask_y, mask_z] / settings.rho_0
            vs.forc_iw_bottom = update(vs.forc_iw_bottom, at[2:-2, 2:-2], tidal_energy_data)

        # average variables in North Atlantic
        na_average_vars = ["taux", "tauy", "qnet", "qnec", "qsol", "t_star", "s_star", "salt", "temp"]

        for var in na_average_vars:
            val = getattr(vs, var)
            new_val = update(val, at[2:-2, 2:-2, ...], self._north_atlantic_zonal_mean(vs, val[2:-2, 2:-2, ...]))
            setattr(vs, var, new_val)

        """
        Initialize penetration profile for solar radiation and store divergence in divpen
        note that pen is set to 0.0 at the surface instead of 1.0 to compensate for the
        shortwave part of the total surface flux
        """
        swarg1 = vs.zw / efold1_shortwave
        swarg2 = vs.zw / efold2_shortwave
        pen = rpart_shortwave * npx.exp(swarg1) + (1.0 - rpart_shortwave) * npx.exp(swarg2)
        pen = update(pen, at[-1], 0.0)
        vs.divpen_shortwave = update(vs.divpen_shortwave, at[1:], (pen[1:] - pen[:-1]) / vs.dzt[1:])
        vs.divpen_shortwave = update(vs.divpen_shortwave, at[0], pen[0] / vs.dzt[0])

    @veros_routine
    def set_forcing(self, state):
        vs = state.variables
        self.set_timestep(state)
        vs.update(set_forcing_kernel(state))

    @veros_routine
    def set_diagnostics(self, state):
        diagnostics = state.diagnostics
        settings = state.settings

        diagnostics["cfl_monitor"].output_frequency = 86400.0
        diagnostics["tracer_monitor"].output_frequency = 86400.0
        diagnostics["snapshot"].output_frequency = 10 * 86400.0
        diagnostics["snapshot"].output_variables += ["na_mask"]
        diagnostics["overturning"].output_frequency = 360 * 86400
        diagnostics["overturning"].sampling_frequency = 10 * 86400
        diagnostics["energy"].output_frequency = 360 * 86400
        diagnostics["energy"].sampling_frequency = 86400.0
        diagnostics["averages"].output_frequency = 360 * 86400
        diagnostics["averages"].sampling_frequency = 86400.0

        average_vars = [
            "surface_taux",
            "surface_tauy",
            "forc_temp_surface",
            "forc_salt_surface",
            "psi",
            "temp",
            "salt",
            "u",
            "v",
            "w",
            "Nsqr",
            "Hd",
            "rho",
            "K_diss_v",
            "P_diss_v",
            "P_diss_nonlin",
            "P_diss_iso",
            "kappaH",
        ]
        if settings.enable_skew_diffusion:
            average_vars += ["B1_gm", "B2_gm"]
        if settings.enable_TEM_friction:
            average_vars += ["kappa_gm", "K_diss_gm"]
        if settings.enable_tke:
            average_vars += ["tke", "Prandtlnumber", "mxl", "tke_diss", "forc_tke_surface", "tke_surf_corr"]
        if settings.enable_idemix:
            average_vars += ["E_iw", "forc_iw_surface", "iw_diss", "c0", "v0"]
        if settings.enable_eke:
            average_vars += ["eke", "K_gm", "L_rossby", "L_rhines"]

        diagnostics["averages"].output_variables = average_vars

    @veros_routine
    def set_timestep(self, state):
        vs = state.variables
        settings = state.settings

        if vs.time < 90 * 86400:
            if settings.dt_tracer != 1800.0:
                with settings.unlock():
                    settings.dt_tracer = settings.dt_mom = 1800.0

                logger.info("Setting time step to 30m")

        else:
            if settings.dt_tracer != 3600.0:
                with settings.unlock():
                    settings.dt_tracer = settings.dt_mom = 3600.0

                logger.info("Setting time step to 1h")

    @veros_routine
    def after_timestep(self, state):
        pass


@veros_kernel
def set_forcing_kernel(state):
    vs = state.variables
    settings = state.settings

    t_rest = 30.0 * 86400.0
    cp_0 = 3991.86795711963  # J/kg /K

    year_in_seconds = 360 * 86400.0
    (n1, f1), (n2, f2) = veros.tools.get_periodic_interval(vs.time, year_in_seconds, year_in_seconds / 12.0, 12)

    vs.surface_taux = f1 * vs.taux[:, :, n1] + f2 * vs.taux[:, :, n2]
    vs.surface_tauy = f1 * vs.tauy[:, :, n1] + f2 * vs.tauy[:, :, n2]

    if settings.enable_tke:
        vs.forc_tke_surface = update(
            vs.forc_tke_surface,
            at[1:-1, 1:-1],
            npx.sqrt(
                (0.5 * (vs.surface_taux[1:-1, 1:-1] + vs.surface_taux[:-2, 1:-1]) / settings.rho_0) ** 2
                + (0.5 * (vs.surface_tauy[1:-1, 1:-1] + vs.surface_tauy[1:-1, :-2]) / settings.rho_0) ** 2
            )
            ** (3.0 / 2.0),
        )

    # W/m^2 K kg/J m^3/kg = K m/s
    fxa = f1 * vs.t_star[..., n1] + f2 * vs.t_star[..., n2]
    qqnec = f1 * vs.qnec[..., n1] + f2 * vs.qnec[..., n2]
    qqnet = f1 * vs.qnet[..., n1] + f2 * vs.qnet[..., n2]
    vs.forc_temp_surface = (
        (qqnet + qqnec * (fxa - vs.temp[..., -1, vs.tau])) * vs.maskT[..., -1] / cp_0 / settings.rho_0
    )
    fxa = f1 * vs.s_star[..., n1] + f2 * vs.s_star[..., n2]
    vs.forc_salt_surface = 1.0 / t_rest * (fxa - vs.salt[..., -1, vs.tau]) * vs.maskT[..., -1] * vs.dzt[-1]

    # apply simple ice mask
    mask1 = vs.temp[:, :, -1, vs.tau] * vs.maskT[:, :, -1] <= -1.8
    mask2 = vs.forc_temp_surface <= 0
    ice = ~(mask1 & mask2)
    vs.forc_temp_surface *= ice
    vs.forc_salt_surface *= ice

    # solar radiation
    if settings.enable_tempsalt_sources:
        vs.temp_source = (
            (f1 * vs.qsol[..., n1, None] + f2 * vs.qsol[..., n2, None])
            * vs.divpen_shortwave[None, None, :]
            * ice[..., None]
            * vs.maskT
            / cp_0
            / settings.rho_0
        )

    return KernelOutput(
        surface_taux=vs.surface_taux,
        surface_tauy=vs.surface_tauy,
        temp_source=vs.temp_source,
        forc_tke_surface=vs.forc_tke_surface,
        forc_temp_surface=vs.forc_temp_surface,
        forc_salt_surface=vs.forc_salt_surface,
    )
