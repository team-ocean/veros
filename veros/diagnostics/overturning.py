import os

from veros import logger, veros_kernel, KernelOutput

from veros.diagnostics.base import VerosDiagnostic
from veros.core import density
from veros.variables import Variable, allocate
from veros.distributed import global_sum
from veros.core.operators import numpy as npx, update, update_add, at, for_loop


VARIABLES = {
    "nitts": Variable("nitts", None, write_to_restart=True),
    "sigma": Variable("Sigma axis", ("sigma",), "kg/m^3", "Sigma axis", time_dependent=False, write_to_restart=True),
    "zarea": Variable(
        "zarea",
        ("yu", "zt"),
        write_to_restart=True,
    ),
    "trans": Variable("Meridional transport", ("yu", "sigma"), "m^3/s", "Meridional transport", write_to_restart=True),
    "vsf_iso": Variable("Meridional transport", ("yu", "zw"), "m^3/s", "Meridional transport", write_to_restart=True),
    "vsf_depth": Variable("Meridional transport", ("yu", "zw"), "m^3/s", "Meridional transport", write_to_restart=True),
    "bolus_iso": Variable(
        "Meridional transport",
        ("yu", "zw"),
        "m^3/s",
        "Meridional transport",
        write_to_restart=True,
        active=lambda settings: settings.enable_neutral_diffusion and settings.enable_skew_diffusion,
    ),
    "bolus_depth": Variable(
        "Meridional transport",
        ("yu", "zw"),
        "m^3/s",
        "Meridional transport",
        write_to_restart=True,
        active=lambda settings: settings.enable_neutral_diffusion and settings.enable_skew_diffusion,
    ),
}

DEFAULT_OUTPUT_VARS = [var for var in VARIABLES.keys() if var not in ("nitts",)]


class Overturning(VerosDiagnostic):
    """Isopycnal overturning diagnostic. Computes and writes vertical streamfunctions
    (zonally averaged).
    """

    name = "overturning"  #:
    output_path = "{identifier}.overturning.nc"  #: File to write to. May contain format strings that are replaced with Veros attributes.
    output_frequency = None  #: Frequency (in seconds) in which output is written.
    sampling_frequency = None  #: Frequency (in seconds) in which variables are accumulated.
    p_ref = 2000.0  #: Reference pressure for isopycnals

    var_meta = VARIABLES

    def __init__(self, state):
        self.output_variables = []

        for var in DEFAULT_OUTPUT_VARS:
            active = self.var_meta[var].active
            if callable(active):
                active = active(state.settings)

            if active:
                self.output_variables.append(var)

    def initialize(self, state):
        vs = state.variables
        settings = state.settings

        # sigma levels
        nlevel = settings.nz * 4
        sige = density.get_potential_rho(state, 35.0, -2.0, press_ref=self.p_ref)
        sigs = density.get_potential_rho(state, 35.0, 30.0, press_ref=self.p_ref)
        dsig = float(sige - sigs) / (nlevel - 1)

        logger.debug(" Sigma ranges for overturning diagnostic:")
        logger.debug(f" Start sigma0 = {sigs:.1f}")
        logger.debug(f" End sigma0 = {sige:.1f}")
        logger.debug(f" Delta sigma0 = {dsig:.1e}")

        if settings.enable_neutral_diffusion and settings.enable_skew_diffusion:
            logger.debug(" Also calculating overturning by eddy-driven velocities")

        self.extra_dimensions = dict(sigma=nlevel)
        self.initialize_variables(state)

        ovt_vs = self.variables

        ovt_vs.sigma = sigs + dsig * npx.arange(nlevel)

        # precalculate area below z levels
        ovt_vs.zarea = update(
            ovt_vs.zarea,
            at[2:-2, :],
            npx.cumsum(
                zonal_sum(
                    npx.sum(
                        vs.dxt[2:-2, npx.newaxis, npx.newaxis]
                        * vs.cosu[npx.newaxis, 2:-2, npx.newaxis]
                        * vs.maskV[2:-2, 2:-2, :],
                        axis=0,
                    )
                )
                * vs.dzt[npx.newaxis, :],
                axis=1,
            ),
        )

        self.initialize_output(state)

    def diagnose(self, state):
        ovt_vs = self.variables
        ovt_vs.update(diagnose_kernel(state, ovt_vs, self.p_ref))
        ovt_vs.nitts = ovt_vs.nitts + 1

    def output(self, state):
        if not os.path.isfile(self.get_output_file_name(state)):
            self.initialize_output(state)

        ovt_vs = self.variables

        mean_variables = ("trans", "vsf_iso", "vsf_depth")

        if ovt_vs.nitts > 0:
            for var in mean_variables:
                if var not in self.output_variables:
                    continue

                val = getattr(ovt_vs, var)
                setattr(ovt_vs, var, val / ovt_vs.nitts)

        self.write_output(state)

        for var in mean_variables:
            if var not in self.output_variables:
                continue

            val = getattr(ovt_vs, var)
            setattr(ovt_vs, var, val * 0)

        ovt_vs.nitts = 0


@veros_kernel
def _interpolate_depth_coords(coords, arr, interp_coords):
    # ensure depth coordinates are monotonically increasing
    coords = -coords
    interp_coords = -interp_coords

    interp_vectorized = npx.vectorize(npx.interp, signature="(n),(m),(m)->(n)")
    return interp_vectorized(interp_coords, coords, arr)


@veros_kernel
def diagnose_kernel(state, ovt_vs, p_ref):
    vs = state.variables
    settings = state.settings

    nlevel = settings.nz * 4

    # sigma at p_ref
    sig_loc = allocate(state.dimensions, ("xt", "yt", "zt"))
    sig_loc = update(
        sig_loc,
        at[2:-2, 2:-1, :],
        density.get_rho(state, vs.salt[2:-2, 2:-1, :, vs.tau], vs.temp[2:-2, 2:-1, :, vs.tau], p_ref),
    )

    # transports below isopycnals and area below isopycnals
    sig_loc_face = 0.5 * (sig_loc[2:-2, 2:-2, :] + sig_loc[2:-2, 3:-1, :])

    trans = allocate(state.dimensions, ("yu", nlevel))
    z_sig = allocate(state.dimensions, ("yu", nlevel))

    fac = (
        vs.dxt[2:-2, npx.newaxis, npx.newaxis]
        * vs.cosu[npx.newaxis, 2:-2, npx.newaxis]
        * vs.dzt[npx.newaxis, npx.newaxis, :]
        * vs.maskV[2:-2, 2:-2, :]
    )

    def loop_body(m, values):
        trans, z_sig = values
        mask = sig_loc_face > ovt_vs.sigma[m]
        trans = update(trans, at[2:-2, m], npx.sum(vs.v[2:-2, 2:-2, :, vs.tau] * fac * mask, axis=(0, 2)))
        z_sig = update(z_sig, at[2:-2, m], npx.sum(fac * mask, axis=(0, 2)))
        return (trans, z_sig)

    trans, z_sig = for_loop(0, nlevel, loop_body, init_val=(trans, z_sig))
    trans = zonal_sum(trans)
    z_sig = zonal_sum(z_sig)

    ovt_vs.trans = ovt_vs.trans + trans

    if settings.enable_neutral_diffusion and settings.enable_skew_diffusion:
        # eddy-driven transports below isopycnals
        bolus_trans = allocate(state.dimensions, ("yu", nlevel))

        def loop_body(m, bolus_trans):
            mask = sig_loc_face > ovt_vs.sigma[m]
            bolus_trans = update(
                bolus_trans,
                at[2:-2, m],
                npx.sum(
                    npx.sum(
                        (vs.B1_gm[2:-2, 2:-2, 1:] - vs.B1_gm[2:-2, 2:-2, :-1])
                        * vs.dxt[2:-2, npx.newaxis, npx.newaxis]
                        * vs.cosu[npx.newaxis, 2:-2, npx.newaxis]
                        * vs.maskV[2:-2, 2:-2, 1:]
                        * mask[:, :, 1:],
                        axis=2,
                    )
                    + vs.B1_gm[2:-2, 2:-2, 0]
                    * vs.dxt[2:-2, npx.newaxis]
                    * vs.cosu[npx.newaxis, 2:-2]
                    * vs.maskV[2:-2, 2:-2, 0]
                    * mask[:, :, 0],
                    axis=0,
                ),
            )
            return bolus_trans

        bolus_trans = for_loop(0, nlevel, loop_body, init_val=bolus_trans)
        bolus_trans = zonal_sum(bolus_trans)

    # streamfunction on geopotentials
    ovt_vs.vsf_depth = update_add(
        ovt_vs.vsf_depth,
        at[2:-2, :],
        npx.cumsum(
            zonal_sum(
                npx.sum(
                    vs.dxt[2:-2, npx.newaxis, npx.newaxis]
                    * vs.cosu[npx.newaxis, 2:-2, npx.newaxis]
                    * vs.v[2:-2, 2:-2, :, vs.tau]
                    * vs.maskV[2:-2, 2:-2, :],
                    axis=0,
                )
            )
            * vs.dzt[npx.newaxis, :],
            axis=1,
        ),
    )

    if settings.enable_neutral_diffusion and settings.enable_skew_diffusion:
        # streamfunction for eddy driven velocity on geopotentials
        ovt_vs.bolus_depth = update_add(
            ovt_vs.bolus_depth,
            at[2:-2, :],
            zonal_sum(
                npx.sum(
                    vs.dxt[2:-2, npx.newaxis, npx.newaxis]
                    * vs.cosu[npx.newaxis, 2:-2, npx.newaxis]
                    * vs.B1_gm[2:-2, 2:-2, :],
                    axis=0,
                )
            ),
        )

    # interpolate from isopycnals to depth
    ovt_vs.vsf_iso = update_add(
        ovt_vs.vsf_iso, at[2:-2, :], _interpolate_depth_coords(z_sig[2:-2, :], trans[2:-2, :], ovt_vs.zarea[2:-2, :])
    )

    if settings.enable_neutral_diffusion and settings.enable_skew_diffusion:
        ovt_vs.bolus_iso = update_add(
            ovt_vs.bolus_iso,
            at[2:-2, :],
            _interpolate_depth_coords(z_sig[2:-2, :], bolus_trans[2:-2, :], ovt_vs.zarea[2:-2, :]),
        )

    return KernelOutput(
        trans=ovt_vs.trans,
        vsf_depth=ovt_vs.vsf_depth,
        vsf_iso=ovt_vs.vsf_iso,
        bolus_iso=ovt_vs.bolus_iso,
        bolus_depth=ovt_vs.bolus_depth,
    )


def zonal_sum(arr):
    return global_sum(arr, axis=0)
