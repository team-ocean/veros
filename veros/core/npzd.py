"""
Contains veros methods for handling bio- and geochemistry
(currently only simple bio)
"""
import numpy as np  # NOTE np is already defined somehow
from .. import veros_method
# from . import numerics, utilities

@veros_method
def npzd(vs):
    """
    Integrate biochemistry: phytoplankton, zooplankton, detritus, po4
    """

    # Import and export of detritus in currently evaluated layer
    detritus_export = np.zeros((vs.detritus.shape[:2]))
    detritus_import = np.zeros_like(detritus_export)

    # Integrated phytplankton
    phyto_integrated = np.zeros_like(vs.phytoplankton[:, :, 0])

    # TODO the ones below should be set to settings just once..
    # sinking speed of detritus
    # can't sink beyond bottom
    wd = np.empty_like(vs.phytoplankton)
    wd[:, :] = (vs.wd0 + vs.mw * np.where(-vs.zw[::-1] < vs.mwz, -vs.zw[::-1], vs.mwz)) / vs.dzt[::-1]
    wd *= np.logical_not(vs.maskT)  # TODO: Is this the correct way to use the mask?
                                    # The intention is to zero any element not in water
    nbio = vs.dt_mom // vs.dt_bio
    # ztt = depth at top of grid box
    ztt = vs.zw[::-1]

    # copy the incomming light forcing
    # we use this variable to decrease the available light
    # down the water column
    swr = vs.swr.copy()
    vs.saturation_constant_P = vs.saturation_constant_N * vs.redfield_ratio_PN

    for k in range(vs.phytoplankton.shape[2]):
        # print("layer", k)

        # incomming radiation at layer
        swr = swr * np.exp(- vs.light_attenuation_phytoplankton * phyto_integrated)
        phyto_integrated = np.maximum(vs.phytoplankton[:, :, k], vs.trcmin) * vs.dzt[::-1][k]
        phyto_integrated += np.maximum(vs.diazotroph[:, :, k], vs.trcmin) * vs.dzt[::-1][k]
        grid_light = swr * np.exp(ztt[k] * vs.rctheta)  # light at top of grid box

        # calculate detritus import pr time step from layer above
        detritus_import = detritus_export / vs.dzt[::-1][k] / vs.dt_bio

        # reset export
        detritus_export[:, :] = 0

        # TODO What is this?
        bct = vs.bbio ** (vs.cbio * vs.temp[:, :, k, vs.taum1])
        bctz = vs.bbio ** (vs.cbio * np.minimum(vs.temp[:, :, k, vs.taum1], 20))  # TODO: Remove magic number
        nud = vs.nud0
        nudon = vs.nudon0
        nudop = vs.nudop0


        """
        --------------------------------------
        Call the npzd ecosystem dynamics model
        """


        tracers = {"po4": vs.po4[:, :, k],
                   "phytoplankton": vs.phytoplankton[:, :, k],
                   "zooplankton": vs.zooplankton[:, :, k],
                   "detritus": vs.detritus[:, :, k],
                   "no3": vs.no3[:, :, k],
                   "DOP": vs.dop[:, :, k],
                   "DON": vs.dop[:, :, k],
                   "diazotroph": vs.diazotroph[:, :, k]}

        # flags to prevent more outgoing flux than available capacity - stability?
        flags = {"po4": True, "phytoplankton": True, "zooplankton": True, "detritus": True, "no3": True, "DOP": True, "diazotroph": True}
        zooplankton_preferences = {"phytoplankton": vs.zprefP, "zooplankton": vs.zprefZ, "detritus": vs.zprefDet, "diazotroph": vs.zprefD}


        # Set flags and tracers based on minimum concentration
        update_flags_and_tracers(vs, flags, tracers, refresh=True)

        # ----------------------------
        # Fotosynthesis
        # After Evans & Parslow (1985)
        # ----------------------------
        light_attenuation = (vs.light_attenuation_water + vs.light_attenuation_phytoplankton * (tracers["phytoplankton"] + tracers["diazotroph"])) * vs.dzt[::-1][k]
        f1 = np.exp(-light_attenuation)

        # TODO what are these things supposed to be?
        jmax = vs.abio_P * bct
        gd = jmax * vs.dt_mom # *(vs.dt_mom / 86400)
        avej = avg_J(vs, f1, gd, grid_light, light_attenuation)

        # TODO remove magic number 2.6
        jmax_D = np.maximum(0, vs.abio_P * (bct - 2.6)) * vs.jdiar
        # TODO remove magic number 1e-14
        gd_D = np.maximum(1e-14, jmax_D) * vs.dt_mom#* (vs.dt_mom / 84600)
        avej_D = avg_J(vs, f1, gd_D, grid_light, light_attenuation)

        # Maximum grazing rate is a function of temperature
        # bctz sets an upper limit on effects of temperature on grazing
        # phytoplankon growth rates bct are unlimited
        # FIXME: These quantities are not well defined or explained...
        gmax = vs.gbio * bctz
        nupt = vs.nupt0 * bct
        nupt_D = vs.nupt0_D * bct


        for _ in range(nbio):

            # TODO: What do these do? What do the variables mean?
            # TODO saturation / redfield is constant -> calculate only once
            # TODO separate into function (until Michaelis-Menten denominator
            limP_dop = vs.hdop * tracers["DOP"] / (vs.saturation_constant_P + tracers["DOP"])
            limP_po4 = tracers["po4"] / (vs.saturation_constant_P + tracers["po4"])

            sat_red = vs.saturation_constant_N / vs.redfield_ratio_PN

            dop_uptake_flag = limP_dop > limP_po4
            # nitrate limitation
            limP = tracers["po4"] / (sat_red + tracers["po4"])
            u_P = np.minimum(avej, jmax * limP)
            u_P = np.minimum(u_P, jmax * tracers["no3"] / (sat_red + tracers["no3"]))
            u_D = np.minimum(avej_D, jmax_D * limP)

            dop_uptake_D_flag = dop_uptake_flag  # TODO should this be here?

            po4P = jmax * tracers["po4"] / (sat_red + tracers["po4"])
            no3_P = jmax * tracers["no3"] / (sat_red + tracers["no3"])
            po4_D = jmax_D * tracers["po4"] / (sat_red + tracers["po4"])

            # Net primary production
            npp = u_P * tracers["phytoplankton"]
            npp_D = np.maximum(0, u_D * tracers["diazotroph"])

            dop_uptake = npp * dop_uptake_flag


            # Michaelis-Menten denominator
            thetaZ = sum([pref_score * tracers[preference] for preference, pref_score in zooplankton_preferences.items()]) + vs.saturation_constant_Z_grazing * vs.redfield_ratio_PN

            # Ingestion by zooplankton based on preference
            ingestion = {preference: pref_score / thetaZ for preference, pref_score in zooplankton_preferences.items()}

            # Grazing is based on availability of food and eaters??
            grazing = {preference: gmax * ingestion[preference] * tracers[preference] * tracers["zooplankton"] for preference in zooplankton_preferences}

            mortality = {"phytoplankton": vs.specific_mortality_phytoplankton * tracers["phytoplankton"],
                         "diazotroph": vs.specific_mortality_diazotroph * tracers["diazotroph"],
                         "zooplankton": vs.quadric_mortality_zooplankton * tracers["zooplankton"] ** 2,
                         }

            morpt = nupt * tracers["phytoplankton"]  # fast-recycling mortality of phytoplankton
            morpt_D = nupt_D * tracers["diazotroph"]  # fast-recycling of diazotrophs
            remineralization_detritus = nud * bct * tracers["detritus"]  # remineralization of detritus

            recycled_don = nudon * bct * tracers["DON"]
            recycled_dop = nudop * bct * tracers["DOP"]
            no3_uptake_D = (0.5 + 0.5 * np.tanh(tracers["no3"] - 5.0)) * npp_D  # nitrate uptake
            dop_uptake_D = npp_D * dop_uptake_D_flag


            expo = wd[:, :, k] * tracers["detritus"]  # temporary detritus export


            # multiply by flags to ensure stability / not loosing more than there is
            for preference in zooplankton_preferences:
                grazing[preference] *= flags[preference] * flags["zooplankton"]

            for plankton in mortality:
                mortality[plankton] *= flags[plankton]


            # phytoplankton and diazotrophs change feeding based on availability
            npp *= flags["no3"] * (dop_uptake_flag + (1 - dop_uptake_flag) * flags["po4"])
            npp_D *= dop_uptake_D_flag * flags["DOP"] + (1 - dop_uptake_D_flag) * flags["po4"]  # TODO something wacky is going on here?


            remineralization_detritus *= flags["detritus"]
            morpt_D *= flags["diazotroph"]
            no3_uptake_D *= flags["no3"]
            recycled_don *= flags["DON"]
            recycled_dop *= flags["DOP"]

            expo *= flags["detritus"]

            """
            Grazing is split into several parts:
            Digestion: How much of the eaten material is put into growing the population
            Excretion: Eaten metrial is disposed as nutrients. This amount is not available for growing
            Sloppy feeding: Material is not consumed by zooplankton and is lost as detritus
            """

            digestion = {preference: vs.assimilation_efficiency * amount_grazed for preference, amount_grazed in grazing.items()}
            digestion["diazotroph"] *= (vs.redfield_ratio_PN / vs.diazotroph_NP)
            digestion_total = sum(digestion.values())  # TOD np.sum?

            excretion = {preference: vs.assimilation_efficiency * (1 - vs.zooplankton_growth_efficiency) * amount_grazed for preference, amount_grazed in grazing.items()}
            excretion["diazotroph"] *= (vs.redfield_ratio_PN / vs.diazotroph_NP)
            excretion_total = sum(excretion.values())  # TODO np.sum?

            nr_excr_D = vs.assimilation_efficiency * grazing["diazotroph"] * (1 - vs.redfield_ratio_PN / vs.diazotroph_NP) + (1 - vs.assimilation_efficiency) * grazing["diazotroph"] * (1 - vs.redfield_ratio_PN / vs.diazotroph_NP)
            # FIXME is it just me, or is there too much math here?

            sloppy_feeding = {preference: (1 - vs.assimilation_efficiency) * amount_grazed for preference, amount_grazed in grazing.items()}
            sloppy_feeding["diazotroph"] *= (vs.redfield_ratio_PN / vs.diazotroph_NP)
            sloppy_feeding_total = sum(sloppy_feeding.values())

            """
            Model dynamics:
            """

            tracers["po4"][:, :] += vs.dt_bio * (vs.redfield_ratio_PN * (remineralization_detritus + excretion_total - npp + (1 - vs.dfrt) * morpt - (npp - dop_uptake)) + vs.diazotroph_NP * (morpt_D - (npp_D - dop_uptake_D)) + recycled_dop)

            tracers["phytoplankton"][:, :] += vs.dt_bio * (npp - mortality["phytoplankton"] - grazing["phytoplankton"] - morpt)

            tracers["zooplankton"][:, :] += vs.dt_bio * (digestion_total - mortality["zooplankton"] - grazing["zooplankton"] - excretion_total)

            tracers["detritus"][:, :] += vs.dt_bio * ((1 - vs.dfr) * mortality["phytoplankton"] + sloppy_feeding_total + mortality["zooplankton"] - remineralization_detritus - grazing["detritus"] + mortality["diazotroph"] * (vs.redfield_ratio_PN / vs.diazotroph_NP) - expo + detritus_import)  # TODO simplify mortality to use sum

            tracers["diazotroph"][:, :] += vs.dt_bio * (npp_D - mortality["diazotroph"] - morpt_D - grazing["diazotroph"])

            tracers["DON"][:, :] += vs.dt_bio * (vs.dfr * mortality["phytoplankton"] + vs.dfrt * morpt - recycled_don)

            tracers["DOP"][:, :] += vs.dt_bio * (vs.redfield_ratio_PN * (vs.dfr * mortality["phytoplankton"] + vs.dfrt * morpt - dop_uptake) - vs.diazotroph_NP * dop_uptake_D - recycled_dop)

            tracers["no3"][:, :] += vs.dt_bio * (excretion_total + remineralization_detritus + (1 - vs.dfrt) * morpt - npp + morpt_D - no3_uptake_D + recycled_don + nr_excr_D + mortality["diazotroph"] * (1 - (vs.redfield_ratio_PN / vs.diazotroph_NP)))


            # update export from layer and flags
            detritus_export[:, :] += expo * vs.dt_bio  # NOTE the additional factor is only for easier debugging. It could be removed by removing the same factor in the import

            ## TODO: Should it be here?
            detritus_export = np.minimum(tracers["detritus"], detritus_export)

            update_flags_and_tracers(vs, flags, tracers)


        """
        End the npzd ecosystem dynamic model
        --------------------------------------
        """

        # Calculate total export to get total import for next layer
        detritus_export[:, :] *= 1.0/nbio
        detritus_export *= vs.dzt[::-1][k]

        # ---------------------------------------------------
        # Set source/sink terms
        # ---------------------------------------------------

        # NOTE Added to get values back into veros
        # TODO can't we just work directly on the values?
        vs.po4[:, :, k] = tracers["po4"][:, :]
        vs.phytoplankton[:, :, k] = tracers["phytoplankton"][:, :]
        vs.zooplankton[:, :, k] = tracers["zooplankton"][:, :]
        vs.detritus[:, :, k] = tracers["detritus"][:, :]
        vs.dop[:, :, k] = tracers["DOP"][:, :]
        vs.don[:, :, k] = tracers["DON"][:, :]
        vs.no3[:, :, k] = tracers["no3"][:, :]
        vs.diazotroph[:, :, k] = tracers["diazotroph"][:, :]


@veros_method
def avg_J(vs, f1, gd, grid_light, light_attenuation):
    """Average J"""
    u1 = np.maximum(grid_light / gd, 1e-6)  # TODO remove magic number 1e-6
    u2 = u1 * f1

    # NOTE: There is an approximation here: u1 < 20 WTF? Why 20?
    phi1 = np.log(u1 + np.sqrt(1 + u1**2)) - (np.sqrt(1 + u1) - 1) / u1
    phi2 = np.log(u2 + np.sqrt(1 + u2**2)) - (np.sqrt(1 + u2) - 1) / u2

    return gd * (phi1 - phi2) / light_attenuation


@veros_method
def update_flags_and_tracers(vs, flags, tracers, refresh=False):
    """Set flags"""

    for key in tracers:
        keep = flags[key] if not refresh else True  # if the flag was already false, keep it false
        flags[key] = (tracers[key] > vs.trcmin) * keep

    for key in tracers:
        tracers[key] = np.maximum(tracers[key], vs.trcmin)
